from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from threading import Lock
from typing import Any, Generic, Literal, TypeVar, cast
from weakref import WeakSet

from opentelemetry.context import Context
from opentelemetry.metrics import (
    CallbackOptions,
    CallbackT,
    Counter,
    Histogram,
    Instrument,
    Meter,
    MeterProvider,
    NoOpMeterProvider,
    ObservableCounter,
    ObservableGauge,
    ObservableUpDownCounter,
    Observation,
    UpDownCounter,
    _Gauge as Gauge,
)
from opentelemetry.sdk.metrics import MeterProvider as SDKMeterProvider
from opentelemetry.trace import get_current_span
from opentelemetry.util.types import Attributes, AttributeValue

from .tracer import _LogfireWrappedSpan  # pyright: ignore[reportPrivateUsage]
from .utils import handle_internal_errors

OverflowPolicy = Literal['overflow', 'drop']
"""What happens to a measurement once a metric's `cardinality_limit` is reached:

- `'overflow'`: measurements with unseen attribute combinations are aggregated
  into a single overflow data point carrying the `otel.metric.overflow` attribute.
- `'drop'`: such measurements are discarded entirely.

In both cases the measurements are counted and the counters can be queried at
runtime with `logfire.metric_cardinality_stats()`.
"""

#: Attribute placed on the synthetic data point aggregating over-limit
#: combinations. The attribute is reserved: the overflow data point carries no
#: other attributes, so backends can identify it as a summary record rather than
#: a real label value.
OVERFLOW_ATTRIBUTE_KEY = 'otel.metric.overflow'

#: The attributes of the overflow data point.
OVERFLOW_ATTRIBUTES: Attributes = {OVERFLOW_ATTRIBUTE_KEY: True}


@dataclasses.dataclass(frozen=True)
class MetricCardinalityStats:
    """Runtime counters of label cardinality governance for one metric.

    Returned by `logfire.metric_cardinality_stats()`. Only metrics created with a
    `cardinality_limit` are included.
    """

    metric_name: str
    """The name of the metric."""

    limit: int
    """The maximum number of distinct attribute combinations passed through."""

    overflow_policy: OverflowPolicy
    """Whether over-limit measurements go to an overflow bucket or are dropped."""

    registered_combinations: int
    """Number of distinct attribute combinations currently passed through (at most `limit`)."""

    overflow_measurements: int
    """Number of measurements merged into the overflow data point."""

    dropped_measurements: int
    """Number of measurements discarded because the limit was reached."""


class _DropMeasurement:
    """Sentinel returned by `_CardinalityLimiter.route` when a measurement must be discarded."""


_DROP_MEASUREMENT = _DropMeasurement()


def _freeze_attribute_value(value: AttributeValue) -> Any:
    """Convert an attribute value to a hashable form for use in a set key."""
    if isinstance(value, (str, bytes, int, float)):
        return value
    if isinstance(value, Sequence):
        return tuple(_freeze_attribute_value(item) for item in value)
    if isinstance(value, Mapping):
        return frozenset((key, _freeze_attribute_value(item)) for key, item in value.items())
    # `None` and any other hashable value can be used directly; non-hashable
    # values are rejected by OpenTelemetry before they reach the exporter.
    return value


def _freeze_attributes(attributes: Attributes) -> frozenset[tuple[str, Any]]:
    """Convert attributes to a hashable key, independent of dict ordering."""
    if not attributes:
        return frozenset()
    return frozenset((key, _freeze_attribute_value(value)) for key, value in attributes.items())


class _CardinalityLimiter:
    """Per-metric accounting of distinct attribute combinations.

    All state is guarded by a single lock so concurrent updates from multiple
    threads can't overwrite each other's counters. Membership checks use a hash
    set, so deciding what to do with a new combination is O(1) regardless of how
    many combinations have already been registered.
    """

    def __init__(self, metric_name: str, limit: int, overflow_policy: OverflowPolicy) -> None:
        self.metric_name = metric_name
        self.limit = limit
        self.overflow_policy: OverflowPolicy = overflow_policy
        self._combinations: set[frozenset[tuple[str, Any]]] = set()
        self._lock = Lock()
        self.registered_combinations = 0
        self.overflow_measurements = 0
        self.dropped_measurements = 0

    def route(self, attributes: Attributes) -> Attributes | _DropMeasurement:
        """Return the attributes to record for a measurement, or the drop sentinel.

        Already registered combinations pass through unchanged. New combinations
        pass through until `limit` is reached; afterwards they either go to the
        overflow data point or are dropped according to `overflow_policy`.
        """
        key = _freeze_attributes(attributes)
        with self._lock:
            if key in self._combinations:
                return attributes
            if self.registered_combinations < self.limit:
                self._combinations.add(key)
                self.registered_combinations += 1
                return attributes
            if self.overflow_policy == 'drop':
                self.dropped_measurements += 1
                return _DROP_MEASUREMENT
            self.overflow_measurements += 1
            return OVERFLOW_ATTRIBUTES

    def stats(self) -> MetricCardinalityStats:
        """Return an immutable snapshot of the current counters."""
        with self._lock:
            return MetricCardinalityStats(
                metric_name=self.metric_name,
                limit=self.limit,
                overflow_policy=self.overflow_policy,
                registered_combinations=self.registered_combinations,
                overflow_measurements=self.overflow_measurements,
                dropped_measurements=self.dropped_measurements,
            )


def _wrap_cardinality_callback(
    callback: CallbackT, limiter: _CardinalityLimiter
) -> Callable[[CallbackOptions], Iterator[Observation]]:
    """Wrap an observable instrument callback so its observations are governed."""

    def cardinality_limited_callback(options: CallbackOptions) -> Iterator[Observation]:
        # `CallbackT` allows both callables and generator objects.
        callback_result: Any = callback(options) if callable(callback) else callback
        for observation in cast(Iterable[Observation], callback_result):
            attributes = limiter.route(observation.attributes)
            if not isinstance(attributes, _DropMeasurement):
                yield Observation(observation.value, attributes, observation.context)

    return cardinality_limited_callback


# The following proxy classes are adapted from OTEL's SDK
@dataclasses.dataclass
class ProxyMeterProvider(MeterProvider):
    provider: MeterProvider
    meters: WeakSet[_ProxyMeter] = dataclasses.field(default_factory=WeakSet['_ProxyMeter'])
    lock: Lock = dataclasses.field(default_factory=Lock)
    suppressed_scopes: set[str] = dataclasses.field(default_factory=set[str])
    cardinality_limiters: dict[tuple[str, str], _CardinalityLimiter] = dataclasses.field(
        default_factory=dict[tuple[str, str], _CardinalityLimiter]
    )
    cardinality_lock: Lock = dataclasses.field(default_factory=Lock)

    def get_meter(
        self,
        name: str,
        version: str | None = None,
        schema_url: str | None = None,
        attributes: Attributes | None = None,
    ) -> _ProxyMeter:
        with self.lock:
            if name in self.suppressed_scopes:
                provider = NoOpMeterProvider()
            else:
                provider = self.provider
            inner_meter = provider.get_meter(name, version, schema_url, *[attributes] if attributes is not None else [])
            meter = _ProxyMeter(
                inner_meter,
                name,
                version,
                schema_url,
                register_cardinality_limiter=self._register_cardinality_limiter,
            )
            self.meters.add(meter)
            return meter

    def _register_cardinality_limiter(
        self, meter_name: str, instrument_name: str, limiter: _CardinalityLimiter
    ) -> None:
        """Register a per-instrument limiter so its counters can be queried at runtime."""
        with self.cardinality_lock:
            self.cardinality_limiters[(meter_name, instrument_name)] = limiter

    def cardinality_stats(self) -> dict[tuple[str, str], MetricCardinalityStats]:
        """Return snapshots of all cardinality counters, keyed by (meter name, metric name)."""
        with self.cardinality_lock:
            return {key: limiter.stats() for key, limiter in self.cardinality_limiters.items()}

    def suppress_scopes(self, *scopes: str) -> None:
        with self.lock:
            self.suppressed_scopes.update(scopes)
            for meter in self.meters:
                if meter.name in scopes:
                    meter.set_meter(NoOpMeterProvider())

    def set_meter_provider(self, meter_provider: MeterProvider) -> None:
        with self.lock:
            self.provider = meter_provider
            for meter in self.meters:
                meter.set_meter(NoOpMeterProvider() if meter.name in self.suppressed_scopes else meter_provider)

    def shutdown(self, timeout_millis: float = 30_000) -> None:
        with self.lock:
            if isinstance(self.provider, SDKMeterProvider):
                self.provider.shutdown(timeout_millis)

    def force_flush(self, timeout_millis: float = 30_000) -> None:
        with self.lock:
            if isinstance(self.provider, SDKMeterProvider):  # pragma: no branch
                self.provider.force_flush(timeout_millis)


class _ProxyMeter(Meter):
    def __init__(
        self,
        meter: Meter,
        name: str,
        version: str | None,
        schema_url: str | None,
        register_cardinality_limiter: Callable[[str, str, _CardinalityLimiter], None],
    ) -> None:
        super().__init__(name, version=version, schema_url=schema_url)
        self._lock = Lock()
        self._meter = meter
        self._instruments: WeakSet[_ProxyInstrument[Any]] = WeakSet()
        self._register_cardinality_limiter = register_cardinality_limiter

    def _make_cardinality_limiter(
        self,
        instrument_name: str,
        cardinality_limit: int | None,
        overflow_policy: OverflowPolicy,
    ) -> _CardinalityLimiter | None:
        """Validate the cardinality options and register a limiter, or return `None`."""
        if cardinality_limit is None:
            return None
        # Use `type(...) is int` so that booleans are rejected even though they're ints,
        # and so that non-integer values are rejected regardless of static annotations.
        if type(cardinality_limit) is not int:
            raise TypeError('cardinality_limit must be a positive integer')
        if cardinality_limit < 1:
            raise ValueError('cardinality_limit must be a positive integer')
        if overflow_policy not in ('overflow', 'drop'):
            raise ValueError("overflow_policy must be either 'overflow' or 'drop'")
        limiter = _CardinalityLimiter(instrument_name, cardinality_limit, overflow_policy)
        self._register_cardinality_limiter(self._name, instrument_name, limiter)
        return limiter

    def set_meter(self, meter_provider: MeterProvider) -> None:
        """Called when a real meter provider is set on the creating _ProxyMeterProvider.

        Creates a real backing meter for this instance and notifies all created
        instruments so they can create real backing instruments.
        """
        real_meter = meter_provider.get_meter(self._name, self._version, self._schema_url)

        with self._lock:
            self._meter = real_meter
            # notify all proxy instruments of the new meter so they can create
            # real instruments to back themselves
            for instrument in self._instruments:
                instrument.on_meter_set(real_meter)

    def _add_proxy_instrument(self, instrument_type: type[_ProxyInstrument[InstrumentT]], **kwargs: Any) -> InstrumentT:
        with self._lock:
            proxy = instrument_type(self._meter, **kwargs)
            self._instruments.add(proxy)
            return proxy  # pyright: ignore[reportReturnType]

    def create_counter(
        self,
        name: str,
        unit: str = '',
        description: str = '',
        *,
        cardinality_limit: int | None = None,
        overflow_policy: OverflowPolicy = 'overflow',
    ) -> Counter:
        limiter = self._make_cardinality_limiter(name, cardinality_limit, overflow_policy)
        return self._add_proxy_instrument(
            _ProxyCounter,
            name=name,
            unit=unit,
            description=description,
            cardinality_limiter=limiter,
        )

    def create_up_down_counter(
        self,
        name: str,
        unit: str = '',
        description: str = '',
        *,
        cardinality_limit: int | None = None,
        overflow_policy: OverflowPolicy = 'overflow',
    ) -> UpDownCounter:
        limiter = self._make_cardinality_limiter(name, cardinality_limit, overflow_policy)
        return self._add_proxy_instrument(
            _ProxyUpDownCounter,
            name=name,
            unit=unit,
            description=description,
            cardinality_limiter=limiter,
        )

    def create_observable_counter(
        self,
        name: str,
        callbacks: Sequence[CallbackT] | None = None,
        unit: str = '',
        description: str = '',
        *,
        cardinality_limit: int | None = None,
        overflow_policy: OverflowPolicy = 'overflow',
    ) -> ObservableCounter:
        limiter = self._make_cardinality_limiter(name, cardinality_limit, overflow_policy)
        callbacks = self._govern_callbacks(callbacks, limiter)
        return self._add_proxy_instrument(
            _ProxyObservableCounter,
            name=name,
            unit=unit,
            description=description,
            callbacks=callbacks,
            cardinality_limiter=limiter,
        )

    def create_histogram(
        self,
        name: str,
        unit: str = '',
        description: str = '',
        *,
        cardinality_limit: int | None = None,
        overflow_policy: OverflowPolicy = 'overflow',
        **kwargs: Any,
    ) -> Histogram:
        limiter = self._make_cardinality_limiter(name, cardinality_limit, overflow_policy)
        return self._add_proxy_instrument(
            _ProxyHistogram,
            name=name,
            unit=unit,
            description=description,
            cardinality_limiter=limiter,
            **kwargs,
        )

    def create_gauge(
        self,
        name: str,
        unit: str = '',
        description: str = '',
        *,
        cardinality_limit: int | None = None,
        overflow_policy: OverflowPolicy = 'overflow',
    ) -> Gauge:
        limiter = self._make_cardinality_limiter(name, cardinality_limit, overflow_policy)
        return self._add_proxy_instrument(
            _ProxyGauge,
            name=name,
            unit=unit,
            description=description,
            cardinality_limiter=limiter,
        )

    def create_observable_gauge(
        self,
        name: str,
        callbacks: Sequence[CallbackT] | None = None,
        unit: str = '',
        description: str = '',
        *,
        cardinality_limit: int | None = None,
        overflow_policy: OverflowPolicy = 'overflow',
    ) -> ObservableGauge:
        limiter = self._make_cardinality_limiter(name, cardinality_limit, overflow_policy)
        callbacks = self._govern_callbacks(callbacks, limiter)
        return self._add_proxy_instrument(
            _ProxyObservableGauge,
            name=name,
            unit=unit,
            description=description,
            callbacks=callbacks,
            cardinality_limiter=limiter,
        )

    def create_observable_up_down_counter(
        self,
        name: str,
        callbacks: Sequence[CallbackT] | None = None,
        unit: str = '',
        description: str = '',
        *,
        cardinality_limit: int | None = None,
        overflow_policy: OverflowPolicy = 'overflow',
    ) -> ObservableUpDownCounter:
        limiter = self._make_cardinality_limiter(name, cardinality_limit, overflow_policy)
        callbacks = self._govern_callbacks(callbacks, limiter)
        return self._add_proxy_instrument(
            _ProxyObservableUpDownCounter,
            name=name,
            unit=unit,
            description=description,
            callbacks=callbacks,
            cardinality_limiter=limiter,
        )

    @staticmethod
    def _govern_callbacks(
        callbacks: Sequence[CallbackT] | None, limiter: _CardinalityLimiter | None
    ) -> Sequence[CallbackT] | None:
        """Wrap observable instrument callbacks with the limiter, if there is one."""
        if limiter is None:
            return callbacks
        return [_wrap_cardinality_callback(callback, limiter) for callback in (callbacks or ())]


InstrumentT = TypeVar('InstrumentT', bound=Instrument)


class _ProxyInstrument(ABC, Generic[InstrumentT]):
    def __init__(
        self,
        meter: Meter,
        *,
        cardinality_limiter: _CardinalityLimiter | None = None,
        **kwargs: Any,
    ) -> None:
        self._kwargs = kwargs
        self._cardinality_limiter = cardinality_limiter
        self._instrument = self._create_real_instrument(meter)

    def on_meter_set(self, meter: Meter) -> None:
        """Called when a real meter is set on the creating _ProxyMeter."""
        # We don't need any locking on proxy instruments because it's OK if some
        # measurements get dropped while a real backing instrument is being
        # created.
        self._instrument = self._create_real_instrument(meter)

    @abstractmethod
    def _create_real_instrument(self, meter: Meter) -> InstrumentT:
        """Create an instance of the real instrument. Implement this."""

    @handle_internal_errors
    def _increment_span_metric(
        self, amount: float, attributes: Attributes | None = None, context: Context | None = None
    ) -> None:
        span = get_current_span(context)
        if isinstance(span, _LogfireWrappedSpan):
            span.increment_metric(self._kwargs['name'], attributes or {}, amount)

    def _route_attributes(self, attributes: Attributes | None) -> Attributes | _DropMeasurement:
        """Apply cardinality governance, or pass through when no limit was declared."""
        if self._cardinality_limiter is None:
            return attributes
        return self._cardinality_limiter.route(attributes)


class _ProxyCounter(_ProxyInstrument[Counter], Counter):
    def add(
        self,
        amount: int | float,
        attributes: Attributes | None = None,
        context: Context | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._increment_span_metric(amount, attributes, context)
        routed_attributes = self._route_attributes(attributes)
        if isinstance(routed_attributes, _DropMeasurement):
            return
        self._instrument.add(amount, routed_attributes, context, *args, **kwargs)

    def _create_real_instrument(self, meter: Meter) -> Counter:
        return meter.create_counter(**self._kwargs)


class _ProxyHistogram(_ProxyInstrument[Histogram], Histogram):
    def record(
        self,
        amount: int | float,
        attributes: Attributes | None = None,
        context: Context | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._increment_span_metric(amount, attributes, context)
        routed_attributes = self._route_attributes(attributes)
        if isinstance(routed_attributes, _DropMeasurement):
            return
        self._instrument.record(amount, routed_attributes, context, *args, **kwargs)

    def _create_real_instrument(self, meter: Meter) -> Histogram:
        return meter.create_histogram(**self._kwargs)


class _ProxyObservableCounter(_ProxyInstrument[ObservableCounter], ObservableCounter):
    def _create_real_instrument(self, meter: Meter) -> ObservableCounter:
        return meter.create_observable_counter(**self._kwargs)


class _ProxyObservableGauge(_ProxyInstrument[ObservableGauge], ObservableGauge):
    def _create_real_instrument(self, meter: Meter) -> ObservableGauge:
        return meter.create_observable_gauge(**self._kwargs)


class _ProxyObservableUpDownCounter(_ProxyInstrument[ObservableUpDownCounter], ObservableUpDownCounter):
    def _create_real_instrument(self, meter: Meter) -> ObservableUpDownCounter:
        return meter.create_observable_up_down_counter(**self._kwargs)


class _ProxyUpDownCounter(_ProxyInstrument[UpDownCounter], UpDownCounter):
    def add(
        self,
        amount: int | float,
        attributes: Attributes | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        routed_attributes = self._route_attributes(attributes)
        if isinstance(routed_attributes, _DropMeasurement):
            return
        self._instrument.add(amount, routed_attributes, *args, **kwargs)

    def _create_real_instrument(self, meter: Meter) -> UpDownCounter:
        return meter.create_up_down_counter(**self._kwargs)


class _ProxyGauge(_ProxyInstrument[Gauge], Gauge):
    def set(
        self,
        amount: int | float,
        attributes: Attributes | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        routed_attributes = self._route_attributes(attributes)
        if isinstance(routed_attributes, _DropMeasurement):
            return
        self._instrument.set(amount, routed_attributes, *args, **kwargs)

    def _create_real_instrument(self, meter: Meter):
        return meter.create_gauge(**self._kwargs)
