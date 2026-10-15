"""A process-wide catalog of active library instrumentation.

Every ``logfire.instrument_*`` method registers itself here. The catalog is
read-only metadata about instrumentation setup: span creation never consults
it, so instrumented code keeps the same behavior and performance.
"""

from __future__ import annotations

import contextlib
import contextvars
import importlib.util
import inspect
import threading
import weakref
from collections.abc import Callable, Collection, Generator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, cast

from .stack_info import is_user_code, warn_at_user_stacklevel

if TYPE_CHECKING:
    from .main import Logfire

DuplicatePolicy = Literal['warn', 'ignore', 'error']

_VALID_POLICIES: tuple[DuplicatePolicy, ...] = ('warn', 'ignore', 'error')

# Directory containing the logfire package, used to recognize our own frames.
# Compare un-resolved paths: frame filenames and these prefixes may go through
# different symlink/junction views (e.g. uv-managed interpreters).
_PACKAGE_DIR = Path(__file__).absolute().parent.parent
# Standard library frames (context managers, import machinery) between a
# registration call and its caller are plumbing, not the caller.
_STDLIB_DIR = Path(inspect.__file__).absolute().parent
_PARAM_VALUE_MAX_LENGTH = 500


def _qualname(obj: Any) -> str:
    name = getattr(obj, '__qualname__', None)
    if name is not None:
        return name
    name = getattr(obj, '__name__', None)
    if name is not None:
        return name
    return type(obj).__qualname__


@dataclass(frozen=True)
class InstrumentationOrigin:
    """Where an instrumentation call came from.

    Attributes:
        called_by: ``'manual'`` for a direct call from user code, otherwise the
            dotted module name of the framework that made the call on the user's behalf.
        file: The file containing the first frame outside the logfire package.
        line: The line number in `file`.
        function: The function executing at `file`/`line`.
        user_file: The file of the nearest frame in user code, if that's a different frame.
        user_line: The line number in `user_file`.
        user_function: The function executing at `user_file`/`user_line`.
    """

    called_by: str
    file: str
    line: int
    function: str
    user_file: str | None = None
    user_line: int | None = None
    user_function: str | None = None

    def __str__(self) -> str:
        if self.user_file is None:
            return f'{self.file}:{self.line} in {self.function}'
        return (
            f'{self.file}:{self.line} in {self.function} '
            f'(user code: {self.user_file}:{self.user_line} in {self.user_function})'
        )


@dataclass(frozen=True)
class InstrumentationInfo:
    """A read-only snapshot of one catalog entry."""

    id: int
    library: str
    target_type: str | None
    params: Mapping[str, Any]
    started_at: datetime
    origin: InstrumentationOrigin
    revertible: bool


class DuplicateInstrumentationError(RuntimeError):
    """Raised by the ``'error'`` duplicate policy when a target is already instrumented."""

    def __init__(self, existing: InstrumentationInfo, origin: InstrumentationOrigin) -> None:
        self.existing = existing
        super().__init__(
            f'Refusing to instrument {existing.library}{_target_suffix(existing.target_type)} again: '
            f'instrumentation #{existing.id} is already active, registered at '
            f'{existing.started_at:%Y-%m-%dT%H:%M:%SZ} from {existing.origin}. '
            f'The new call came from {origin}.'
        )


class InstrumentationNotRevertibleError(RuntimeError):
    """Raised when trying to uninstrument an entry without an existing undo mechanism."""


class _TargetKey(NamedTuple):
    library: str
    target_type: str | None
    object_id: int | None


@dataclass
class _Entry:
    id: int
    key: _TargetKey
    target_ref: Callable[[], Any] | None
    params: Mapping[str, Any]
    started_at: datetime
    origin: InstrumentationOrigin
    revertible: bool
    logfire_ref: Callable[[], Logfire | None] | None
    revert: Callable[[], None] | None = None
    is_active_probe: Callable[[], bool] | None = None
    active: bool = True

    def info(self) -> InstrumentationInfo:
        return InstrumentationInfo(
            id=self.id,
            library=self.key.library,
            target_type=self.key.target_type,
            params=dict(self.params),
            started_at=self.started_at,
            origin=self.origin,
            revertible=self.revertible,
        )


def _target_suffix(target_type: str | None) -> str:
    return '' if target_type is None else f' ({target_type})'


def _dotted_name(cls: type) -> str:
    return cls.__qualname__ if cls.__module__ == 'builtins' else f'{cls.__module__}.{cls.__qualname__}'


def _target_type_label(target: object) -> str:
    if isinstance(target, ModuleType):
        return target.__name__
    if isinstance(target, type):
        return _dotted_name(target)
    if isinstance(target, tuple):
        if not target:
            return 'tuple'
        tuple_items = cast(tuple[Any, ...], target)
        labels = list(dict.fromkeys(_target_type_label(item) for item in tuple_items))
        return f'tuple[{", ".join(labels)}]'
    if inspect.isfunction(target) or inspect.ismethod(target):
        module = getattr(target, '__module__', '')
        qualname = _qualname(target)
        return qualname if not module or module == 'builtins' else f'{module}.{qualname}'
    return _dotted_name(type(target))


def _sanitize_param(value: Any, depth: int = 0) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, str) and len(value) > _PARAM_VALUE_MAX_LENGTH:
            return value[: _PARAM_VALUE_MAX_LENGTH - 3] + '...'
        return value
    if isinstance(value, type):
        return _dotted_name(value)
    if isinstance(value, ModuleType):
        return value.__name__
    if callable(value):
        module = getattr(value, '__module__', '')
        label = f'<callable {_qualname(value)}>'
        if module and module != 'builtins':
            label = f'<callable {module}.{_qualname(value)}>'
        return label
    if depth < 3:
        if isinstance(value, Mapping):
            mapping = cast(Mapping[Any, Any], value)
            return {str(key): _sanitize_param(item, depth + 1) for key, item in mapping.items()}
        if isinstance(value, (list, tuple, set, frozenset)):
            items = cast(Collection[Any], value)
            return [_sanitize_param(item, depth + 1) for item in items]
    value_type = cast(type[Any], type(value))
    return f'<{_dotted_name(value_type)}>'


def _sanitize_params(params: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _sanitize_param(value) for key, value in params.items() if not str(key).startswith('_')}


def _frame_is_plumbing(frame: Any) -> bool:
    """Whether the frame is logfire itself or standard-library import/context-manager plumbing."""
    try:
        path = Path(frame.f_code.co_filename).absolute()
    except (ValueError, OSError):
        return True
    return path.is_relative_to(_PACKAGE_DIR) or path.is_relative_to(_STDLIB_DIR)


def _display_path(filename: str) -> str:
    """A file path for display, relative to the current directory when possible."""
    path = Path(filename).absolute()
    with contextlib.suppress(ValueError, OSError):
        cwd = Path.cwd().absolute()
        if path.is_relative_to(cwd):
            return str(path.relative_to(cwd))
    return str(path)


def _code_function_name(code: Any) -> str:
    qualname: str | None = getattr(code, 'co_qualname', None)
    return qualname if qualname is not None else code.co_name


def _get_origin() -> InstrumentationOrigin:
    """Find the call site: the first frame outside logfire and the nearest user frame."""
    frame = inspect.currentframe()
    caller_frame: Any = None
    user_frame: Any = None
    while frame is not None:
        if not _frame_is_plumbing(frame):
            if caller_frame is None:
                caller_frame = frame
            if is_user_code(frame.f_code):
                user_frame = frame
                break
        frame = frame.f_back
    if caller_frame is None:  # pragma: no cover
        # Every public call goes through a logfire frame, so this can't happen in practice.
        return InstrumentationOrigin('manual', '<unknown>', 0, '<unknown>')

    code = caller_frame.f_code
    caller_path = _display_path(code.co_filename)
    if user_frame is not None and user_frame is not caller_frame:
        user_code = user_frame.f_code
        user_path = _display_path(user_code.co_filename)
        called_by: str = caller_frame.f_globals.get('__name__', '') or 'unknown'
        return InstrumentationOrigin(
            called_by,
            caller_path,
            caller_frame.f_lineno,
            _code_function_name(code),
            user_path,
            user_frame.f_lineno,
            _code_function_name(user_code),
        )
    return InstrumentationOrigin(
        'manual',
        caller_path,
        caller_frame.f_lineno,
        _code_function_name(code),
    )


_policy_var: contextvars.ContextVar[DuplicatePolicy | None] = contextvars.ContextVar(
    'logfire_instrumentation_duplicate_policy', default=None
)

_MISSING = object()

# Lazily imported/constructed OpenTelemetry instrumentors.
_instrumentor_classes: dict[tuple[str, str], type] = {}


def _otel_instrumentor(module_path: str, class_name: str, *, required: bool = True) -> Any:
    key = (module_path, class_name)
    cls = _instrumentor_classes.get(key)
    if cls is None:
        if not required:
            try:
                if importlib.util.find_spec(module_path) is None:
                    return None
            except (ImportError, ValueError):
                # A missing parent package raises ModuleNotFoundError; partially
                # initialized packages can report a None spec as ValueError.
                return None
        module = __import__(module_path, fromlist=[class_name])
        cls = getattr(module, class_name)
        _instrumentor_classes[key] = cls
    return cls()


def otel_uninstrument_call(
    module_path: str, class_name: str, method: str = 'uninstrument', obj: Any = _MISSING
) -> Callable[[], None]:
    """Build a zero-argument revert that calls an OpenTelemetry instrumentor teardown method."""

    def revert() -> None:
        instrumentor = _otel_instrumentor(module_path, class_name)
        method_fn = getattr(instrumentor, method)
        if obj is _MISSING:
            method_fn()
        else:
            method_fn(obj)

    return revert


def otel_optional_uninstrument_call(module_path: str, class_name: str) -> Callable[[], None]:
    """Like `otel_uninstrument_call`, but a no-op when the instrumentor package isn't installed."""

    def revert() -> None:
        instrumentor = _otel_instrumentor(module_path, class_name, required=False)
        if instrumentor is not None:
            instrumentor.uninstrument()

    return revert


def otel_active_probe(module_path: str, class_name: str) -> Callable[[], bool]:
    """Build a probe that reads an OpenTelemetry instrumentor's global instrumented flag."""

    def probe() -> bool:
        instrumentor = _otel_instrumentor(module_path, class_name, required=False)
        return instrumentor is not None and bool(instrumentor.is_instrumented_by_opentelemetry)

    return probe


def object_active_probe(obj: Any, attribute: str = '_is_instrumented_by_opentelemetry') -> Callable[[], bool]:
    """Build a probe that reads an ``_is_instrumented_by_*`` marker placed on an instrumented object."""

    def probe() -> bool:
        return bool(getattr(obj, attribute, False))

    return probe


def combine_reverts(reverts: Collection[Callable[[], None]]) -> Callable[[], None]:
    def revert() -> None:
        for item in reverts:
            item()

    return revert


def any_active(probes: Collection[Callable[[], bool]]) -> Callable[[], bool]:
    def probe() -> bool:
        for item in probes:
            with contextlib.suppress(Exception):
                if item():
                    return True
        return False

    return probe


class _BoundInstrumentationContextManager(AbstractContextManager[None]):
    """A proxy undo context manager that also drops the catalog entry on exit."""

    def __init__(self, cm: AbstractContextManager[Any], state: dict[str, bool], close: Callable[[], None]):
        self._cm = cm
        self._state = state
        self._close = close

    def __enter__(self) -> None:
        self._state['entered'] = True
        self._cm.__enter__()

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        self._close()
        return False


class RegistrationHandle:
    """Configures the catalog entry for one in-progress `instrument_*` call."""

    def __init__(self, registry: _InstrumentationRegistry, entry: _Entry) -> None:
        self._registry = registry
        self._entry = entry
        self.skipped = False

    @property
    def entry(self) -> _Entry:
        return self._entry

    def set_revert(self, revert: Callable[[], None] | None) -> None:
        """Register the callable that removes this instrumentation, when it supports removal."""
        self._entry.revert = revert

    def set_active_probe(self, probe: Callable[[], bool] | None) -> None:
        """Register a probe reporting whether the underlying instrumentation is still active."""
        self._entry.is_active_probe = probe

    def mark_not_revertible(self) -> None:
        self._entry.revertible = False

    def skip(self) -> None:
        """Drop the registration because the instrumentation call turned out to be a no-op."""
        self.skipped = True
        self._registry.discard_entry(self._entry)

    def bind_context_manager(self, cm: AbstractContextManager[Any]) -> AbstractContextManager[None]:
        """Tie an existing undo context manager to this entry.

        Returns a transparent proxy context manager. When it exits (or is closed
        by `uninstrument`), the entry is removed from the catalog.
        """
        state = {'closed': False, 'entered': False}

        def inner_finished() -> bool:
            # Generator context managers expose their generator; once it's both
            # started and finalized (e.g. by garbage collection), the undo ran.
            generator = getattr(cm, 'gen', None)
            return state['entered'] and generator is not None and generator.gi_frame is None

        def probe() -> bool:
            return not state['closed'] and not inner_finished()

        def close() -> None:
            if state['closed']:
                return
            state['closed'] = True
            if not inner_finished():
                if not state['entered']:
                    # Patches are applied before the context yields, so an unentered
                    # manager is torn down by briefly entering it.
                    cm.__enter__()
                cm.__exit__(None, None, None)
            self._registry.entry_finished(self._entry)

        self._entry.revert = close
        self._entry.is_active_probe = probe

        return _BoundInstrumentationContextManager(cm, state, close)


class _InstrumentationRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: dict[int, _Entry] = {}
        self._next_id = 1
        self._global_policy: DuplicatePolicy = 'warn'

    # -- policy ------------------------------------------------------------

    @property
    def duplicate_policy(self) -> DuplicatePolicy:
        return _policy_var.get() or self._global_policy

    def set_duplicate_policy(self, policy: DuplicatePolicy) -> None:
        if policy not in _VALID_POLICIES:
            raise ValueError(f'Invalid duplicate instrumentation policy {policy!r}; expected one of {_VALID_POLICIES}.')
        self._global_policy = policy

    @contextmanager
    def on_duplicate(self, policy: DuplicatePolicy) -> Generator[None, None, None]:
        if policy not in _VALID_POLICIES:
            raise ValueError(f'Invalid duplicate instrumentation policy {policy!r}; expected one of {_VALID_POLICIES}.')
        token = _policy_var.set(policy)
        try:
            yield
        finally:
            _policy_var.reset(token)

    # -- registration ------------------------------------------------------

    def begin(
        self,
        library: str,
        target: Any,
        params: Mapping[str, Any],
        origin: InstrumentationOrigin,
        revertible: bool,
        logfire: Logfire | None,
    ) -> RegistrationHandle:
        target_type = None if target is None else _target_type_label(target)
        object_id = None if target is None else id(target)
        key = _TargetKey(library, target_type, object_id)
        target_ref: Callable[[], Any] | None = None
        if target is not None:
            with contextlib.suppress(TypeError):
                target_ref = weakref.ref(target)
        logfire_ref: Callable[[], Logfire | None] | None = None
        if logfire is not None:
            with contextlib.suppress(TypeError):
                logfire_ref = weakref.ref(logfire)

        with self._lock:
            existing = self._active_entries_for_key_locked(key)
            if existing:
                old_entry = existing[-1]
                policy = self.duplicate_policy
                if policy == 'error':
                    raise DuplicateInstrumentationError(old_entry.info(), origin)
                if policy == 'warn':
                    warn_at_user_stacklevel(
                        self._duplicate_message(old_entry),
                        UserWarning,
                    )
            entry = _Entry(
                id=self._next_id,
                key=key,
                target_ref=target_ref,
                params=params,
                started_at=datetime.now(timezone.utc),
                origin=origin,
                revertible=revertible,
                logfire_ref=logfire_ref,
            )
            self._next_id += 1
            self._entries[entry.id] = entry
            return RegistrationHandle(self, entry)

    def discard_entry(self, entry: _Entry) -> None:
        with self._lock:
            self._entries.pop(entry.id, None)
            entry.active = False

    def commit_entry(self, entry: _Entry) -> None:
        # An entry whose target object has been garbage collected is already inactive.
        if entry.target_ref is not None and entry.target_ref() is None:
            self.discard_entry(entry)

    def entry_finished(self, entry: _Entry) -> None:
        with self._lock:
            self._entries.pop(entry.id, None)
            entry.active = False

    def _probe_active_locked(self, entry: _Entry) -> bool:
        if not entry.active:
            return False
        if entry.target_ref is not None and entry.target_ref() is None:
            return False
        probe = entry.is_active_probe
        if probe is None:
            return True
        try:
            return bool(probe())
        except Exception:
            # Catalog introspection must never break instrumentation setup.
            return True

    def _active_entries_for_key_locked(self, key: _TargetKey) -> list[_Entry]:
        active: list[_Entry] = []
        for entry in list(self._entries.values()):
            if entry.key != key:
                continue
            if self._probe_active_locked(entry):
                active.append(entry)
            else:
                self._entries.pop(entry.id, None)
                entry.active = False
        return active

    def _duplicate_message(self, entry: _Entry) -> str:
        info = entry.info()
        manual = entry.origin.called_by == 'manual'
        via = '' if manual else f" via framework '{entry.origin.called_by}'"
        return (
            f'`logfire.instrument_{info.library}()` was called again on the same target'
            f'{_target_suffix(info.target_type)}, which is already instrumented. '
            f'The existing instrumentation (#{info.id}) was registered at '
            f'{info.started_at:%Y-%m-%dT%H:%M:%SZ} from {info.origin.function} '
            f'({info.origin.file}:{info.origin.line}){via}. '
            'Repeated instrumentation can create duplicate spans. '
            "Use `with logfire.instrumentations.on_duplicate('ignore'):` to allow stacking, "
            "`logfire.instrumentations.set_duplicate_policy('error')` to raise instead, "
            'or `logfire.instrumentations.uninstrument(...)` to remove the existing instrumentation first.'
        )

    # -- queries -----------------------------------------------------------

    def _prune_locked(self) -> None:
        for entry in list(self._entries.values()):
            if not self._probe_active_locked(entry):
                self._entries.pop(entry.id, None)
                entry.active = False

    def list(self, library: str | None = None) -> tuple[InstrumentationInfo, ...]:
        with self._lock:
            self._prune_locked()
            entries = sorted(self._entries.values(), key=lambda item: item.id)
            if library is not None:
                entries = [entry for entry in entries if entry.key.library == library]
            return tuple(entry.info() for entry in entries)

    def get(self, instrumentation_id: int) -> InstrumentationInfo:
        with self._lock:
            self._prune_locked()
            entry = self._entries.get(instrumentation_id)
            if entry is None:
                raise LookupError(f'No active instrumentation with id {instrumentation_id}.')
            return entry.info()

    def is_instrumented(self, library: str, target: Any = None) -> bool:
        target_type = None if target is None else _target_type_label(target)
        object_id = None if target is None else id(target)
        key = _TargetKey(library, target_type, object_id)
        with self._lock:
            return bool(self._active_entries_for_key_locked(key))

    # -- teardown ----------------------------------------------------------

    def uninstrument(self, instrumentation: int | InstrumentationInfo) -> None:
        instrumentation_id = instrumentation.id if isinstance(instrumentation, InstrumentationInfo) else instrumentation
        with self._lock:
            self._prune_locked()
            entry = self._entries.get(instrumentation_id)
            if entry is None:
                raise LookupError(f'No active instrumentation with id {instrumentation_id}.')
            if not entry.revertible or entry.revert is None:
                raise InstrumentationNotRevertibleError(
                    f'Instrumentation #{entry.id} ({entry.key.library}{_target_suffix(entry.key.target_type)}) '
                    'has no undo mechanism, so it cannot be removed through the instrumentation catalog.'
                )
            revert = entry.revert
            revert()
            self._entries.pop(entry.id, None)
            entry.active = False

    def uninstrument_all(self) -> tuple[InstrumentationInfo, ...]:
        """Revert every revertible entry; return the entries that had to be kept."""
        with self._lock:
            self._prune_locked()
            entries = sorted(self._entries.values(), key=lambda item: item.id, reverse=True)
            not_reverted: list[InstrumentationInfo] = []
            for entry in entries:
                if not entry.revertible or entry.revert is None:
                    not_reverted.append(entry.info())
                    continue
                entry.revert()
                self._entries.pop(entry.id, None)
                entry.active = False
            return tuple(not_reverted)


_REGISTRY = _InstrumentationRegistry()


@contextmanager
def instrumentation_registration(
    library: str,
    *,
    target: Any = None,
    params: Mapping[str, Any] | None = None,
    logfire: Logfire | None = None,
    revertible: bool = True,
) -> Generator[RegistrationHandle, None, None]:
    """Register one instrumentation call for its duration.

    The entry is reserved on enter (so duplicate detection runs before any
    patching) and committed when the body finishes without raising. Call
    `RegistrationHandle.skip()` when the instrumentation call didn't actually
    install anything (e.g. an unnecessary no-op call).
    """
    origin = _get_origin()
    clean_params = _sanitize_params(params or {})
    handle = _REGISTRY.begin(library, target, clean_params, origin, revertible, logfire)
    entry = handle.entry
    try:
        yield handle
    except BaseException:
        if not handle.skipped:
            _REGISTRY.discard_entry(entry)
        raise
    if not handle.skipped:
        _REGISTRY.commit_entry(entry)


class Instrumentations:
    """Read and manage the process-wide instrumentation catalog.

    Access the singleton as [`logfire.instrumentations`][logfire.instrumentations]
    or via [`Logfire.instrumentations`][logfire.Logfire.instrumentations].
    """

    @property
    def duplicate_policy(self) -> DuplicatePolicy:
        """The policy in effect: the context-scoped value or the global default."""
        return _REGISTRY.duplicate_policy

    def set_duplicate_policy(self, policy: DuplicatePolicy) -> None:
        """Set the process-wide policy used when no `on_duplicate` context is active."""
        _REGISTRY.set_duplicate_policy(policy)

    def on_duplicate(self, policy: DuplicatePolicy) -> AbstractContextManager[None]:
        """Scope the duplicate policy for instrumentation calls inside the context."""
        return _REGISTRY.on_duplicate(policy)

    def list(self, library: str | None = None) -> tuple[InstrumentationInfo, ...]:
        """Return active instrumentation entries, oldest first.

        Set `library` to only return entries for one instrumented library.
        """
        return _REGISTRY.list(library)

    def get(self, instrumentation_id: int) -> InstrumentationInfo:
        """Return one entry by id, or raise `LookupError` if it isn't active anymore."""
        return _REGISTRY.get(instrumentation_id)

    def is_instrumented(self, library: str, *, target: Any = None) -> bool:
        """Whether the given library (or one specific target) currently has an active entry."""
        return _REGISTRY.is_instrumented(library, target)

    def uninstrument(self, instrumentation: int | InstrumentationInfo) -> None:
        """Remove one catalog entry and revert the instrumentation it describes.

        Raises `InstrumentationNotRevertibleError` for entries without an undo mechanism.
        """
        _REGISTRY.uninstrument(instrumentation)

    def uninstrument_all(self) -> tuple[InstrumentationInfo, ...]:
        """Revert every entry that supports it; return the entries that could not be reverted."""
        return _REGISTRY.uninstrument_all()


instrumentations = Instrumentations()
