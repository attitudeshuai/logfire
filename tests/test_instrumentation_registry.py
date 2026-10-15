from __future__ import annotations

import builtins
import contextlib
import gc
import sys
import types
import warnings
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import requests
from opentelemetry.instrumentation.requests import RequestsInstrumentor

import logfire
from logfire._internal import instrumentation_registry as catalog
from logfire._internal.instrumentation_registry import (
    DuplicateInstrumentationError,
    InstrumentationNotRevertibleError,
    instrumentations,
)
from logfire.testing import TestExporter


class ModuleLevelCustom:
    pass


class ModuleLevelTypeTarget:
    pass


@pytest.fixture(autouse=True)
def reset_catalog() -> Iterator[None]:
    catalog._REGISTRY._entries.clear()
    catalog._REGISTRY.set_duplicate_policy('warn')
    yield
    # Entries for non-revertible integrations (asgi wrappers) can't be uninstalled, so reset by hand.
    catalog._REGISTRY._entries.clear()
    catalog._REGISTRY.set_duplicate_policy('warn')
    if RequestsInstrumentor().is_instrumented_by_opentelemetry:
        RequestsInstrumentor().uninstrument()


@contextmanager
def catch_warnings() -> Iterator[list[warnings.WarningMessage]]:
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter('always')
        yield records


def catalog_params(library: str) -> dict[str, object]:
    entries = instrumentations.list(library)
    assert len(entries) == 1
    return dict(entries[0].params)


def test_catalog_starts_empty() -> None:
    assert instrumentations.list() == ()
    assert instrumentations.list('print') == ()
    with pytest.raises(LookupError):
        instrumentations.get(123)
    assert instrumentations.is_instrumented('print') is False


def test_print_registers_and_cm_removes() -> None:
    cm = logfire.instrument_print()
    try:
        entries = instrumentations.list()
        assert len(entries) == 1
        entry = entries[0]
        assert entry.library == 'print'
        assert entry.target_type is None
        assert entry.revertible is True
        assert entry.params == {}
        assert entry.origin.called_by == 'manual'
        assert entry.origin.file.endswith('test_instrumentation_registry.py')
        assert entry.origin.function == 'test_print_registers_and_cm_removes'
        assert instrumentations.get(entry.id) == entry
        assert instrumentations.is_instrumented('print') is True
    finally:
        cm.__exit__(None, None, None)
    assert instrumentations.list() == ()


def test_print_cm_entered_and_exited() -> None:
    with logfire.instrument_print():
        assert instrumentations.list('print')
    assert instrumentations.list() == ()


def test_uninstrument_unentered_cm_restores_print() -> None:
    original_print = builtins.print
    cm = logfire.instrument_print()
    assert builtins.print is not original_print
    entry = instrumentations.list('print')[0]
    instrumentations.uninstrument(entry)
    assert builtins.print is original_print
    assert instrumentations.list() == ()
    # The returned proxy can still be "exited" afterwards without error.
    cm.__exit__(None, None, None)


def test_uninstrument_accepts_info_and_is_idempotent() -> None:
    logfire.instrument_print()
    entry = instrumentations.list('print')[0]
    instrumentations.uninstrument(entry.id)
    with pytest.raises(LookupError):
        instrumentations.uninstrument(entry.id)


def test_duplicate_warns_with_existing_registration_details() -> None:
    first = logfire.instrument_print()
    try:
        with pytest.warns(UserWarning, match='already instrumented') as records:
            second = logfire.instrument_print()
        message = str(records[0].message)
        old = instrumentations.list('print')[0]
        assert f'#{old.id}' in message
        assert old.started_at.strftime('%Y-%m-%dT%H:%M:%SZ') in message
        assert 'test_duplicate_warns_with_existing_registration_details' in message
        assert instrumentations.list('print') == (old, instrumentations.list('print')[1])
        second.__exit__(None, None, None)
    finally:
        first.__exit__(None, None, None)


def test_duplicate_policy_ignore_stacks_silently() -> None:
    logfire.instrument_print()
    try:
        with catch_warnings() as records:
            with instrumentations.on_duplicate('ignore'):
                assert instrumentations.duplicate_policy == 'ignore'
                logfire.instrument_print()
                logfire.instrument_print()
        assert not records
        assert instrumentations.duplicate_policy == 'warn'
        assert len(instrumentations.list('print')) == 3
    finally:
        instrumentations.uninstrument_all()


def test_duplicate_policy_error_raises_before_patching() -> None:
    logfire.instrument_print()
    try:
        with instrumentations.on_duplicate('error'):
            with pytest.raises(DuplicateInstrumentationError):
                logfire.instrument_print()
        # The refused call must not have installed another patch/entry.
        assert len(instrumentations.list('print')) == 1
    finally:
        instrumentations.uninstrument_all()


def test_set_duplicate_policy_and_invalid_values() -> None:
    instrumentations.set_duplicate_policy('ignore')
    assert instrumentations.duplicate_policy == 'ignore'
    with catch_warnings():
        logfire.instrument_print()
        logfire.instrument_print()
    instrumentations.set_duplicate_policy('warn')
    with pytest.raises(ValueError, match='Invalid duplicate'):
        instrumentations.set_duplicate_policy('bogus')  # type: ignore[arg-type]
    with pytest.raises(ValueError, match='Invalid duplicate'):
        with instrumentations.on_duplicate('bogus'):  # type: ignore[arg-type]
            pass
    instrumentations.uninstrument_all()


def test_stale_entry_from_out_of_band_uninstrument_is_silently_replaced() -> None:
    logfire.instrument_requests()
    assert RequestsInstrumentor().is_instrumented_by_opentelemetry
    assert catalog_params('requests')['request_hook'] is False
    # Something else (another tool, test teardown) tears the OTel patch down directly.
    RequestsInstrumentor().uninstrument()
    with catch_warnings() as records:
        logfire.instrument_requests()
    assert not records
    assert RequestsInstrumentor().is_instrumented_by_opentelemetry
    assert len(instrumentations.list('requests')) == 1
    instrumentations.uninstrument_all()
    assert not RequestsInstrumentor().is_instrumented_by_opentelemetry


def test_requests_spans_stop_after_uninstrument_all(exporter: TestExporter) -> None:
    def send(self: object, request: requests.Request, **kwargs: object) -> requests.Response:
        response = requests.Response()
        response.status_code = 200
        response.headers = request.headers
        return response

    with pytest.MonkeyPatch.context() as monkeypatch:
        # Patch send first: RequestsInstrumentor wraps Session.send itself.
        monkeypatch.setattr(requests.Session, 'send', send)
        logfire.instrument_requests()

        with logfire.span('parent'):
            requests.get('https://example.org/foo')
        assert any(span['name'] == 'GET' for span in exporter.exported_spans_as_dict())

        instrumentations.uninstrument_all()
        exporter.clear()
        with logfire.span('parent two'):
            requests.get('https://example.org/bar')
        assert not any(span['name'] == 'GET' for span in exporter.exported_spans_as_dict())


def test_non_revertible_entry_is_listed_and_refused(exporter: TestExporter) -> None:
    async def app(scope: object, receive: object, send: object) -> None:
        return None

    wrapped = logfire.instrument_asgi(app)
    assert wrapped is not app
    [entry] = instrumentations.list('asgi')
    assert entry.revertible is False
    assert entry.target_type is not None and entry.target_type.endswith('.app')
    with pytest.raises(InstrumentationNotRevertibleError):
        instrumentations.uninstrument(entry)
    [remaining] = instrumentations.uninstrument_all()
    assert remaining == entry


def test_non_revertible_unique_targets_do_not_duplicate_warn() -> None:
    async def app_one(scope: object, receive: object, send: object) -> None: ...
    async def app_two(scope: object, receive: object, send: object) -> None: ...

    with catch_warnings() as records:
        logfire.instrument_asgi(app_one)
        logfire.instrument_asgi(app_two)
    assert not records
    assert len(instrumentations.list('asgi')) == 2


def test_pydantic_off_skips_registration() -> None:
    logfire.instrument_pydantic('off')
    assert instrumentations.list('pydantic') == ()


def test_pydantic_entry_reverts_plugin() -> None:
    from logfire.integrations.pydantic import get_pydantic_plugin_config, set_pydantic_plugin_config

    set_pydantic_plugin_config(None)
    logfire.instrument_pydantic('metrics', include=['inc'], exclude=('exc',))
    [entry] = instrumentations.list('pydantic')
    assert entry.params == {'record': 'metrics', 'include': ['inc'], 'exclude': ['exc']}
    instrumentations.uninstrument(entry)
    assert instrumentations.list() == ()
    # Config is back to the default (no explicitly installed plugin).
    import logfire.integrations.pydantic as pydantic_integration

    assert pydantic_integration._pydantic_plugin_config_value is None
    assert get_pydantic_plugin_config() is not None


def test_mcp2_noop_is_not_catalogged() -> None:
    try:
        import mcp.shared.session  # noqa: F401
    except ImportError:
        with pytest.warns(UserWarning, match='unnecessary'):
            logfire.instrument_mcp()
        assert instrumentations.list('mcp') == ()
    else:
        pytest.skip('mcp 1 is installed; the call patches session classes.')


def test_failure_during_installation_discards_entry() -> None:
    def boom() -> None:
        raise RuntimeError('kaboom')

    with pytest.raises(RuntimeError, match='kaboom'):
        with catalog.instrumentation_registration('never', params={'a': 1}):
            assert instrumentations.is_instrumented('never')
            boom()
    assert instrumentations.list('never') == ()


def test_skipped_registration_leaves_no_entry() -> None:
    with catalog.instrumentation_registration('noop') as handle:
        assert instrumentations.is_instrumented('noop')
        handle.skip()
    assert instrumentations.list('noop') == ()


def test_registration_carries_sanitized_params() -> None:
    def hook() -> None: ...

    with catalog.instrumentation_registration(
        'demo',
        params={
            'flag': True,
            'number': 3,
            'none': None,
            'long': 'x' * 600,
            'cls': ModuleLevelCustom,
            'fn': hook,
            'nested': {'list': [1, {'deep': ModuleLevelCustom()}]},
            '_private': 'dropped',
        },
    ):
        [entry] = instrumentations.list('demo')
        params = entry.params
        assert params['flag'] is True
        assert params['number'] == 3
        assert params['none'] is None
        assert params['long'].endswith('...')
        assert len(params['long']) == 500
        assert params['cls'] == 'tests.test_instrumentation_registry.ModuleLevelCustom'
        assert params['fn'].startswith('<callable') and params['fn'].endswith('hook>')
        assert params['nested'] == {'list': [1, {'deep': '<tests.test_instrumentation_registry.ModuleLevelCustom>'}]}
        assert '_private' not in params


def test_target_type_labels() -> None:
    assert catalog._target_type_label(ModuleLevelTypeTarget) == (
        'tests.test_instrumentation_registry.ModuleLevelTypeTarget'
    )
    assert catalog._target_type_label(ModuleLevelTypeTarget()).startswith(
        'tests.test_instrumentation_registry.ModuleLevelTypeTarget'
    )
    assert catalog._target_type_label(sys) == 'sys'
    assert catalog._target_type_label(()) == 'tuple'
    assert catalog._target_type_label((int, str)) == 'tuple[int, str]'
    assert catalog._target_type_label(catalog_params) == 'tests.test_instrumentation_registry.catalog_params'


def test_framework_call_is_attributed_to_framework_and_user(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    framework_file = tmp_path / 'fake_framework.py'
    framework_file.write_text(
        'import logfire\n\n\ndef setup():\n    logfire.instrument_requests()\n',
        encoding='utf-8',
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    logfire.add_non_user_code_prefix(str(tmp_path))
    sys.modules.pop('fake_framework', None)
    import fake_framework

    fake_framework.setup()
    [entry] = instrumentations.list('requests')
    assert entry.origin.called_by == 'fake_framework'
    assert entry.origin.file.endswith('fake_framework.py')
    assert entry.origin.function == 'setup'
    assert entry.origin.user_file is not None and entry.origin.user_file.endswith('test_instrumentation_registry.py')
    assert entry.origin.user_function == 'test_framework_call_is_attributed_to_framework_and_user'


# -- low level helpers ------------------------------------------------------


def test_otel_helpers_with_fake_instrumentor(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class FakeInstrumentor:
        _flag = False

        def instrument(self) -> None:
            type(self)._flag = True

        def uninstrument(self) -> None:
            calls.append('uninstrument')
            type(self)._flag = False

        def uninstrument_app(self, app: object) -> None:
            calls.append(f'app:{app}')

        @property
        def is_instrumented_by_opentelemetry(self) -> bool:
            return type(self)._flag

    cache_key = ('fake_otel_pkg.instrumentation.fake', 'FakeInstrumentor')
    monkeypatch.setitem(catalog._instrumentor_classes, cache_key, FakeInstrumentor)

    probe = catalog.otel_active_probe('fake_otel_pkg.instrumentation.fake', 'FakeInstrumentor')
    assert probe() is False
    FakeInstrumentor().instrument()
    assert probe() is True

    revert_global = catalog.otel_uninstrument_call('fake_otel_pkg.instrumentation.fake', 'FakeInstrumentor')
    revert_global()
    assert calls == ['uninstrument']

    app = object()
    revert_app = catalog.otel_uninstrument_call(
        'fake_otel_pkg.instrumentation.fake', 'FakeInstrumentor', 'uninstrument_app', app
    )
    revert_app()
    assert calls[-1] == f'app:{app}'


def test_otel_optional_helper_skips_missing_package(monkeypatch: pytest.MonkeyPatch) -> None:
    # A missing top-level package makes find_spec return None directly.
    assert catalog._otel_instrumentor('definitely_not_installed_package_xyz', 'X', required=False) is None
    catalog.otel_optional_uninstrument_call('definitely_not_installed_package_xyz', 'X')()
    # A missing parent package raises ModuleNotFoundError, which is treated the same way.
    assert catalog._otel_instrumentor('some.missing.instrumentor', 'X', required=False) is None

    # With required=True a resolvable (fake) module is imported and instantiated.
    fake_module = ModuleType('fake_required_pkg.instrumentation.fake')

    class RequiredFakeInstrumentor:
        pass

    fake_module.RequiredFakeInstrumentor = RequiredFakeInstrumentor
    fake_parent = ModuleType('fake_required_pkg.instrumentation')
    fake_parent.__path__ = []  # type: ignore[attr-defined]
    fake_root = ModuleType('fake_required_pkg')
    fake_root.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, 'fake_required_pkg', fake_root)
    monkeypatch.setitem(sys.modules, 'fake_required_pkg.instrumentation', fake_parent)
    monkeypatch.setitem(sys.modules, 'fake_required_pkg.instrumentation.fake', fake_module)
    instance = catalog._otel_instrumentor('fake_required_pkg.instrumentation.fake', 'RequiredFakeInstrumentor')
    assert isinstance(instance, RequiredFakeInstrumentor)


def test_any_active_swallows_probe_errors() -> None:
    def raising() -> bool:
        raise RuntimeError('boom')

    assert catalog.any_active([raising, lambda: True])() is True
    assert catalog.any_active([raising, lambda: False])() is False


def test_object_active_probe() -> None:
    obj = types.SimpleNamespace()
    probe = catalog.object_active_probe(obj)
    assert probe() is False
    obj._is_instrumented_by_opentelemetry = True
    assert probe() is True


def test_origin_str_and_plumbing_classification() -> None:
    origin = catalog.InstrumentationOrigin('manual', 'a.py', 1, 'f')
    assert str(origin) == 'a.py:1 in f'
    origin_with_user = catalog.InstrumentationOrigin('fw', 'fw.py', 2, 'g', 'u.py', 3, 'h')
    assert 'user code: u.py:3' in str(origin_with_user)

    class TestCode:
        co_filename = __file__

    class TestFrame:
        f_code = TestCode()

    # A frame in the test tree is neither logfire nor standard-library plumbing.
    assert catalog._frame_is_plumbing(TestFrame()) is False

    class StdlibCode:
        co_filename = str(Path(contextlib.__file__))

    class StdlibFrame:
        f_code = StdlibCode()

    assert catalog._frame_is_plumbing(StdlibFrame()) is True

    class BrokenCode:
        @property
        def co_filename(self) -> str:
            raise ValueError('bad path')

    class BrokenFrame:
        f_code = BrokenCode()

    assert catalog._frame_is_plumbing(BrokenFrame()) is True


def test_code_function_name_fallback() -> None:
    class Code:
        co_name = 'named'

    assert catalog._code_function_name(Code()) == 'named'
    assert catalog._code_function_name(catalog_params.__code__) == 'catalog_params'


def test_qualname_fallbacks_and_module_and_deep_params() -> None:
    class WithNameOnly:
        def __getattribute__(self, name: str) -> Any:
            if name == '__qualname__':
                raise AttributeError
            if name == '__name__':
                return 'just_a_name'
            return object.__getattribute__(self, name)

    assert catalog._qualname(WithNameOnly()) == 'just_a_name'

    class WithoutNames:
        def __getattribute__(self, name: str) -> Any:
            if name in ('__qualname__', '__name__'):
                raise AttributeError
            return object.__getattribute__(self, name)

    assert catalog._qualname(WithoutNames()).endswith('.WithoutNames')

    with catalog.instrumentation_registration(
        'misc',
        params={
            'module': sys,
            'deep': [[[{'x': ModuleLevelCustom}]]],
            'builtin_callable': print,
            'shallow_object': [object()],
        },
    ):
        [entry] = instrumentations.list('misc')
        assert entry.params['module'] == 'sys'
        # Values three levels deep aren't recursed into and are only labelled by type.
        assert entry.params['deep'] == [[['<dict>']]]
        assert entry.params['builtin_callable'] == '<callable print>'
        assert entry.params['shallow_object'] == ['<object>']


def test_combine_reverts_runs_each() -> None:
    order: list[str] = []
    catalog.combine_reverts([lambda: order.append('a'), lambda: order.append('b')])()
    assert order == ['a', 'b']


def test_mark_not_revertible_and_probe_exceptions_and_stale_pruning() -> None:
    # A raising probe must not hide an active entry.
    with catalog.instrumentation_registration('probed') as handle:
        handle.set_active_probe(lambda: (_ for _ in ()).throw(RuntimeError('probe boom')))
        assert instrumentations.is_instrumented('probed') is True

    # An entry whose probe reports inactive is pruned on access.
    with catalog.instrumentation_registration('stale') as handle:
        assert instrumentations.is_instrumented('stale')
        handle.set_active_probe(lambda: False)
        assert instrumentations.is_instrumented('stale') is False
        assert instrumentations.list('stale') == ()

    # get() also prunes stale entries and then reports them missing.
    with catalog.instrumentation_registration('staleget') as handle:
        entry_id = handle.entry.id
        handle.set_active_probe(lambda: False)
        with pytest.raises(LookupError):
            instrumentations.get(entry_id)

    # skip() followed by a failing body must not try to discard twice.
    with pytest.raises(RuntimeError, match='body failed'):
        with catalog.instrumentation_registration('skipthenfail') as handle:
            handle.skip()
            raise RuntimeError('body failed')
    assert instrumentations.list('skipthenfail') == ()

    # mark_not_revertible flips the entry flag.
    with catalog.instrumentation_registration('norevert') as handle:
        handle.mark_not_revertible()
        assert handle.entry.revertible is False
        with pytest.raises(InstrumentationNotRevertibleError):
            instrumentations.uninstrument(handle.entry.id)

    # Entries marked inactive are treated as prunable.
    with catalog.instrumentation_registration('manualinactive') as handle:
        handle.entry.active = False
        assert instrumentations.is_instrumented('manualinactive') is False


def test_dead_target_is_pruned_by_probe() -> None:
    def make_entry() -> catalog._Entry:
        target = ModuleLevelTypeTarget()
        with catalog.instrumentation_registration('deadtarget', target=target) as handle:
            return handle.entry

    entry = make_entry()
    gc.collect()
    assert instrumentations.list('deadtarget') == ()
    assert entry.active is False


def test_bound_cm_close_is_idempotent_and_tolerates_finalized_inner() -> None:
    proxy = logfire.instrument_print()
    # Unified close before entry works; closing again is a no-op.
    instrumentations.uninstrument(instrumentations.list('print')[0])
    proxy.__exit__(None, None, None)

    proxy = logfire.instrument_print()
    # Enter and finalize the inner undo generator directly, then exit the proxy.
    proxy.__enter__()
    inner = proxy._cm  # type: ignore[attr-defined]
    inner.gen.close()
    proxy.__exit__(None, None, None)
    assert instrumentations.list('print') == ()


def test_otel_optional_helper_uninstruments_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class FakeInstrumentor:
        def uninstrument(self) -> None:
            calls.append('done')

    key = ('fake_optional_pkg.instrumentation.fake', 'FakeInstrumentor')
    monkeypatch.setitem(catalog._instrumentor_classes, key, FakeInstrumentor)
    catalog.otel_optional_uninstrument_call(*key)()
    assert calls == ['done']


def test_weakref_target_pruned_on_commit() -> None:
    def make_entry() -> tuple[catalog._Entry, weakref.ref[ModuleLevelTypeTarget]]:
        target = ModuleLevelTypeTarget()
        with catalog.instrumentation_registration('shortlived', target=target) as handle:
            return handle.entry, weakref.ref(target)

    entry, target_ref = make_entry()
    gc.collect()
    assert target_ref() is None
    catalog._REGISTRY.commit_entry(entry)
    assert instrumentations.list('shortlived') == ()
