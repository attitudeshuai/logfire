from __future__ import annotations

import re
from dataclasses import asdict
from threading import Thread
from typing import Any

import pytest
from opentelemetry._logs import LogRecord, SeverityNumber

import logfire
from logfire._internal.constants import ATTRIBUTES_MESSAGE_TEMPLATE_KEY
from logfire.quotas import QuotaOptions, QuotaRule
from logfire.quotas._quota import (
    QUOTA_COUNT_KEY,
    QUOTA_KIND_KEY,
    QUOTA_MATCHED_NAME_KEY,
    QUOTA_RULE_KEY,
    QUOTA_STRATEGY_KEY,
    QUOTA_SUMMARY_NAME,
)
from logfire.testing import TestExporter


class FakeClock:
    """Controllable monotonic-ns clock used instead of `time.monotonic_ns`."""

    def __init__(self, now_ns: int = 0) -> None:
        self.now_ns = now_ns

    def __call__(self) -> int:
        return self.now_ns

    def advance(self, seconds: float) -> None:
        self.now_ns += int(seconds * 1_000_000_000)


def _use_fake_clock(clock: FakeClock) -> None:
    """Replace the configured quota manager's clock for deterministic window tests."""
    logfire.DEFAULT_LOGFIRE_INSTANCE.config._quota_manager.now = clock  # pyright: ignore[reportPrivateUsage]


def _exported(exporter: TestExporter) -> list[dict[str, Any]]:
    return exporter.exported_spans_as_dict()


def _names(exporter: TestExporter) -> list[str]:
    return [span['name'] for span in _exported(exporter)]


def _summaries(exporter: TestExporter) -> list[dict[str, Any]]:
    return [span for span in _exported(exporter) if span['name'] == QUOTA_SUMMARY_NAME]


def test_no_rules_means_no_quota_processing(exporter: TestExporter, config_kwargs: dict[str, Any]) -> None:
    logfire.configure(**config_kwargs)

    with logfire.span('whatever'):
        pass
    logfire.info('some log')

    # Everything is exported, accounting is empty, and the query API works before anything is configured.
    assert logfire.get_quota_stats() == []
    assert len(_exported(exporter)) == 2


def test_drop_spans(exporter: TestExporter, config_kwargs: dict[str, Any]) -> None:
    config_kwargs['quotas'] = QuotaOptions(
        rules=[QuotaRule(name_pattern='noisy', limit=3, period=60, on_exceed='drop')]
    )
    logfire.configure(**config_kwargs)
    _use_fake_clock(FakeClock())

    for _ in range(8):
        with logfire.span('noisy'):
            pass

    assert _names(exporter) == ['noisy'] * 3
    [stats] = logfire.get_quota_stats()
    assert stats.name == 'noisy'
    assert stats.matched == 8
    assert stats.allowed == 3
    assert stats.dropped == 5
    assert stats.aggregated == 0
    assert stats.summaries_emitted == 0
    # No summary records are emitted by the drop strategy: it is observably distinct from aggregate.
    assert _summaries(exporter) == []


def test_glob_pattern_matches_span_names(exporter: TestExporter, config_kwargs: dict[str, Any]) -> None:
    config_kwargs['quotas'] = QuotaOptions(
        rules=[QuotaRule(name_pattern='chat completion *', limit=1, period=60, on_exceed='drop')]
    )
    logfire.configure(**config_kwargs)
    _use_fake_clock(FakeClock())

    with logfire.span('chat completion gpt-4o'):
        pass
    with logfire.span('chat completion claude'):
        pass
    with logfire.span('unrelated'):
        pass

    assert _names(exporter) == ['chat completion gpt-4o', 'unrelated']


def test_regex_pattern(exporter: TestExporter, config_kwargs: dict[str, Any]) -> None:
    config_kwargs['quotas'] = QuotaOptions(
        rules=[QuotaRule(name_pattern=re.compile(r'^retry-\d+$'), limit=1, period=60, on_exceed='drop')]
    )
    logfire.configure(**config_kwargs)
    _use_fake_clock(FakeClock())

    for name in ('retry-1', 'retry-2', 'retry-later'):
        with logfire.span(name):
            pass

    assert _names(exporter) == ['retry-1', 'retry-later']


def test_drop_logs_by_template(exporter: TestExporter, config_kwargs: dict[str, Any]) -> None:
    config_kwargs['quotas'] = QuotaOptions(
        rules=[
            QuotaRule(
                name_pattern='request failed: {error}',
                limit=2,
                period=60,
                on_exceed='drop',
                applies_to='log',
            )
        ]
    )
    logfire.configure(**config_kwargs)
    _use_fake_clock(FakeClock())

    for i in range(5):
        logfire.error('request failed: {error}', error=f'boom-{i}')

    assert _names(exporter) == ['request failed: {error}'] * 2
    [stats] = logfire.get_quota_stats()
    assert stats.matched == 5
    assert stats.allowed == 2
    assert stats.dropped == 3


def test_applies_to_does_not_cross_kinds(exporter: TestExporter, config_kwargs: dict[str, Any]) -> None:
    config_kwargs['quotas'] = QuotaOptions(
        rules=[QuotaRule(name_pattern='same-name', limit=1, period=60, on_exceed='drop', applies_to='span')]
    )
    logfire.configure(**config_kwargs)
    _use_fake_clock(FakeClock())

    with logfire.span('same-name'):
        pass
    with logfire.span('same-name'):
        pass
    # The rule only applies to spans, so identically named logs are unaffected.
    logfire.info('same-name')
    logfire.info('same-name')

    assert _names(exporter) == ['same-name', 'same-name', 'same-name']


def test_rules_are_accounted_independently(exporter: TestExporter, config_kwargs: dict[str, Any]) -> None:
    config_kwargs['quotas'] = QuotaOptions(
        rules=[
            QuotaRule(name_pattern='a', limit=2, period=60, on_exceed='drop', name='rule-a'),
            QuotaRule(name_pattern='b', limit=1, period=60, on_exceed='drop', name='rule-b'),
        ]
    )
    logfire.configure(**config_kwargs)
    _use_fake_clock(FakeClock())

    for _ in range(4):
        with logfire.span('a'):
            pass
    for _ in range(3):
        with logfire.span('b'):
            pass
    with logfire.span('c'):
        pass

    by_name = {stats.name: stats for stats in logfire.get_quota_stats()}
    assert (by_name['rule-a'].matched, by_name['rule-a'].allowed, by_name['rule-a'].dropped) == (4, 2, 2)
    assert (by_name['rule-b'].matched, by_name['rule-b'].allowed, by_name['rule-b'].dropped) == (3, 1, 2)
    assert sorted(_names(exporter)) == ['a', 'a', 'b', 'c']


def test_first_matching_rule_wins(exporter: TestExporter, config_kwargs: dict[str, Any]) -> None:
    config_kwargs['quotas'] = QuotaOptions(
        rules=[
            QuotaRule(name_pattern='*', limit=1, period=60, on_exceed='drop', name='catch-all'),
            QuotaRule(name_pattern='specific', limit=100, period=60, on_exceed='drop', name='specific'),
        ]
    )
    logfire.configure(**config_kwargs)
    _use_fake_clock(FakeClock())

    for _ in range(3):
        with logfire.span('specific'):
            pass

    [catch_all, specific] = logfire.get_quota_stats()
    assert catch_all.matched == 3 and catch_all.dropped == 2
    assert specific.matched == 0 and specific.dropped == 0


def test_aggregate_emits_one_count_summary_on_window_rollover(
    exporter: TestExporter, config_kwargs: dict[str, Any]
) -> None:
    config_kwargs['quotas'] = QuotaOptions(
        rules=[QuotaRule(name_pattern='storm', limit=2, period=10, on_exceed='aggregate')]
    )
    logfire.configure(**config_kwargs)
    clock = FakeClock()
    _use_fake_clock(clock)

    for _ in range(5):
        with logfire.span('storm'):
            pass

    # Within the window the first two pass and the excess is held back; nothing summarized yet.
    assert _names(exporter) == ['storm', 'storm']
    [stats] = logfire.get_quota_stats()
    assert (stats.matched, stats.allowed, stats.aggregated, stats.summaries_emitted) == (5, 2, 3, 0)

    # Closing the window (by producing one more record) releases one summary for the previous window.
    clock.advance(11)
    with logfire.span('storm'):
        pass

    assert [s['attributes'][QUOTA_STRATEGY_KEY] for s in _summaries(exporter)] == ['aggregate']
    [summary] = _summaries(exporter)
    attrs = summary['attributes']
    assert attrs[QUOTA_RULE_KEY] == 'storm'
    assert attrs[QUOTA_KIND_KEY] == 'span'
    assert attrs[QUOTA_MATCHED_NAME_KEY] == 'storm'
    assert attrs[QUOTA_COUNT_KEY] == 3
    assert attrs['logfire.quota.period'] == 10
    assert attrs['logfire.quota.limit'] == 2

    # The new window's record passes through normally.
    assert _names(exporter).count('storm') == 3
    [stats] = logfire.get_quota_stats()
    assert (stats.matched, stats.allowed, stats.aggregated, stats.summaries_emitted) == (6, 3, 3, 1)
    # Accounting invariant for every rule.
    assert stats.matched == stats.allowed + stats.dropped + stats.aggregated


def test_aggregate_log_summary_is_marked_as_log(exporter: TestExporter, config_kwargs: dict[str, Any]) -> None:
    config_kwargs['quotas'] = QuotaOptions(
        rules=[QuotaRule(name_pattern='retry *', limit=1, period=10, on_exceed='aggregate', applies_to='log')]
    )
    logfire.configure(**config_kwargs)
    clock = FakeClock()
    _use_fake_clock(clock)

    for i in range(4):
        logfire.warn('retry attempt {i}', i=i)

    clock.advance(11)
    logfire.warn('retry attempt {i}', i=99)

    [summary] = _summaries(exporter)
    attrs = summary['attributes']
    assert attrs[QUOTA_KIND_KEY] == 'log'
    assert attrs['logfire.span_type'] == 'log'
    assert attrs[QUOTA_COUNT_KEY] == 3
    # The summary carries the highest level seen among the folded records.
    assert attrs['logfire.level_num'] == 13


def test_force_flush_releases_final_window_summary(exporter: TestExporter, config_kwargs: dict[str, Any]) -> None:
    config_kwargs['quotas'] = QuotaOptions(
        rules=[QuotaRule(name_pattern='final', limit=1, period=10, on_exceed='aggregate')]
    )
    logfire.configure(**config_kwargs)
    _use_fake_clock(FakeClock())

    for _ in range(3):
        with logfire.span('final'):
            pass
    assert _summaries(exporter) == []

    assert logfire.force_flush() is not False
    assert [s['attributes'][QUOTA_COUNT_KEY] for s in _summaries(exporter)] == [2]
    [stats] = logfire.get_quota_stats()
    assert stats.summaries_emitted == 1


def test_window_limit_resets_each_window(exporter: TestExporter, config_kwargs: dict[str, Any]) -> None:
    config_kwargs['quotas'] = QuotaOptions(rules=[QuotaRule(name_pattern='w', limit=2, period=10, on_exceed='drop')])
    logfire.configure(**config_kwargs)
    clock = FakeClock()
    _use_fake_clock(clock)

    for _ in range(5):
        with logfire.span('w'):
            pass
    assert _names(exporter) == ['w', 'w']

    clock.advance(10)
    for _ in range(3):
        with logfire.span('w'):
            pass
    assert _names(exporter) == ['w', 'w', 'w', 'w']
    [stats] = logfire.get_quota_stats()
    assert stats.current_window_used == 3


def test_concurrent_rules_keep_independent_accounts(exporter: TestExporter, config_kwargs: dict[str, Any]) -> None:
    config_kwargs['quotas'] = QuotaOptions(
        rules=[
            QuotaRule(name_pattern='a', limit=100, period=1000, on_exceed='drop', name='a'),
            QuotaRule(name_pattern='b', limit=50, period=1000, on_exceed='aggregate', name='b'),
        ]
    )
    logfire.configure(**config_kwargs)
    # Freeze the clock so every thread operates inside one quota window.
    _use_fake_clock(FakeClock())

    def worker(letter: str) -> None:
        for _ in range(250):
            with logfire.span(letter):
                pass

    threads = [Thread(target=worker, args=(letter,)) for letter in ('a', 'b') for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    by_name = {stats.name: stats for stats in logfire.get_quota_stats()}
    a, b = by_name['a'], by_name['b']
    assert a.matched == 1000 and a.allowed == 100 and a.dropped == 900 and a.aggregated == 0
    assert b.matched == 1000 and b.allowed == 50 and b.aggregated == 950 and b.dropped == 0
    for stats in (a, b):
        assert stats.matched == stats.allowed + stats.dropped + stats.aggregated

    logfire.force_flush()
    b_summaries = [s for s in _summaries(exporter) if s['attributes'][QUOTA_RULE_KEY] == 'b']
    assert [s['attributes'][QUOTA_COUNT_KEY] for s in b_summaries] == [950]
    assert not [s for s in _summaries(exporter) if s['attributes'][QUOTA_RULE_KEY] == 'a']


def test_unsampled_traces_are_not_charged(exporter: TestExporter, config_kwargs: dict[str, Any]) -> None:
    config_kwargs['sampling'] = logfire.SamplingOptions(head=0.0)
    config_kwargs['quotas'] = QuotaOptions(
        rules=[QuotaRule(name_pattern='sampled', limit=1, period=60, on_exceed='drop')]
    )
    logfire.configure(**config_kwargs)
    _use_fake_clock(FakeClock())

    for _ in range(10):
        with logfire.span('sampled'):
            pass

    assert _exported(exporter) == []
    [stats] = logfire.get_quota_stats()
    assert stats.matched == 0


def test_min_level_records_are_not_charged(exporter: TestExporter, config_kwargs: dict[str, Any]) -> None:
    config_kwargs['min_level'] = 'warning'
    config_kwargs['quotas'] = QuotaOptions(rules=[QuotaRule(name_pattern='debug spam *', limit=1, period=60)])
    logfire.configure(**config_kwargs)
    _use_fake_clock(FakeClock())

    for i in range(5):
        logfire.debug('debug spam {i}', i=i)
    logfire.error('debug spam {i}', i=99)

    [stats] = logfire.get_quota_stats()
    assert stats.matched == 1 and stats.allowed == 1


def test_quota_stats_after_shutdown(exporter: TestExporter, config_kwargs: dict[str, Any]) -> None:
    config_kwargs['quotas'] = QuotaOptions(
        rules=[QuotaRule(name_pattern='bye', limit=1, period=10, on_exceed='aggregate')]
    )
    logfire.configure(**config_kwargs)
    _use_fake_clock(FakeClock())

    for _ in range(4):
        with logfire.span('bye'):
            pass

    logfire.shutdown()

    # Totals stay readable at close-out, and shutting down flushed the outstanding summary.
    [stats] = logfire.get_quota_stats()
    assert (stats.matched, stats.allowed, stats.aggregated, stats.summaries_emitted) == (4, 1, 3, 1)
    assert [s['attributes'][QUOTA_COUNT_KEY] for s in _summaries(exporter)] == [3]


def test_otel_log_record_pipeline_drop_and_aggregate(logs_exporter: Any, config_kwargs: dict[str, Any]) -> None:
    config_kwargs['quotas'] = QuotaOptions(
        rules=[QuotaRule(name_pattern='otlp noisy *', limit=1, period=10, on_exceed='aggregate')]
    )
    logfire.configure(**config_kwargs)
    clock = FakeClock()
    _use_fake_clock(clock)

    logger = logfire.DEFAULT_LOGFIRE_INSTANCE.config.get_logger_provider().get_logger('test-scope')
    for i in range(3):
        logger.emit(
            LogRecord(
                body=f'otlp noisy body {i}',
                attributes={ATTRIBUTES_MESSAGE_TEMPLATE_KEY: 'otlp noisy template'},
                severity_number=SeverityNumber.ERROR,
            )
        )

    exported = logs_exporter.exported_logs_as_dicts()
    assert len(exported) == 1

    logfire.force_flush()
    exported = logs_exporter.exported_logs_as_dicts()
    summaries = [record for record in exported if record['attributes'].get(QUOTA_STRATEGY_KEY) == 'aggregate']
    assert len(summaries) == 1
    attrs = summaries[0]['attributes']
    assert attrs[QUOTA_RULE_KEY] == 'otlp noisy *'
    assert attrs[QUOTA_KIND_KEY] == 'log'
    assert attrs[QUOTA_COUNT_KEY] == 2
    [stats] = logfire.get_quota_stats()
    assert (stats.matched, stats.allowed, stats.aggregated) == (3, 1, 2)


def test_quotas_with_tail_sampling(exporter: TestExporter, config_kwargs: dict[str, Any]) -> None:
    # Tail sampling buffers on_end; the quota layer wraps the tail processor and still sees
    # exactly the spans that survive sampling.
    config_kwargs['sampling'] = logfire.SamplingOptions(tail=lambda span_info: 1.0)
    config_kwargs['quotas'] = QuotaOptions(
        rules=[QuotaRule(name_pattern='tailed', limit=2, period=60, on_exceed='drop')]
    )
    logfire.configure(**config_kwargs)
    _use_fake_clock(FakeClock())

    for _ in range(5):
        with logfire.span('tailed'):
            pass

    assert _names(exporter) == ['tailed'] * 2
    [stats] = logfire.get_quota_stats()
    assert (stats.matched, stats.allowed, stats.dropped) == (5, 2, 3)


def test_options_validation() -> None:
    with pytest.raises(ValueError, match='limit must be a positive integer'):
        QuotaRule(name_pattern='x', limit=0)
    with pytest.raises(ValueError, match='period must be greater than 0'):
        QuotaRule(name_pattern='x', limit=1, period=0)
    with pytest.raises(ValueError, match='on_exceed'):
        QuotaRule(name_pattern='x', limit=1, on_exceed='nope')  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match='unique'):
        QuotaOptions(
            rules=[
                QuotaRule(name_pattern='x', limit=1, name='dup'),
                QuotaRule(name_pattern='y', limit=1, name='dup'),
            ]
        )


def test_options_survive_dict_round_trip() -> None:
    # This is the serialization shape used for ProcessPoolExecutor config forwarding.
    from logfire.quotas._quota import coerce_quota_options

    options = QuotaOptions(rules=[QuotaRule(name_pattern=re.compile('^x.*'), limit=5, name='regex-rule')])
    serialized = asdict(options)

    restored = coerce_quota_options(serialized)
    [rule] = restored.rules
    assert rule.name == 'regex-rule'
    assert isinstance(rule.name_pattern, re.Pattern)
    assert rule.name_pattern.match('xyz') is not None
    # None also normalizes to an empty options object rather than raising.
    assert coerce_quota_options(None).rules == ()
