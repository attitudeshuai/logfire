---
title: "Logfire: Guide for Scrubbing Sensitive Data from Logs"
description: "Learn how to use Logfire to keep logs & sensitive data safe: Automatically scan spans & logs to scrub data like passwords, tokens & PII before exporting."
---
# Scrubbing sensitive data

The **Logfire** SDK scans for and redacts potentially sensitive data from logs and spans before exporting them.

## Disabling scrubbing

To disable scrubbing entirely, set [`scrubbing`][logfire.configure(scrubbing)] to `False`:

```python
import logfire

logfire.configure(scrubbing=False)
```

## Scrubbing more with custom patterns

By default, the SDK looks for some sensitive regular expressions. To add your own patterns, set [`extra_patterns`][logfire.ScrubbingOptions.extra_patterns] to a list of regex strings:

```python
import logfire

logfire.configure(scrubbing=logfire.ScrubbingOptions(extra_patterns=['my_pattern']))

logfire.info(
    'Hello',
    data={
        'key_matching_my_pattern': 'This string will be redacted because its key matches',
        'other_key': 'This string will also be redacted because it matches MY_PATTERN case-insensitively',
        'password': 'This will be redacted because custom patterns are combined with the default patterns',
    },
)
```

Here are the default scrubbing patterns:

```python
[
    'password',
    'passwd',
    'mysql_pwd',
    'secret',
    'auth(?!ors?\\b)',
    'credential',
    'private[._ -]?key',
    'api[._ -]?key',
    'session',
    'cookie',
    'social[._ -]?security',
    'credit[._ -]?card',
    'logfire[._ -]?token',
    '(?:access|refresh|bearer|(?:\\b|_)id)[._ -]?token',
    'pylf_v\\d+_',
    '(?:\\b|_)csrf(?:\\b|_)',
    '(?:\\b|_)xsrf(?:\\b|_)',
    '(?:\\b|_)jwt(?:\\b|_)',
    '(?:\\b|_)ssn(?:\\b|_)',
]
```

## Scrubbing less with a callback

On the other hand, if the scrubbing is too aggressive, you can pass a function to [`callback`][logfire.ScrubbingOptions.callback] to prevent certain data from being redacted.

The function will be called for each potential match found by the scrubber. If it returns `None`, the value is redacted. Otherwise, the returned value replaces the matched value. The function accepts a single argument of type [`logfire.ScrubMatch`][logfire.ScrubMatch].

Here's an example:

```python
import logfire


def scrubbing_callback(match: logfire.ScrubMatch):
    # `my_safe_value` often contains the string 'password' but it's not actually sensitive.
    if match.path == ('attributes', 'my_safe_value') and match.pattern_match.group(0) == 'password':
        # Return the original value to prevent redaction.
        return match.value


logfire.configure(scrubbing=logfire.ScrubbingOptions(callback=scrubbing_callback))
```

## Dry runs and hit summaries

A new scrubbing pattern is hard to validate: writing it too broadly wipes out normal fields, while writing it too narrowly lets sensitive data through, and the only feedback is a full deploy-and-observe cycle. To make this safer, the SDK supports **dry-run mode** and a **hit summary**.

### Dry-run mode

Set [`dry_run`][logfire.ScrubbingOptions.dry_run] to `True` to run the scrubber without modifying anything:

```python
import logfire

logfire.configure(
    scrubbing=logfire.ScrubbingOptions(
        extra_patterns=['employee_id'],
        dry_run=True,
        summary=logfire.ScrubbingSummaryOptions(),
    )
)
```

In dry-run mode, matched values are left in place and exported normally, exactly as if scrubbing hadn't run. Meanwhile every hit is recorded in the summary (see below), including where it would have been redacted. Once the numbers look right, remove `dry_run=True` to start enforcing the pattern.

If a [`callback`][logfire.ScrubbingOptions.callback] is configured, it is still called in dry-run mode so that hits it would refuse to redact are recorded separately, but its return value never changes the exported data.

### Reading the summary

Pass [`ScrubbingSummaryOptions`][logfire.ScrubbingSummaryOptions] to [`summary`][logfire.ScrubbingOptions.summary] to record hits, in either dry-run or normal mode. Retrieve an immutable snapshot with [`logfire.get_scrubbing_summary()`][logfire.get_scrubbing_summary]:

```python
import logfire

summary = logfire.get_scrubbing_summary()
for pattern, pattern_summary in summary.patterns.items():
    print(pattern, pattern_summary.hits, pattern_summary.redacted, pattern_summary.unchanged)
    for path, path_summary in pattern_summary.paths.items():
        print('  ', path, path_summary.hits)
```

Each [`ScrubbingPatternSummary`][logfire.ScrubbingPatternSummary] reports:

- `hits`: total number of times the pattern matched.
- `redacted` / `unchanged`: matches that were actually redacted or replaced, versus matches left unchanged because the callback returned the original value. In dry-run mode, `redacted` counts values that *would* have been redacted.
- `paths`: how hits were distributed across attribute paths (e.g. `('attributes', 'password')`), each with its own [`ScrubbingPathSummary`][logfire.ScrubbingPathSummary].
- `first_hit` and `last_hit`: timestamps of the first and most recent match.

The summary is bounded by [`max_entries`][logfire.ScrubbingSummaryOptions.max_entries] (the number of `(pattern, path)` entries kept, default 10,000). When it is full, the entry that was hit least recently is discarded (the only supported [`eviction_policy`][logfire.ScrubbingSummaryOptions.eviction_policy], `'drop-oldest'`), and [`dropped_entries`][logfire.ScrubbingSummary.dropped_entries] is incremented. Pattern-level hit counts stay exact regardless; only the per-path breakdown loses detail.

When scrubbing is disabled (`scrubbing=False`) or no `summary` options are configured, no summary is collected and `get_scrubbing_summary()` returns `None`.

## Security tips

### Use message templates

The full span/log message is not scrubbed, only the fields within. For example, this:

```python skip="true" skip-reason="incomplete"
logfire.info('User details: {user}', user=User(id=123, password='secret'))
```

...may log something like:

```
User details: [Scrubbed due to 'password']
```

...but this:

```python skip="true" skip-reason="incomplete"
user = User(id=123, password='secret')
logfire.info('User details: ' + str(user))
```

will log:

```
User details: User(id=123, password='secret')
```

This is necessary so that safe messages such as 'Password is correct' are not redacted completely.

Using f-strings (e.g. `logfire.info(f'User details: {user}')`) *is* safe if `inspect_arguments` is enabled (the default in Python 3.11+) and working correctly.
[See here](../guides/onboarding-checklist/add-manual-tracing.md#f-strings) for more information.

In short, don't format the message yourself. This is also a good practice in general for [other reasons](../guides/onboarding-checklist/add-manual-tracing.md#messages-and-span-names).

### Keep sensitive data out of URLs

The attribute `"http.url"` which is recorded by OpenTelemetry instrumentation libraries is considered safe so that URLs like `"http://example.com/users/123/authenticate"` are not redacted.

As a general rule, not just for Logfire, assume that URLs (including query parameters) will be logged, so sensitive data should be put in the request body or headers instead.

### Use parameterized database queries

The `"db.statement"` attribute which is recorded by OpenTelemetry database instrumentation libraries is considered safe so that SQL queries like `"SELECT secret_value FROM table WHERE ..."` are not redacted.

Use parameterized queries (e.g. prepared statements) so that sensitive data is not interpolated directly into the query string, even if
you use an interpolation method that's safe from SQL injection.

### LLM and AI messages

Scrubbing is **disabled** for LLM message attributes such as `gen_ai.input.messages`, `gen_ai.output.messages`, and `pydantic_ai.all_messages`. This is intentional because:

1. **False positives**: LLMs frequently produce content containing words like "password" or "secret" in normal conversation (e.g., "Your password has been reset" or "The secret to success is..."), which would trigger false positives.
2. **Ineffective detection**: LLMs might output sensitive data without using any keywords that regex-based scrubbing could detect.

Because of these limitations, if your LLM interactions might contain sensitive data, the recommended approach is to **exclude message content from logging entirely** rather than relying on scrubbing. For example, with [Pydantic AI](../integrations/llms/pydanticai.md):

```python skip-run="true" skip-reason="external-connection"
import logfire

logfire.configure()
logfire.instrument_pydantic_ai(include_content=False)
```

This will still log spans for LLM calls and agent runs with timing and metadata, but will exclude the actual prompts, completions, and tool call arguments/responses.
