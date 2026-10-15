from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from datetime import timedelta

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

import logfire


class StdoutExporter(SpanExporter):
    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        print('\n'.join(span.name for span in spans))
        return SpanExportResult.SUCCESS


def open_span_in_suspended_generator() -> Iterator[None]:
    with logfire.span('open span at shutdown'):
        yield


logfire.configure(
    send_to_logfire=False,
    console=False,
    inspect_arguments=False,
    open_spans=logfire.OpenSpansOptions(max_age=timedelta(milliseconds=10)),
    additional_span_processors=[SimpleSpanProcessor(StdoutExporter())],
)
generator = open_span_in_suspended_generator()
next(generator)
# Stay suspended long enough to exceed the configured max age before the process exits.
time.sleep(0.05)
