from typing import Literal, Sequence

QuotaAction = Literal['drop', 'aggregate']
QuotaKind = Literal['span', 'log']
QuotaTarget = Literal['any', 'span', 'log']

class QuotaRule:
    name_pattern: str | object
    limit: int
    period: float
    on_exceed: QuotaAction
    applies_to: QuotaTarget
    name: str | None
    def __init__(
        self,
        name_pattern: str | object,
        limit: int,
        period: float = 1.0,
        on_exceed: QuotaAction = 'drop',
        applies_to: QuotaTarget = 'any',
        name: str | None = None,
    ) -> None: ...

class QuotaOptions:
    rules: Sequence[QuotaRule]
    def __init__(self, rules: Sequence[QuotaRule] = ()) -> None: ...

class QuotaStats:
    name: str
    name_pattern: str
    applies_to: QuotaTarget
    limit: int
    period: float
    on_exceed: QuotaAction
    matched: int
    allowed: int
    dropped: int
    aggregated: int
    summaries_emitted: int
    current_window: int
    current_window_used: int

class QuotaManager:
    def __init__(self, rules: Sequence[QuotaRule] = (), *, now: object = ...) -> None: ...
    @property
    def enabled(self) -> bool: ...
    def stats(self) -> list[QuotaStats]: ...
