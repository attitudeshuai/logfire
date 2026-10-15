"""Types for configuring production quotas.

See [`logfire.configure(quotas=...)`][logfire.configure] and
[`logfire.get_quota_stats()`][logfire.get_quota_stats].
"""

from ._quota import (
    QuotaAction,
    QuotaKind,
    QuotaManager,
    QuotaOptions,
    QuotaRule,
    QuotaStats,
    QuotaTarget,
)

__all__ = [
    'QuotaAction',
    'QuotaKind',
    'QuotaManager',
    'QuotaOptions',
    'QuotaRule',
    'QuotaStats',
    'QuotaTarget',
]
