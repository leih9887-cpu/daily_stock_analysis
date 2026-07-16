"""Sync active portfolio holdings into STOCK_LIST.

Keeps the long-form watchlist in `STOCK_LIST` in sync with the symbols actually held
in active `portfolio_accounts`. Designed to be invoked by the scheduler background
loop and exposed through `runtime_scheduler.build_portfolio_watchlist_sync_background_tasks`.

Operational contract (do not change without reading docs/full-guide.md "持仓与 STOCK_LIST 联动"):

* **Direction**: single-direction; portfolio holdings → STOCK_LIST.
* **Default**: disabled (`portfolio_watchlist_sync_enabled=False`) so the legacy
  StockAnalysis watchlist behavior is preserved.
* **Scope**: iterates only `is_active=True` accounts and `quantity>0` positions
  via `PortfolioRepository.list_cached_position_identities`.
* **Append-only**: symbols present in `STOCK_LIST` but absent from active holdings
  are preserved (so user-added "extra watchlist" entries like 600237 are not lost).
* **Idempotent**: if the merged set is identical to the last successful write, the
  service no-ops without touching `.env`.
* **Fail-open**: writer errors are caught and logged as warnings; the scheduler
  loop must not crash on a transient persistence failure.
* **No realtime price fetch**: reads cached `portfolio_positions.symbol` only.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.services.portfolio_service import PortfolioService
from src.services.system_config_service import SystemConfigService

logger = logging.getLogger(__name__)

_STOCK_LIST_KEY = "STOCK_LIST"
_LIST_VALUE_SEPARATOR = ","
_CN_MARKET = "cn"


@dataclass(frozen=True)
class PortfolioWatchlistSyncResult:
    """Outcome of a single sync pass; safe to serialize for diagnostics."""

    enabled: bool
    account_count: int
    holding_symbols: List[str]
    previous_symbols: List[str]
    appended_symbols: List[str] = field(default_factory=list)
    final_symbols: List[str] = field(default_factory=list)
    written: bool = False
    skipped_reason: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "account_count": self.account_count,
            "holding_symbols": list(self.holding_symbols),
            "previous_symbols": list(self.previous_symbols),
            "appended_symbols": list(self.appended_symbols),
            "final_symbols": list(self.final_symbols),
            "written": self.written,
            "skipped_reason": self.skipped_reason,
            "error": self.error,
        }


def normalize_symbols(symbols: Sequence[str]) -> List[str]:
    """Normalize/preserve order, drop blanks, dedupe case-insensitively."""
    seen: Dict[str, str] = {}
    out: List[str] = []
    for raw in symbols:
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        key = text.upper()
        if key in seen:
            continue
        seen[key] = text
        out.append(text)
    return out


def _hash_symbols(symbols: Sequence[str]) -> str:
    payload = _LIST_VALUE_SEPARATOR.join(normalize_symbols(symbols))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class PortfolioWatchlistSync:
    """Periodic sync that mirrors active portfolio symbols into `STOCK_LIST`."""

    def __init__(
        self,
        *,
        config_provider: Optional[Callable[[], Any]] = None,
        portfolio_service: Optional[PortfolioService] = None,
        config_service: Optional[SystemConfigService] = None,
    ) -> None:
        self._config_provider = config_provider
        self._portfolio_service = portfolio_service or PortfolioService()
        self._config_service = config_service or SystemConfigService()
        self._last_written_hash: Optional[str] = None

    # ---- Public API --------------------------------------------------------

    def run_once(self) -> PortfolioWatchlistSyncResult:
        """Run one pass. Never raises; returns a structured result instead."""
        try:
            return self._run_once_inner()
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning("[PortfolioWatchlistSync] pass failed with error: %s", exc)
            return PortfolioWatchlistSyncResult(
                enabled=False,
                account_count=0,
                holding_symbols=[],
                previous_symbols=[],
                error=str(exc),
            )

    # ---- Internal ----------------------------------------------------------

    def _is_enabled(self) -> bool:
        config = self._safe_get_config()
        if config is None:
            return False
        return bool(getattr(config, "portfolio_watchlist_sync_enabled", False))

    def _safe_get_config(self) -> Any:
        if self._config_provider is not None:
            try:
                return self._config_provider()
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.warning("[PortfolioWatchlistSync] config provider raised: %s", exc)
        try:
            from src.config import Config

            return Config.get_instance()
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning("[PortfolioWatchlistSync] Config.get_instance() failed: %s", exc)
            return None

    def _collect_active_holding_symbols(self) -> List[str]:
        """Return sorted/normalized CN symbols held across all active accounts."""
        repo = getattr(self._portfolio_service, "repo", None)
        if repo is None or not hasattr(repo, "list_cached_position_identities"):
            return []
        try:
            identities = repo.list_cached_position_identities()
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning(
                "[PortfolioWatchlistSync] repo.list_cached_position_identities failed: %s",
                exc,
            )
            return []

        out: List[str] = []
        for market, symbol in identities or []:
            if (market or "").lower() != _CN_MARKET:
                continue
            if symbol:
                out.append(str(symbol))
        return normalize_symbols(out)

    def _count_active_accounts(self) -> int:
        try:
            accounts = self._portfolio_service.list_accounts(include_inactive=False)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning("[PortfolioWatchlistSync] list_accounts failed: %s", exc)
            return 0
        return len(accounts or [])

    def _read_current_stock_list(self) -> List[str]:
        try:
            config_data = self._config_service.get_config(include_schema=False)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning(
                "[PortfolioWatchlistSync] config_service.get_config failed: %s", exc
            )
            return []

        items = config_data.get("items", []) if isinstance(config_data, dict) else []
        for item in items:
            if not isinstance(item, dict):
                continue
            if str(item.get("key", "")).upper() != _STOCK_LIST_KEY:
                continue
            raw_value = item.get("value")
            if raw_value in (None, ""):
                return []
            try:
                from src.services.stock_list_parser import split_stock_list

                return split_stock_list(str(raw_value))
            except Exception:  # pragma: no cover - parser import fallback
                text = str(raw_value).replace("，", ",")
                return [code for code in text.split(",") if code.strip()]
        return []

    def _write_stock_list(self, codes: Sequence[str]) -> str:
        merged = normalize_symbols(codes)
        new_value = _LIST_VALUE_SEPARATOR.join(merged)
        try:
            config_data = self._config_service.get_config(include_schema=False)
            config_version = (
                config_data.get("config_version", "") if isinstance(config_data, dict) else ""
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning(
                "[PortfolioWatchlistSync] cannot fetch config_version before write: %s",
                exc,
            )
            raise

        # `reload_now=False` is intentional: the periodic loop already runs against
        # the freshly persisted `.env`, and a full Config singleton reset on every
        # tick would disrupt downstream consumers (e.g. analysis pipeline caches).
        self._config_service.update(
            config_version=config_version,
            items=[{"key": _STOCK_LIST_KEY, "value": new_value}],
            mask_token="******",
            reload_now=False,
        )
        return new_value

    # ---- Core pass ---------------------------------------------------------

    def _run_once_inner(self) -> PortfolioWatchlistSyncResult:
        enabled = self._is_enabled()
        if not enabled:
            return PortfolioWatchlistSyncResult(
                enabled=False,
                account_count=0,
                holding_symbols=[],
                previous_symbols=[],
                skipped_reason="disabled",
            )

        account_count = self._count_active_accounts()
        holding_symbols = self._collect_active_holding_symbols()
        previous_symbols = self._read_current_stock_list()

        if not holding_symbols:
            return PortfolioWatchlistSyncResult(
                enabled=True,
                account_count=account_count,
                holding_symbols=[],
                previous_symbols=previous_symbols,
                skipped_reason="no_holdings",
            )

        previous_set = {code.upper() for code in previous_symbols}
        holding_set = {code.upper() for code in holding_symbols}
        appended = [code for code in holding_symbols if code.upper() not in previous_set]
        extras = [
            code for code in previous_symbols if code.upper() not in holding_set
        ]
        merged = normalize_symbols(list(previous_symbols) + appended + extras)

        new_hash = _hash_symbols(merged)
        if new_hash == self._last_written_hash:
            return PortfolioWatchlistSyncResult(
                enabled=True,
                account_count=account_count,
                holding_symbols=holding_symbols,
                previous_symbols=previous_symbols,
                appended_symbols=[],
                final_symbols=merged,
                skipped_reason="unchanged",
            )

        try:
            self._write_stock_list(merged)
        except Exception as exc:
            logger.warning(
                "[PortfolioWatchlistSync] failed to write STOCK_LIST (%s); leaving config untouched",
                exc,
            )
            return PortfolioWatchlistSyncResult(
                enabled=True,
                account_count=account_count,
                holding_symbols=holding_symbols,
                previous_symbols=previous_symbols,
                appended_symbols=appended,
                final_symbols=normalize_symbols(previous_symbols),
                written=False,
                error=str(exc),
            )

        self._last_written_hash = new_hash
        if appended:
            logger.info(
                "[PortfolioWatchlistSync] appended %d symbol(s) to STOCK_LIST: %s",
                len(appended),
                _LIST_VALUE_SEPARATOR.join(appended),
            )
        else:
            logger.debug(
                "[PortfolioWatchlistSync] wrote STOCK_LIST unchanged (%d entries)",
                len(merged),
            )
        return PortfolioWatchlistSyncResult(
            enabled=True,
            account_count=account_count,
            holding_symbols=holding_symbols,
            previous_symbols=previous_symbols,
            appended_symbols=appended,
            final_symbols=merged,
            written=True,
        )
