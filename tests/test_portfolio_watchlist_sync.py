# -*- coding: utf-8 -*-
"""Unit tests for PortfolioWatchlistSync (D 选项: long-running STOCK_LIST 同步).

These tests exercise the sync logic against an in-memory SQLite database via the
existing ``PortfolioService`` + ``PortfolioRepository`` flow, and stub out
``SystemConfigService`` so we don't touch the real ``.env``.
"""

from __future__ import annotations

import os
import tempfile
import threading
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.config import Config
from src.services.portfolio_watchlist_sync import (
    PortfolioWatchlistSync,
    normalize_symbols,
)
from src.storage import (
    DatabaseManager,
    PortfolioAccount,
    PortfolioPosition,
    PortfolioPositionLot,
    PortfolioTrade,
)


class _StubConfigService:
    """Minimal in-memory stand-in for SystemConfigService.

    Records the latest STOCK_LIST value passed to ``update`` so we can assert
    what was actually written without touching ``.env``.
    """

    def __init__(self, initial_stock_list: str) -> None:
        self._items = [{"key": "STOCK_LIST", "value": initial_stock_list}]
        self._config_version = "test-version-1"
        self.updates = []
        self.write_error = None

    def get_config(self, *, include_schema: bool = True):  # noqa: ARG002 - signature parity
        return {
            "config_version": self._config_version,
            "items": list(self._items),
        }

    def update(self, *, config_version: str, items, mask_token: str, reload_now: bool):  # noqa: ARG002
        if self.write_error is not None:
            raise self.write_error
        for item in items:
            self._items = [it for it in self._items if it["key"] != item["key"]]
            self._items.append({"key": item["key"], "value": item["value"]})
            self.updates.append(dict(item, mask_token=mask_token, reload_now=reload_now))
        # Bump version to mimic real flow.
        self._config_version = f"{config_version}-done"
        return {"updated_keys": [item["key"] for item in items], "new_version": self._config_version}


class PortfolioWatchlistSyncTestCase(unittest.TestCase):
    """Verify PortfolioWatchlistSync behavior end-to-end against a temp DB."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.env_path = Path(self.temp_dir.name) / ".env"
        self.db_path = Path(self.temp_dir.name) / "portfolio_sync.db"
        self.env_path.write_text(
            "\n".join(
                [
                    "STOCK_LIST=600237,159215",
                    "GEMINI_API_KEY=test",
                    "ADMIN_AUTH_ENABLED=false",
                    f"DATABASE_PATH={self.db_path}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        os.environ["ENV_FILE"] = str(self.env_path)
        os.environ["DATABASE_PATH"] = str(self.db_path)
        Config.reset_instance()
        DatabaseManager.reset_instance()

        self.db = DatabaseManager.get_instance()
        self.service_lock = threading.Lock()

    def tearDown(self) -> None:
        DatabaseManager.reset_instance()
        Config.reset_instance()
        os.environ.pop("ENV_FILE", None)
        os.environ.pop("DATABASE_PATH", None)
        self.temp_dir.cleanup()

    def _seed_account(self, *, name: str, is_active: bool = True) -> int:
        from src.repositories.portfolio_repo import PortfolioRepository

        repo = PortfolioRepository()
        with self.service_lock:
            with self.db.get_session() as session:
                account = PortfolioAccount(
                    name=name,
                    broker="Demo",
                    market="cn",
                    base_currency="CNY",
                    is_active=is_active,
                )
                session.add(account)
                session.flush()
                account_id = int(account.id)
                session.commit()
                return account_id

    def _seed_trade(
        self,
        *,
        account_id: int,
        symbol: str,
        quantity: float,
        price: float = 10.0,
        trade_date: date | None = None,
        side: str = "buy",
        market: str = "cn",
        currency: str = "CNY",
    ) -> None:
        with self.service_lock:
            with self.db.get_session() as session:
                trade = PortfolioTrade(
                    account_id=account_id,
                    symbol=symbol,
                    market=market,
                    side=side,
                    quantity=quantity,
                    price=price,
                    fee=0.0,
                    tax=0.0,
                    currency=currency,
                    trade_date=trade_date or date(2026, 7, 1),
                )
                session.add(trade)
                session.flush()
                position = (
                    session.query(PortfolioPosition)
                    .filter_by(account_id=account_id, symbol=symbol, market=market)
                    .one_or_none()
                )
                if position is None:
                    position = PortfolioPosition(
                        account_id=account_id,
                        symbol=symbol,
                        market=market,
                        currency=currency,
                        cost_method="fifo",
                        quantity=quantity,
                        avg_cost=price,
                        total_cost=quantity * price,
                        last_price=price,
                        market_value_base=quantity * price,
                        unrealized_pnl_base=0.0,
                        valuation_currency="CNY" if currency == "CNY" else currency,
                    )
                    session.add(position)
                else:
                    position.quantity = (position.quantity or 0.0) + quantity
                    position.avg_cost = price
                    position.total_cost = (position.quantity or 0.0) * price
                lot = PortfolioPositionLot(
                    account_id=account_id,
                    symbol=symbol,
                    market=market,
                    currency=currency,
                    cost_method="fifo",
                    open_date=trade_date or date(2026, 7, 1),
                    remaining_quantity=quantity,
                    unit_cost=price,
                )
                session.add(lot)
                session.commit()

    def _build_config(self, *, enabled: bool, interval: int = 300) -> SimpleNamespace:
        return SimpleNamespace(
            portfolio_watchlist_sync_enabled=enabled,
            portfolio_watchlist_sync_interval_seconds=interval,
        )

    # ---- Tests -------------------------------------------------------------

    def test_disabled_no_op(self):
        cfg = self._build_config(enabled=False)
        stub = _StubConfigService(initial_stock_list="600237,159215")
        sync = PortfolioWatchlistSync(config_provider=lambda: cfg, config_service=stub)

        result = sync.run_once()
        self.assertFalse(result.enabled)
        self.assertEqual(result.skipped_reason, "disabled")
        self.assertEqual(stub.updates, [])

    def test_appends_missing_and_preserves_extras(self):
        account_id = self._seed_account(name="account-A", is_active=True)
        self._seed_trade(account_id=account_id, symbol="600519", quantity=100.0)
        self._seed_trade(account_id=account_id, symbol="000999", quantity=200.0)

        cfg = self._build_config(enabled=True)
        stub = _StubConfigService(initial_stock_list="600237,159215")
        sync = PortfolioWatchlistSync(config_provider=lambda: cfg, config_service=stub)

        result = sync.run_once()
        self.assertTrue(result.enabled)
        self.assertTrue(result.written)
        self.assertEqual(set(result.appended_symbols), {"000999", "600519"})
        # Existing extras + newly appended; order preserved (extras first, then appended).
        self.assertEqual(result.final_symbols, ["600237", "159215", "000999", "600519"])
        self.assertEqual(len(stub.updates), 1)
        new_value = stub.updates[0]["value"]
        for symbol in ("000999", "600519", "600237", "159215"):
            self.assertIn(symbol, new_value)

    def test_idempotent_skips_unchanged_write(self):
        account_id = self._seed_account(name="account-A", is_active=True)
        self._seed_trade(account_id=account_id, symbol="600519", quantity=100.0)

        cfg = self._build_config(enabled=True)
        stub = _StubConfigService(initial_stock_list="600519")
        sync = PortfolioWatchlistSync(config_provider=lambda: cfg, config_service=stub)

        first = sync.run_once()
        self.assertTrue(first.written)
        self.assertEqual(stub.updates, [{"key": "STOCK_LIST", "value": "600519", "mask_token": "******", "reload_now": False}])

        second = sync.run_once()
        self.assertFalse(second.written)
        self.assertEqual(second.skipped_reason, "unchanged")
        # No additional writes triggered.
        self.assertEqual(len(stub.updates), 1)

    def test_empty_portfolio_no_op(self):
        cfg = self._build_config(enabled=True)
        stub = _StubConfigService(initial_stock_list="600237,159215")
        sync = PortfolioWatchlistSync(config_provider=lambda: cfg, config_service=stub)

        result = sync.run_once()
        self.assertTrue(result.enabled)
        self.assertEqual(result.skipped_reason, "no_holdings")
        self.assertEqual(stub.updates, [])

    def test_filters_inactive_accounts(self):
        active_id = self._seed_account(name="active", is_active=True)
        inactive_id = self._seed_account(name="inactive", is_active=False)
        self._seed_trade(account_id=active_id, symbol="600519", quantity=50.0)
        self._seed_trade(account_id=inactive_id, symbol="000001", quantity=10.0)

        cfg = self._build_config(enabled=True)
        stub = _StubConfigService(initial_stock_list="")
        sync = PortfolioWatchlistSync(config_provider=lambda: cfg, config_service=stub)

        result = sync.run_once()
        self.assertTrue(result.written)
        self.assertEqual(result.appended_symbols, ["600519"])
        self.assertNotIn("000001", result.final_symbols)

    def test_filters_non_cn_markets(self):
        cn_id = self._seed_account(name="cn-account", is_active=True)
        # HK and US symbols must be ignored by watchlist sync (CN-only).
        self._seed_trade(
            account_id=cn_id,
            symbol="hk00700",
            quantity=100.0,
            price=300.0,
            market="hk",
            currency="HKD",
        )
        self._seed_trade(
            account_id=cn_id,
            symbol="600519",
            quantity=10.0,
            price=1500.0,
        )

        cfg = self._build_config(enabled=True)
        stub = _StubConfigService(initial_stock_list="")
        sync = PortfolioWatchlistSync(config_provider=lambda: cfg, config_service=stub)

        result = sync.run_once()
        self.assertTrue(result.written)
        self.assertNotIn("hk00700", result.final_symbols)
        self.assertIn("600519", result.final_symbols)

    def test_writer_error_fail_open(self):
        account_id = self._seed_account(name="account", is_active=True)
        self._seed_trade(account_id=account_id, symbol="600519", quantity=100.0)

        cfg = self._build_config(enabled=True)
        stub = _StubConfigService(initial_stock_list="")
        stub.write_error = RuntimeError("simulated persistence failure")
        sync = PortfolioWatchlistSync(config_provider=lambda: cfg, config_service=stub)

        result = sync.run_once()
        self.assertFalse(result.written)
        self.assertIn("simulated persistence failure", result.error or "")
        # final_symbols should still reflect the *previous* STOCK_LIST (no-op visually).
        self.assertEqual(result.final_symbols, [])

    def test_normalize_symbols_helper(self):
        self.assertEqual(
            normalize_symbols([" 600519", "600519", "000999 ", "", None]),
            ["600519", "000999"],
        )
        # Case insensitive dedupe preserves the first-seen casing.
        self.assertEqual(
            normalize_symbols(["hk00700", "HK00700"]),
            ["hk00700"],
        )


class PortfolioWatchlistSyncResultTestCase(unittest.TestCase):
    """Smoke tests for the result dataclass."""

    def test_to_dict_round_trip(self):
        from src.services.portfolio_watchlist_sync import PortfolioWatchlistSyncResult

        result = PortfolioWatchlistSyncResult(
            enabled=True,
            account_count=1,
            holding_symbols=["600519"],
            previous_symbols=["000999"],
            appended_symbols=["600519"],
            final_symbols=["000999", "600519"],
            written=True,
            skipped_reason=None,
            error=None,
        )
        as_dict = result.to_dict()
        self.assertEqual(as_dict["enabled"], True)
        self.assertEqual(as_dict["appended_symbols"], ["600519"])
        self.assertEqual(as_dict["written"], True)


if __name__ == "__main__":
    unittest.main()