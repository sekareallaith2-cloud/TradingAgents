"""Precious metals as an asset type, with the macro analyst and gold's FRED drivers.

Gold has no company behind it, so a metal run drops the fundamentals analyst and gains a macro analyst
that reads real yields, the dollar and inflation expectations. Stock and crypto runs must not change.
All FRED access is mocked.
"""
from __future__ import annotations

import re
from unittest import mock
from unittest.mock import MagicMock

import pytest

from cli.models import AnalystType, AssetType
from cli.prompts import detect_asset_type, filter_analysts_for_asset_type, parse_analysts
from tests.test_tool_date_enforcement import TRADE_DATE, _run
from tradingagents.agents import tools
from tradingagents.agents.context import build_instrument_context, macro_report_section
from tradingagents.agents.researchers.bear_researcher import create_bear_researcher
from tradingagents.agents.researchers.bull_researcher import create_bull_researcher
from tradingagents.agents.risk_mgmt.aggressive_debator import create_aggressive_debator
from tradingagents.agents.risk_mgmt.conservative_debator import create_conservative_debator
from tradingagents.agents.risk_mgmt.neutral_debator import create_neutral_debator
from tradingagents.dataflows import router
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.vendors import fred
from tradingagents.graph.analyst_execution import build_analyst_execution_plan
from tradingagents.graph.propagation import Propagator
from tradingagents.reporting import _report_parts, write_report_tree

STOCK_ANALYSTS = [AnalystType.MARKET, AnalystType.SOCIAL, AnalystType.NEWS, AnalystType.FUNDAMENTALS]


# --- asset detection and analyst selection ----------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("raw", ["GC=F", "XAUUSD", "XAUUSD+", "GOLD", "SI=F", "XAGUSD", "PL=F"])
def test_precious_metals_are_detected(raw):
    assert detect_asset_type(raw) == AssetType.METAL


@pytest.mark.unit
@pytest.mark.parametrize("raw", ["GLD", "AAPL", "HG=F", "CL=F", "BTC-USD"])
def test_everything_else_keeps_its_asset_type(raw):
    assert detect_asset_type(raw) != AssetType.METAL


@pytest.mark.unit
def test_a_metal_drops_fundamentals_and_keeps_macro():
    chosen = filter_analysts_for_asset_type(list(AnalystType), AssetType.METAL)
    assert AnalystType.FUNDAMENTALS not in chosen and AnalystType.MACRO in chosen


@pytest.mark.unit
@pytest.mark.parametrize("asset", [AssetType.STOCK, AssetType.CRYPTO])
def test_stock_and_crypto_never_get_the_macro_analyst(asset):
    assert AnalystType.MACRO not in filter_analysts_for_asset_type(list(AnalystType), asset)


@pytest.mark.unit
def test_the_default_stock_team_is_unchanged():
    assert filter_analysts_for_asset_type(list(AnalystType), AssetType.STOCK) == STOCK_ANALYSTS


@pytest.mark.unit
def test_analyst_flags_are_checked_against_the_asset():
    assert parse_analysts("market,news,macro", AssetType.METAL) == [
        AnalystType.MARKET, AnalystType.NEWS, AnalystType.MACRO]
    with pytest.raises(ValueError, match="macro is not available for stock"):
        parse_analysts("macro", AssetType.STOCK)
    with pytest.raises(ValueError, match="fundamentals is not available for metal"):
        parse_analysts("fundamentals", AssetType.METAL)


# --- the macro analyst's place in the graph and the prompts -----------------

@pytest.mark.unit
def test_the_macro_analyst_has_its_own_node_report_and_tools():
    (spec,) = build_analyst_execution_plan(["macro"]).specs
    assert (spec.agent_node, spec.report_key) == ("Macro Analyst", "macro_report")
    assert "get_gold_macro_snapshot" in {t.name for t in spec.tools}


@pytest.mark.unit
def test_the_initial_state_has_an_empty_macro_report():
    state = Propagator().create_initial_state("GC=F", "2026-10-09", asset_type="metal")
    assert state["macro_report"] == "" and state["asset_type"] == "metal"


@pytest.mark.unit
def test_the_metal_context_says_there_are_no_company_fundamentals():
    metal = build_instrument_context("GC=F", "metal")
    assert "precious-metal futures" in metal and "real yields" in metal
    assert "precious-metal" not in build_instrument_context("AAPL", "stock")


@pytest.mark.unit
def test_the_macro_section_is_empty_unless_the_run_has_a_macro_report():
    assert macro_report_section({}, "Macro") == ""
    assert macro_report_section({"macro_report": "  "}, "Macro") == ""
    assert macro_report_section({"macro_report": "real yields up"}, "Macro") == "\nMacro: real yields up"


_REPORTS = {
    "company_of_interest": "GC=F", "asset_type": "metal",
    "market_report": "m", "sentiment_report": "s", "news_report": "n", "fundamentals_report": "f",
    "trader_investment_plan": "plan",
}
_INVEST = {"history": "", "bull_history": "", "bear_history": "", "current_response": "", "count": 0}
_RISK = {
    "current_aggressive_response": "", "current_conservative_response": "", "current_neutral_response": "",
    "history": "", "aggressive_history": "", "conservative_history": "", "neutral_history": "", "count": 0,
}
DEBATERS = [
    (create_bull_researcher, {"investment_debate_state": _INVEST}),
    (create_bear_researcher, {"investment_debate_state": _INVEST}),
    (create_aggressive_debator, {"risk_debate_state": _RISK}),
    (create_conservative_debator, {"risk_debate_state": _RISK}),
    (create_neutral_debator, {"risk_debate_state": _RISK}),
]


def _prompt_of(factory, extra, **state):
    captured = {}
    llm = MagicMock()
    llm.invoke.side_effect = lambda prompt: captured.__setitem__("prompt", prompt) or MagicMock(content="argument")
    factory(llm)({**_REPORTS, **extra, **state})
    return captured["prompt"]


@pytest.mark.unit
@pytest.mark.parametrize("factory,extra", DEBATERS, ids=lambda v: getattr(v, "__name__", ""))
def test_every_debater_reads_the_macro_report_only_when_there_is_one(factory, extra):
    with_macro = _prompt_of(factory, extra, macro_report="REAL-YIELD-REPORT")
    without = _prompt_of(factory, extra)
    assert "REAL-YIELD-REPORT" in with_macro
    assert "REAL-YIELD-REPORT" not in without and "acro" not in without.replace("macroeconomic", "")


# --- saved reports ----------------------------------------------------------

@pytest.mark.unit
def test_the_macro_report_is_saved_with_the_analyst_reports(tmp_path):
    state = {"market_report": "M", "macro_report": "MACRO TEXT", "final_trade_decision": "**Rating**: Hold"}
    heading, folder, written = _report_parts(state)[0]
    assert folder == "1_analysts" and [a for a, _f, _t in written] == ["Market Analyst", "Macro Analyst"]
    path = write_report_tree(state, "GC=F", tmp_path, html=False)
    assert (tmp_path / "1_analysts" / "macro.md").read_text(encoding="utf-8") == "MACRO TEXT"
    assert "### Macro Analyst" in path.read_text(encoding="utf-8")


# --- the gold macro tool ----------------------------------------------------

@pytest.mark.unit
def test_the_gold_snapshot_tool_takes_its_date_from_the_run():
    args = _run(tools.get_gold_macro_snapshot, {"curr_date": "2026-12-31"}, tools)
    assert args == ("get_gold_macro_snapshot", TRADE_DATE)


def _series(*points):
    return {"observations": [{"date": d, "value": v} for d, v in points]}


# 150 days of history ending 2026-10-09, so every window has a start value.
def _history(start, end):
    return _series(("2026-05-12", str(start)), ("2026-08-10", str(start)), ("2026-09-01", str(start)),
                   ("2026-09-09", str(start)), ("2026-10-02", str(end)), ("2026-10-08", str(end)))


FAKE = {
    "DFII10": _history(2.40, 2.90),      # +0.50pp: headwind
    "DTWEXBGS": _history(100.0, 98.0),   # -2.0%: tailwind (a falling dollar helps gold)
    "T10YIE": _history(2.35, 2.36),      # +0.01pp: neutral
    "VIXCLS": _history(15.0, 15.5),      # +3.3%: inside the VIX noise band, neutral
    "DGS10": _history(4.50, 5.20),
    "DGS2": _history(4.20, 4.70),
}


def _request(path, params):
    assert path == "series/observations"
    series = params["series_id"]
    if series == "DFF":
        raise ValueError("FRED request failed: Bad Request. The series does not exist.")
    return FAKE[series]


@pytest.mark.unit
def test_the_gold_drivers_are_scored_from_the_numbers(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "test-key")
    with mock.patch.object(fred, "_request", side_effect=_request) as requested:
        out = fred.get_gold_drivers("2026-10-09")

    rows = {m.group(1): line for line in out.splitlines() if (m := re.match(r"\| (.+?) \([A-Z0-9]+\) \|", line))}
    assert rows["10y real yield (TIPS)"].endswith("| headwind |")
    assert rows["Broad dollar index"].endswith("| tailwind |")
    assert rows["10y breakeven inflation"].endswith("| neutral |")
    assert rows["VIX"].endswith("| neutral |")
    assert rows["10y nominal yield"].endswith("| - |")                  # context rows are not scored
    assert "unavailable" in rows["Effective fed funds rate"]           # one bad series must not sink the table
    assert "2.90% (2026-10-08)" in rows["10y real yield (TIPS)"] and "+0.50pp" in rows["10y real yield (TIPS)"]
    assert "**Scorecard (30d):** headwinds: 10y real yield (TIPS). tailwinds: Broad dollar index." in out

    # Every request is capped at the analysis date and pins the data vintage published by then.
    for call in requested.call_args_list:
        params = call.args[1]
        assert params["observation_end"] == "2026-10-09"
        assert params["realtime_start"] == params["realtime_end"] == "2026-10-09"


@pytest.mark.unit
def test_a_window_without_a_start_value_is_not_made_up():
    points = [("2026-10-01", 2.0), ("2026-10-08", 2.1)]
    assert fred._change(points, 7, "rate") == pytest.approx(0.1)     # the start value on or before the cutoff
    assert fred._change(points, 90, "rate") is None                  # no history that old: no number


@pytest.mark.unit
def test_gold_aliases_resolve_to_the_real_yield_and_breakeven_series():
    assert fred._resolve_series_id("real_yield_10y") == "DFII10"
    assert fred._resolve_series_id("Real Yield") == "DFII10"
    assert fred._resolve_series_id("breakeven_10y") == "T10YIE"
    assert fred._resolve_series_id("fed_funds_daily") == "DFF"


@pytest.mark.unit
def test_without_a_fred_key_the_snapshot_degrades_instead_of_aborting_the_run(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    set_config({"data_vendors": {"macro_data": "fred"}})
    result = router.route_to_vendor("get_gold_macro_snapshot", "2026-10-09")
    assert result.startswith("DATA_UNAVAILABLE") and "do not fabricate" in result
