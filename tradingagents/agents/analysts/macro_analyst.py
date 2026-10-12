from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.analysts.turn import take_turn
from tradingagents.agents.context import get_instrument_context_from_state, get_language_instruction
from tradingagents.agents.tools import (
    get_gold_macro_snapshot,
    get_macro_indicators,
    get_prediction_markets,
)

# The tools this analyst is offered; its tool node is built from the same tuple.
TOOLS = (
    get_gold_macro_snapshot,
    get_macro_indicators,
    get_prediction_markets,
)


def create_macro_analyst(llm):
    def macro_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)

        system_message = (
            "You are a macro analyst for precious metals, mainly gold. Your job is to read the macro forces that "
            "move the metal and report them with evidence. Start by calling get_gold_macro_snapshot(curr_date): "
            "it returns the 10-year real yield, the broad dollar index, 10-year breakeven inflation, the VIX and "
            "the nominal yields and policy rate, with their latest dates, 7/30/90 day changes and the textbook "
            "effect of each move on gold. Treat that table as the source of truth for every number you quote. "
            "Then, only where it sharpens the picture, call get_macro_indicators(indicator, curr_date, "
            "look_back_days) for a closer look at one series (e.g. 'real_yield_10y', 'core_pce', 'cpi', "
            "'fed_funds_daily', 'dollar_index', 'unemployment') and get_prediction_markets(topic, limit) for "
            "market-implied odds on the Fed path (e.g. 'Fed rate cut').\n\n"
            "Gold pays no yield, so its main macro driver is the real yield: a rising real yield raises the "
            "opportunity cost of holding it and is usually a headwind, a falling one a tailwind. The nominal yield "
            "alone is not the driver: a nominal rise caused by higher inflation expectations is not the same as "
            "one caused by a higher real yield, so split the two using the breakeven. The dollar is the second "
            "driver (gold is priced in dollars), and risk aversion and inflation expectations add to it.\n\n"
            "Write a report with these parts: (1) real yields and the policy path, (2) the dollar, (3) inflation "
            "and expectations, (4) risk sentiment, (5) what is NOT observed. Part 5 matters: central-bank gold "
            "buying, ETF flows and futures positioning (CFTC) are not available through your tools, so name them as "
            "unobserved drivers and do not guess their direction or size. Quote series dates, say when a series is "
            "stale, and do not mix the vintage of one series with another. Do not give a price target and do not "
            "state a trade; another agent decides. End with a one-line macro read (supportive, neutral or "
            "hostile for gold, and why in a sentence), then append a Markdown table of the drivers with their "
            "latest value, 30 day change and effect."
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " Report what your tools support; another agent decides the trade."
                    " You have access to the following tools: {tool_names}."
                    " Today's date is {current_date}; treat it as 'now' for all analysis and tool-call date ranges. {instrument_context}\n"
                    "{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in TOOLS]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        result, report = take_turn(prompt, llm, TOOLS, state["messages"])

        return {
            "messages": [result],
            "macro_report": report,
        }

    return macro_analyst_node
