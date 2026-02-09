"""Full research agent with sub-agent delegation and context isolation.

This module implements a complete research agent that:
- Uses TODOs to track tasks
- Stores raw search results in files for context offloading
- Delegates research tasks to specialized sub-agents with isolated contexts
- Integrates with MCP server tools including Tavily search
"""

import logging
import os
from datetime import datetime
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

from src.state import DeepAgentState

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(override=True)

# ============================================================================
# System Prompts - Imported from MCP server
# ============================================================================

from src.mcp.server.mcp_server import (
    get_today_str,
    TODO_USAGE_INSTRUCTIONS,
    FILE_USAGE_INSTRUCTIONS,
    SUBAGENT_USAGE_INSTRUCTIONS,
)

# ============================================================================
# Agent Configuration
# ============================================================================

# Import tools from MCP server instead of redefining them
from src.mcp.server.mcp_server import (
    think_tool,
    tavily_search,
    ls,
    read_file,
    write_file,
    write_todos,
    read_todos,
)


def create_research_agent():
    """Create and configure the full research agent with sub-agents.

    Returns:
        Configured LangGraph agent ready to handle research tasks
    """
    # Initialize model (use explicit provider prefix, consistent with your other modules)
    model = init_chat_model(model="openai:gpt-5-mini", temperature=0.0)

    # Configuration limits
    max_concurrent_research_units = 3
    max_researcher_iterations = 3

    # Build main agent prompt
    subagent_instructions = SUBAGENT_USAGE_INSTRUCTIONS.format(
        max_concurrent_research_units=max_concurrent_research_units,
        max_researcher_iterations=max_researcher_iterations,
    )

    main_prompt = (
        "# TODO MANAGEMENT\n"
        + TODO_USAGE_INSTRUCTIONS
        + "\n\n"
        + "=" * 80
        + "\n\n"
        + "# FILE SYSTEM USAGE\n"
        + FILE_USAGE_INSTRUCTIONS
        + "\n\n"
        + "=" * 80
        + "\n\n"
        + "# SUB-AGENT DELEGATION\n"
        + subagent_instructions
        + f"\n\nToday's date is {get_today_str()}"
    )

    # IMPORTANT: tools must be actual Tool objects/callables, not strings
    tools = [
        tavily_search,
        think_tool,
        ls,
        read_file,
        write_file,
        write_todos,
        read_todos,
        # "task" tool intentionally omitted until you implement real sub-agent delegation
    ]

    agent = create_agent(
        model,
        tools=tools,
        system_prompt=main_prompt,
        state_schema=DeepAgentState,
    )

    return agent


def run_research_query(query: str):
    """Run a research query through the agent.

    Args:
        query: The research question or topic to investigate

    Returns:
        Agent result containing messages, files, and todos
    """
    agent = create_research_agent()

    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result


if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Example usage
    logger.info("Running research query...")
    result = run_research_query("Give me an overview of Model Context Protocol (MCP).")

    # Log the final response
    logger.info("=" * 80)
    logger.info("FINAL RESPONSE")
    logger.info("=" * 80)
    logger.info(result["messages"][-1].content)

    # Log created files
    if "files" in result and result["files"]:
        logger.info("=" * 80)
        logger.info("FILES CREATED")
        logger.info("=" * 80)
        for filename in result["files"].keys():
            logger.info(f"  - {filename}")
