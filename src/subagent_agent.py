"""Create a supervisor agent with sub-agent delegation capabilities.

This module demonstrates how to:
- Build a supervisor agent that coordinates research through sub-agents
- Use context isolation to prevent context clash and confusion
- Enable parallel research execution through multiple sub-agents
- Delegate specialized tasks to focused sub-agents with isolated contexts
"""

import logging
import os
from datetime import datetime

from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_community.tools.tavily_search import TavilySearchResults

from src.mcp.server.mcp_server import (
    SUBAGENT_USAGE_INSTRUCTIONS,
    SubAgent,
    create_task_tool,
)
from src.state import DeepAgentState

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(os.path.join("..", ".env"), override=True)


# ============================================================================
# Configuration
# ============================================================================

# Limits for delegation
MAX_CONCURRENT_RESEARCH_UNITS = 3
MAX_RESEARCHER_ITERATIONS = 3

# Simple research instructions for sub-agents
SIMPLE_RESEARCH_INSTRUCTIONS = """You are a researcher. Research the topic provided to you. IMPORTANT: Just make a single call to the web_search tool and use the result provided by the tool to answer the provided topic."""


# ============================================================================
# Sub-agent Definitions
# ============================================================================

# Create research sub-agent configuration
research_sub_agent: SubAgent = {
    "name": "research-agent",
    "description": "Delegate research to the sub-agent researcher. Only give this researcher one topic at a time.",
    "prompt": SIMPLE_RESEARCH_INSTRUCTIONS,
    "tools": ["tavily_search_results_json"],
}


# ============================================================================
# Agent Creation
# ============================================================================

def create_subagent_supervisor():
    """Create a supervisor agent with sub-agent delegation capabilities.

    Returns:
        Compiled agent graph with task delegation tool and sub-agents
    """
    # Initialize model
    model = init_chat_model(model="openai:gpt-4o-mini", temperature=0.0)

    # Tools for sub-agents
    web_search = TavilySearchResults(max_results=3)
    sub_agent_tools = [web_search]

    # Create task tool to delegate tasks to sub-agents
    task_tool = create_task_tool(
        sub_agent_tools, [research_sub_agent], model, DeepAgentState
    )

    # Tools for supervisor agent
    delegation_tools = [task_tool]

    # Create supervisor agent with system prompt
    agent = create_agent(
        model,
        delegation_tools,
        system_prompt=SUBAGENT_USAGE_INSTRUCTIONS.format(
            max_concurrent_research_units=MAX_CONCURRENT_RESEARCH_UNITS,
            max_researcher_iterations=MAX_RESEARCHER_ITERATIONS,
            date=datetime.now().strftime("%a %b %-d, %Y"),
        ),
        state_schema=DeepAgentState,
    )

    return agent


if __name__ == "__main__":
    # Create the supervisor agent
    logger.info("Creating subagent supervisor")
    agent = create_subagent_supervisor()

    # Example usage
    logger.info("Invoking agent with example query")
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Give me an overview of Model Context Protocol (MCP).",
                }
            ],
        }
    )

    # Log results
    logger.info("=" * 80)
    logger.info("FINAL RESULT")
    logger.info("=" * 80)

    # Log the conversation
    for msg in result["messages"]:
        if hasattr(msg, "content") and msg.content:
            role = getattr(msg, "type", "unknown")
            logger.info(f"{role.upper()}: {msg.content}")

    # Log the files if any were created
    if result.get("files"):
        logger.info("=" * 80)
        logger.info("VIRTUAL FILESYSTEM")
        logger.info("=" * 80)
        for file_path, content in result.get("files", {}).items():
            logger.info(f"{file_path}:")
            logger.info("-" * 40)
            logger.info(content[:200] + ("..." if len(content) > 200 else ""))
