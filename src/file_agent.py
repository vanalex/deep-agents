"""Create a ReAct agent with file system capabilities for context offloading.

This module demonstrates how to:
- Build a ReAct agent with virtual file system tools (ls, read_file, write_file)
- Use file operations for context offloading in long-running tasks
- Store and retrieve information across agent interactions
- Integrate file tools with research/web search capabilities
"""

import logging
import os

from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from src.mcp.server.mcp_server import (
    FILE_USAGE_INSTRUCTIONS,
    ls as mcp_ls,
    read_file as mcp_read_file,
    write_file as mcp_write_file,
)
from state import DeepAgentState

# Load environment variables
load_dotenv(os.path.join("..", ".env"), override=True)

# Wrap MCP tools with @tool decorator
ls = tool(mcp_ls)
read_file = tool(mcp_read_file)
write_file = tool(mcp_write_file)


# ============================================================================
# Web Search Tool
# ============================================================================

from langchain_community.tools.tavily_search import TavilySearchResults

# Create Tavily search tool
web_search = TavilySearchResults(max_results=3)


# ============================================================================
# Agent Creation
# ============================================================================

SIMPLE_RESEARCH_INSTRUCTIONS = """IMPORTANT: Just make a single call to the web_search tool and use the result provided by the tool to answer the user's question."""


def create_file_agent():
    """Create a ReAct agent with file system capabilities.

    Returns:
        Compiled agent graph with file tools and web search
    """
    # Full prompt
    instructions = (
        FILE_USAGE_INSTRUCTIONS
        + "\n\n"
        + "=" * 80
        + "\n\n"
        + SIMPLE_RESEARCH_INSTRUCTIONS
    )

    # Use explicit provider prefix to avoid model-resolution issues
    model = init_chat_model(model="openai:gpt-4o-mini", temperature=0.0)
    tools = [ls, read_file, write_file, web_search]

    agent = create_agent(
        model,
        tools,
        system_prompt=instructions,
        state_schema=DeepAgentState,
    )

    return agent


if __name__ == "__main__":
    # Create the agent
    logger.info("Creating file agent")
    agent = create_file_agent()

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
            "files": {},
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

    # Log the files
    logger.info("=" * 80)
    logger.info("VIRTUAL FILESYSTEM")
    logger.info("=" * 80)
    for file_path, content in result.get("files", {}).items():
        logger.info(f"{file_path}:")
        logger.info("-" * 40)
        logger.info(content[:200] + ("..." if len(content) > 200 else ""))
