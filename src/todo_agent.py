"""Create a ReAct agent with TODO tracking capabilities.

This module demonstrates how to:
- Build a ReAct agent with TODO management tools
- Use write_todos and read_todos tools for task planning
- Track progress through complex multi-step workflows
- Integrate with external tools (like web search)
"""

import logging
import os

from dotenv import load_dotenv
from typing import Annotated, NotRequired

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from src.mcp.server.mcp_server import (
    TODO_USAGE_INSTRUCTIONS,
    read_todos as mcp_read_todos,
    write_todos as mcp_write_todos,
)

# Load environment variables
load_dotenv(os.path.join("..", ".env"), override=True)


def file_reducer(left: dict[str, str] | None, right: dict[str, str] | None) -> dict[str, str] | None:
    """Merge two file dictionaries, with right side taking precedence."""
    if left is None:
        return right
    if right is None:
        return left
    return {**left, **right}


class DeepAgentState(TypedDict, total=False):
    """Application state schema (safe for Input/Output).

    IMPORTANT: Do NOT inherit from LangGraph/LangChain AgentState here, because some
    AgentState variants include managed channels (e.g., remaining_steps) which are
    forbidden in Input/Output schema.
    """

    messages: Annotated[list, add_messages]
    todos: NotRequired[list[dict]]
    files: Annotated[NotRequired[dict[str, str]], file_reducer]


# Create Tavily search tool
web_search = TavilySearchResults(max_results=3)

# Wrap MCP tools with @tool decorator
read_todos = tool(mcp_read_todos)
write_todos = tool(mcp_write_todos)


def create_todo_agent():
    """Create a ReAct agent with TODO tracking capabilities.

    Returns:
        Compiled agent graph with TODO tools and web search
    """
    SIMPLE_RESEARCH_INSTRUCTIONS = (
        "IMPORTANT: Just make a single call to the web_search tool and use the "
        "result provided by the tool to answer the user's question."
    )

    # Use explicit provider prefix to avoid model-resolution issues.
    model = init_chat_model(model="openai:gpt-5-mini", temperature=0.0)
    tools = [write_todos, web_search, read_todos]

    agent = create_agent(
        model,
        tools,
        system_prompt=TODO_USAGE_INSTRUCTIONS
        + "\n\n"
        + "=" * 80
        + "\n\n"
        + SIMPLE_RESEARCH_INSTRUCTIONS,
        state_schema=DeepAgentState,
    )

    return agent


if __name__ == "__main__":
    # Create the agent
    logger.info("Creating todo agent")
    agent = create_todo_agent()

    # Example usage
    logger.info("Invoking agent with example query")
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Give me a short summary of the Model Context Protocol (MCP).",
                }
            ],
            "todos": [],
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

    # Log the todos
    logger.info("=" * 80)
    logger.info("TODO LIST")
    logger.info("=" * 80)
    for i, todo in enumerate(result.get("todos", []), 1):
        status_emoji = {"pending": "⏳", "in_progress": "🔄", "completed": "✅"}
        emoji = status_emoji.get(todo["status"], "❓")
        logger.info(f"{i}. {emoji} {todo['content']} ({todo['status']})")
