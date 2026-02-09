"""Create a ReAct agent with calculator tools using LangGraph.

This module demonstrates how to:
- Build a ReAct agent with custom tools
- Access and modify agent state within tools
- Use InjectedState and InjectedToolCallId annotations
- Import tools from MCP server
"""

import logging
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from src.mcp.server.mcp_server import calculate, calculate_wstate, CalcState as MCPCalcState

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(os.path.join("..", ".env"), override=True)


# Import tools directly from MCP server
calculator = tool(calculate)
calculator_wstate = tool(calculate_wstate)
CalcState = MCPCalcState


def create_simple_agent():
    """Create a simple ReAct agent without state tracking."""
    SYSTEM_PROMPT = "You are a helpful arithmetic assistant who is an expert at using a calculator."

    model = init_chat_model(model="openai:gpt-5-mini", temperature=0.0)
    tools = [calculator]

    # Create agent
    agent = create_react_agent(
        model,
        tools,
        prompt=SYSTEM_PROMPT,
    ).with_config({"recursion_limit": 20})

    return agent


def create_stateful_agent():
    """Create a ReAct agent with state tracking."""
    SYSTEM_PROMPT = "You are a helpful arithmetic assistant who is an expert at using a calculator."

    model = init_chat_model(model="openai:gpt-4o-mini", temperature=0.0)
    tools = [calculator_wstate]

    # Create agent
    agent = create_react_agent(
        model,
        tools,
        prompt=SYSTEM_PROMPT,
        state_schema=CalcState,
    ).with_config({"recursion_limit": 20})

    return agent


if __name__ == "__main__":
    # Example usage
    agent = create_stateful_agent()

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What is 3.1 * 4.2?",
                }
            ],
        }
    )

    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info(f"Result: {result}")
    logger.info(f"Operations performed: {result.get('ops', [])}")
