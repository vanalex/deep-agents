"""FastMCP server for calculator agent.

This server exposes calculator tools through the Model Context Protocol,
allowing AI assistants to perform arithmetic operations.
"""

from typing import Annotated, List, Literal, Union

from fastmcp import FastMCP
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.types import Command

# Create MCP server
mcp = FastMCP("calculator-agent")


def reduce_list(left: list | None, right: list | None) -> list:
    """Safely combine two lists, handling cases where either or both inputs might be None.

    Args:
        left (list | None): The first list to combine, or None.
        right (list | None): The second list to combine, or None.

    Returns:
        list: A new list containing all elements from both input lists.
               If an input is None, it's treated as an empty list.
    """
    if not left:
        left = []
    if not right:
        right = []
    return left + right


class CalcState(AgentState):
    """Extended agent state with operation history."""
    ops: Annotated[List[str], reduce_list]


def calculate(
    operation: Literal["add", "subtract", "multiply", "divide"],
    a: Union[int, float],
    b: Union[int, float],
) -> Union[int, float, dict]:
    """Perform basic arithmetic operations.

    Args:
        operation: The operation to perform ('add', 'subtract', 'multiply', 'divide')
        a: The first number
        b: The second number

    Returns:
        The result of the operation or an error dict
    """
    if operation == 'divide' and b == 0:
        return {"error": "Division by zero is not allowed."}

    if operation == 'add':
        result = a + b
    elif operation == 'subtract':
        result = a - b
    elif operation == 'multiply':
        result = a * b
    elif operation == 'divide':
        result = a / b
    else:
        return {"error": "Unknown operation"}

    return result


def calculate_wstate(
    operation: Literal["add", "subtract", "multiply", "divide"],
    a: Union[int, float],
    b: Union[int, float],
    state: Annotated[CalcState, InjectedState],   # not sent to LLM
    tool_call_id: Annotated[str, InjectedToolCallId]  # not sent to LLM
) -> Command:
    """Perform arithmetic operations with state tracking.

    Args:
        operation: The operation to perform ('add', 'subtract', 'multiply', 'divide')
        a: The first number
        b: The second number

    Returns:
        Command with updated state including operation history
    """
    # Use the calculate function for the actual computation
    result = calculate(operation, a, b)

    ops = [f"({operation}, {a}, {b}),"]
    return Command(
        update={
            "ops": ops,
            "messages": [
                ToolMessage(f"{result}", tool_call_id=tool_call_id)
            ],
        }
    )


# Register tools with MCP server
mcp.tool()(calculate)
mcp.tool()(calculate_wstate)


if __name__ == "__main__":
    mcp.run()
