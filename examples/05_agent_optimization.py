"""
Demonstrates optimized LangChain agent implementation for Claude 3.7 Sonnet.
"""

import sys
import os
import time

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.client import create_optimized_client, create_claude_llm
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_react_agent
from config import BUDGET_BALANCED


def simple_calculator(expression):
    """Simple calculator tool"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {str(e)}"


def compare_agent_implementations():
    """
    Compare a complex vs. optimized agent implementation
    """
    client = create_optimized_client()

    # Create base LLM
    llm = create_claude_llm(
        client=client,
        thinking_budget=BUDGET_BALANCED
    )

    # Complex agent with verbose instructions
    complex_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a sophisticated financial advisor AI assistant with access to powerful tools. 
        Your task is to provide detailed, accurate financial advice and calculations.

        When using tools, follow these detailed steps:
        1. Carefully analyze the user's query to understand their financial question
        2. Determine which of the available tools is most appropriate for this specific query
        3. Format the input to the tool exactly as required
        4. If the tool returns an error, carefully review the error message and try again with corrected input
        5. Once you have the tool result, incorporate it into your comprehensive response
        6. Provide additional context and explanation beyond just the tool result
        7. Consider if additional tool calls are needed to fully answer the query
        8. Format your final answer with clear headings and sections

        Remember that you have access to a calculator tool that can perform basic mathematical operations.
        When using the calculator, you can perform addition, subtraction, multiplication, division, 
        exponentiation, and other operations supported by standard calculator syntax.

        Examples of valid calculator inputs:
        - 5 + 10 * 2
        - 1000 * (1 + 0.05) ** 30
        - 500000 * 0.045 / 12
        - 100000 / (1 - (1 + 0.04/12) ** (-30 * 12))

        Be thorough, precise, and always verify your calculations.
        """),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Optimized agent with minimal instructions
    optimized_prompt = ChatPromptTemplate.from_messages([
        ("system", "You're a financial advisor. Use tools when needed. Be concise."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Tool definition (same for both agents)
    tools = [
        Tool(
            name="calculator",
            func=simple_calculator,
            description="Calculator for math operations. Input: mathematical expression."
        )
    ]

    # Create complex agent
    complex_agent = create_react_agent(llm, tools, complex_prompt)
    complex_executor = AgentExecutor(
        agent=complex_agent,
        tools=tools,
        verbose=True,  # Show the agent's thought process
        max_iterations=10  # Allow many iterations
    )

    # Create optimized agent
    optimized_agent = create_react_agent(llm, tools, optimized_prompt)
    optimized_executor = AgentExecutor(
        agent=optimized_agent,
        tools=tools,
        verbose=True,  # Show the agent's thought process
        max_iterations=3,  # Limit iterations
        early_stopping_method="generate"  # Stop early when possible
    )

    # Test query requiring calculation
    test_query = """
    If I invest $10,000 in an account with 6% annual interest compounded monthly,
    how much will I have after 20 years?
    """

    print("Testing complex agent implementation...")
    start_time = time.time()
    complex_result = complex_executor.invoke({"input": test_query})
    complex_time = time.time() - start_time
    print(f"\nResponse time: {complex_time:.2f} seconds")

    print("\n" + "=" * 50 + "\n")

    print("Testing optimized agent implementation...")
    start_time = time.time()
    optimized_result = optimized_executor.invoke({"input": test_query})
    optimized_time = time.time() - start_time
    print(f"\nResponse time: {optimized_time:.2f} seconds")

    print("\n" + "=" * 50 + "\n")

    print("Performance comparison:")
    print(f"Complex agent: {complex_time:.2f} seconds")
    print(f"Optimized agent: {optimized_time:.2f} seconds")
    improvement = (1 - optimized_time / complex_time) * 100
    print(f"Improvement: {improvement:.1f}%")

    print("\nKey optimizations:")
    print("1. Minimal instructions (reduced token count)")
    print("2. Limited iterations (prevents overthinking)")
    print("3. Early stopping (avoids unnecessary steps)")
    print("4. Focused tool descriptions")


if __name__ == "__main__":
    compare_agent_implementations()