"""
Demonstrates optimized LangChain agent implementation for Claude 3.7 Sonnet.
Uses a different LangChain agent format that's compatible with Claude.
"""

import sys
import os
import time
import re

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.client import create_optimized_client, create_claude_llm
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor, initialize_agent, AgentType
from config import BUDGET_BALANCED


def enhanced_calculator(expression):
    """
    Enhanced calculator tool that handles both Python syntax and common mathematical notation.
    Automatically converts ^ to ** for exponentiation and provides detailed error messages.
    """
    try:
        # Clean up the expression
        cleaned_expression = expression.strip()

        # Replace ^ with ** for exponentiation
        cleaned_expression = re.sub(r'(\d+|\))\s*\^\s*(\d+|\()', r'\1**\2', cleaned_expression)

        # Replace common mathematical functions if needed
        math_replacements = {
            "sin": "math.sin",
            "cos": "math.cos",
            "tan": "math.tan",
            "log": "math.log",
            "sqrt": "math.sqrt"
        }

        # Check if we need to import math
        need_math = any(func in cleaned_expression for func in math_replacements)

        if need_math:
            import math
            for func, replacement in math_replacements.items():
                # Only replace if it's a function call (followed by parenthesis)
                cleaned_expression = re.sub(r'\b' + func + r'\(', replacement + '(', cleaned_expression)

        # Execute the calculation
        result = eval(cleaned_expression)

        # Format the result nicely - round to 2 decimal places if it's a float
        if isinstance(result, float):
            # For financial calculations, often we want to display 2 decimal places
            result_str = f"{result:.2f}"
            # But if it's a round number, keep it simple
            if result_str.endswith('.00'):
                result_str = str(int(result))
            # If it has less than 2 decimal places, keep it as is
            elif float(result_str) != result:
                result_str = str(result)
        else:
            result_str = str(result)

        return result_str

    except Exception as e:
        # Provide helpful error messages based on common issues
        if "^" in expression and "**" not in cleaned_expression:
            return f"Error: There might still be an issue with the exponentiation syntax. Try using '**' instead of '^'. Details: {str(e)}"
        elif "(" in expression and ")" not in expression:
            return "Error: Mismatched parentheses. Check that all opening parentheses '(' have a closing ')'."
        elif ")" in expression and "(" not in expression:
            return "Error: Mismatched parentheses. Check that all closing parentheses ')' have an opening '('."
        else:
            return f"Error: {str(e)}. Try reformatting your expression."


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

    # Tool definition (same for both agents)
    tools = [
        Tool(
            name="calculator",
            func=enhanced_calculator,
            description="Calculator for math operations. Supports standard Python syntax and common mathematical notation including ^ for exponentiation."
        )
    ]

    # Create a complex agent with verbose instructions using initialize_agent
    # This avoids the issues with creating a custom prompt template
    complex_agent_executor = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        max_iterations=10,
        handle_parsing_errors=True,
        agent_kwargs={
            "prefix": """
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
            When using the calculator, you can use either Python syntax or common mathematical notation:
            - Addition: 5 + 10
            - Subtraction: 5 - 3
            - Multiplication: 5 * 2
            - Division: 10 / 2
            - Exponentiation: 2 ** 3 or 2 ^ 3 (either will work)
            
            Examples of valid calculator inputs:
            - 5 + 10 * 2
            - 1000 * (1 + 0.05) ** 30  or  1000 * (1 + 0.05) ^ 30
            - 500000 * 0.045 / 12
            - 100000 / (1 - (1 + 0.04/12) ** (-30 * 12))

            Be thorough, precise, and always verify your calculations.
            """
        }
    )

    # Create an optimized agent with minimal instructions
    optimized_agent_executor = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        max_iterations=3,
        early_stopping_method="generate",
        handle_parsing_errors=True,
        agent_kwargs={
            "prefix": """
            You're a financial advisor. Use tools when needed. Be concise.
            
            The calculator tool accepts standard math syntax and will convert ^ to ** for exponentiation automatically.
            """
        }
    )

    # Test query requiring calculation
    test_query = """
    If I invest $10,000 in an account with 6% annual interest compounded monthly,
    how much will I have after 20 years?
    """

    print("Testing complex agent implementation...")
    start_time = time.time()
    try:
        complex_result = complex_agent_executor.invoke({"input": test_query})
        complex_time = time.time() - start_time
        print(f"\nResponse time: {complex_time:.2f} seconds")
        print(f"\nComplex agent response:\n{complex_result['output']}")
    except Exception as e:
        complex_time = time.time() - start_time
        print(f"\nError with complex agent: {str(e)}")
        print(f"Time before error: {complex_time:.2f} seconds")

    print("\n" + "=" * 50 + "\n")

    print("Testing optimized agent implementation...")
    start_time = time.time()
    try:
        optimized_result = optimized_agent_executor.invoke({"input": test_query})
        optimized_time = time.time() - start_time
        print(f"\nResponse time: {optimized_time:.2f} seconds")
        print(f"\nOptimized agent response:\n{optimized_result['output']}")
    except Exception as e:
        optimized_time = time.time() - start_time
        print(f"\nError with optimized agent: {str(e)}")
        print(f"Time before error: {optimized_time:.2f} seconds")

    print("\n" + "=" * 50 + "\n")

    if 'complex_time' in locals() and 'optimized_time' in locals():
        print("Performance comparison:")
        print(f"Complex agent: {complex_time:.2f} seconds")
        print(f"Optimized agent: {optimized_time:.2f} seconds")
        if complex_time > 0:
            improvement = (1 - optimized_time / complex_time) * 100
            print(f"Improvement: {improvement:.1f}%")

    print("\nKey optimizations:")
    print("1. Minimal instructions (reduced token count)")
    print("2. Limited iterations (prevents overthinking)")
    print("3. Early stopping (avoids unnecessary steps)")
    print("4. Focused tool descriptions")
    print("5. Enhanced calculator that handles common mathematical notation")


if __name__ == "__main__":
    compare_agent_implementations()
