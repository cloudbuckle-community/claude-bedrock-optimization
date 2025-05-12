"""
Demonstrates proper API integration for Claude 3.7 Sonnet with LangChain.
"""

import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.client import create_optimized_client, create_claude_llm
from utils.performance import time_llm_response
from utils.prompts import FINANCIAL_QUESTIONS


def demonstrate_correct_integration():
    """
    Show the proper LangChain integration for Claude 3.7 Sonnet
    """
    # Create client with default settings (optimized timeout)
    client = create_optimized_client()

    # Create Claude 3.7 Sonnet LLM with proper integration
    llm = create_claude_llm(client=client)

    print("Using ChatBedrockConverse with proper configuration for Claude 3.7 Sonnet")
    print("Testing with a simple financial question...\n")

    # Use a simple financial question for testing
    prompt = FINANCIAL_QUESTIONS[0]

    # Time the response
    result = time_llm_response(llm, prompt)

    print("\nResponse:")
    print(result.get("response", "No response received"))

    print("\nCorrect integration ensures:")
    print("1. Using the Messages API format required by Claude 3.7")
    print("2. Proper handling of response format")
    print("3. Support for all Claude 3.7 capabilities")


if __name__ == "__main__":
    demonstrate_correct_integration()