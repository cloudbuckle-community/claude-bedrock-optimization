"""
Demonstrates the impact of different reasoning budgets on Claude 3.7 Sonnet.
"""

import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.client import (
    create_optimized_client,
    get_standard_llm,
    get_fast_thinking_llm,
    get_balanced_thinking_llm,
    get_deep_thinking_llm
)
from utils.performance import compare_llms
from utils.prompts import FINANCIAL_QUESTIONS
from config import BUDGET_MINIMAL, BUDGET_BALANCED, BUDGET_DEEP


def demonstrate_reasoning_budgets():
    """
    Compare the performance of different reasoning budgets
    """
    # Create client with optimized timeout
    client = create_optimized_client()

    # Create different LLM configurations
    llms = {
        "Standard Mode (No Reasoning)": get_standard_llm(client),
        f"Fast Thinking ({BUDGET_MINIMAL} tokens)": get_fast_thinking_llm(client),
        f"Balanced Thinking ({BUDGET_BALANCED} tokens)": get_balanced_thinking_llm(client),
        f"Deep Thinking ({BUDGET_DEEP} tokens)": get_deep_thinking_llm(client)
    }

    print("Comparing different reasoning budgets for Claude 3.7 Sonnet")
    print("Testing with financial questions of increasing complexity...\n")

    # Use first three financial questions (simple to complex)
    test_prompts = FINANCIAL_QUESTIONS[:3]

    # Run comparison
    results = compare_llms(llms, test_prompts)

    print("\nSummary of average response times:")
    for llm_name, llm_results in results.items():
        times = [result["average_time"] for result in llm_results.values()]
        avg_time = sum(times) / len(times)
        print(f"{llm_name}: {avg_time:.2f} seconds")

    print("\nThe key tradeoff is between response time and reasoning quality.")
    print("For simple queries, standard mode or minimal thinking budget is sufficient.")
    print("For complex financial calculations, a larger thinking budget improves accuracy.")
    print("Consider using a dynamic approach based on query complexity.")


if __name__ == "__main__":
    demonstrate_reasoning_budgets()
    