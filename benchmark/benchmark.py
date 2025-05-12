"""
Benchmarking for Claude 3.7 Sonnet optimization techniques.
"""

import sys
import os
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.client import (
    create_optimized_client,
    get_standard_llm,
    get_fast_thinking_llm,
    get_balanced_thinking_llm,
    get_deep_thinking_llm,
    get_cached_llm,
    get_optimal_llm
)
from utils.performance import time_llm_response, compare_llms, time_direct_invoke
from utils.prompts import FINANCIAL_QUESTIONS, SAMPLE_DOCUMENT, get_document_qa_prompts
from config import CLAUDE_3_7_SONNET


def benchmark_reasoning_budgets():
    """
    Benchmark different reasoning budgets
    """
    print("Benchmarking different reasoning budgets...")

    client = create_optimized_client()

    llms = {
        "Standard Mode": get_standard_llm(client),
        "Fast Thinking": get_fast_thinking_llm(client),
        "Balanced Thinking": get_balanced_thinking_llm(client),
        "Deep Thinking": get_deep_thinking_llm(client)
    }

    # Test with all financial questions
    results = compare_llms(llms, FINANCIAL_QUESTIONS, display_results=False)

    # Prepare results for DataFrame
    df_data = []

    for prompt_idx, prompt in enumerate(FINANCIAL_QUESTIONS):
        prompt_name = f"Prompt {prompt_idx + 1}"
        for llm_name, llm_results in results.items():
            result = llm_results[prompt_name]
            df_data.append({
                "LLM": llm_name,
                "Prompt": prompt_name,
                "PromptComplexity": prompt_idx + 1,  # 1 = simple, 4 = complex
                "ResponseTime": result["average_time"]
            })

    df = pd.DataFrame(df_data)

    # Create results directory if it doesn't exist
    os.makedirs("benchmark/results", exist_ok=True)

    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"benchmark/results/reasoning_budget_benchmark_{timestamp}.csv"
    df.to_csv(csv_file, index=False)

    # Generate visualization
    plt.figure(figsize=(12, 8))

    for llm_name in df["LLM"].unique():
        llm_data = df[df["LLM"] == llm_name]
        plt.plot(
            llm_data["PromptComplexity"],
            llm_data["ResponseTime"],
            marker="o",
            label=llm_name
        )

    plt.title("Response Time by Reasoning Mode and Prompt Complexity")
    plt.xlabel("Prompt Complexity (1 = Simple, 4 = Complex)")
    plt.ylabel("Response Time (seconds)")
    plt.xticks([1, 2, 3, 4])
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    # Save visualization
    plt_file = f"benchmark/results/reasoning_budget_benchmark_{timestamp}.png"
    plt.savefig(plt_file)

    print(f"Results saved to {csv_file}")
    print(f"Visualization saved to {plt_file}")

    return df


def benchmark_prompt_caching():
    """
    Benchmark prompt caching performance
    """
    print("Benchmarking prompt caching...")

    client = create_optimized_client()
    document = SAMPLE_DOCUMENT
    qa_prompts = get_document_qa_prompts(document)

    # Prepare API call structure
    def create_body(query, use_caching=False):
        return {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "I need help understanding this document:\n\n" + document,
                            "cache_control": {
                                "type": "ephemeral"
                            } if use_caching else {}
                        },
                        {
                            "type": "text",
                            "text": query
                        }
                    ]
                }
            ]
        }

    # Test each prompt with and without caching
    results = []

    # Without caching
    print("Testing without caching...")
    for i, prompt in enumerate(qa_prompts):
        print(f"Prompt {i + 1}/4...")
        body = create_body(prompt, use_caching=False)
        result = time_direct_invoke(
            client=client,
            body=body,
            model_id=CLAUDE_3_7_SONNET,
            verbose=False
        )
        results.append({
            "Query": i + 1,
            "Caching": "No",
            "ResponseTime": result["average_time"],
            "Usage": result["usages"][0] if result["usages"] else {}
        })

    # First with caching (cache write)
    print("Testing with caching (first run - cache write)...")
    for i, prompt in enumerate(qa_prompts):
        print(f"Prompt {i + 1}/4...")
        body = create_body(prompt, use_caching=True)
        result = time_direct_invoke(
            client=client,
            body=body,
            model_id=CLAUDE_3_7_SONNET,
            verbose=False
        )
        results.append({
            "Query": i + 1,
            "Caching": "First (Write)",
            "ResponseTime": result["average_time"],
            "Usage": result["usages"][0] if result["usages"] else {}
        })

    # Second with caching (cache read)
    print("Testing with caching (second run - cache read)...")
    for i, prompt in enumerate(qa_prompts):
        print(f"Prompt {i + 1}/4...")
        body = create_body(prompt, use_caching=True)
        result = time_direct_invoke(
            client=client,
            body=body,
            model_id=CLAUDE_3_7_SONNET,
            verbose=False
        )
        results.append({
            "Query": i + 1,
            "Caching": "Second (Read)",
            "ResponseTime": result["average_time"],
            "Usage": result["usages"][0] if result["usages"] else {}
        })

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Add cache statistics
    df["CacheRead"] = df["Usage"].apply(
        lambda x: x.get("cacheReadInputTokens", 0) if isinstance(x, dict) else 0
    )
    df["CacheWrite"] = df["Usage"].apply(
        lambda x: x.get("cacheWriteInputTokens", 0) if isinstance(x, dict) else 0
    )

    # Create results directory if it doesn't exist
    os.makedirs("benchmark/results", exist_ok=True)

    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"benchmark/results/prompt_caching_benchmark_{timestamp}.csv"
    df.to_csv(csv_file, index=False)

    # Generate visualization
    plt.figure(figsize=(12, 8))

    # Group data for plotting
    no_cache = df[df["Caching"] == "No"]["ResponseTime"].mean()
    cache_write = df[df["Caching"] == "First (Write)"]["ResponseTime"].mean()
    cache_read = df[df["Caching"] == "Second (Read)"]["ResponseTime"].mean()

    plt.bar(
        ["No Cache", "Cache Write", "Cache Read"],
        [no_cache, cache_write, cache_read]
    )

    plt.title("Average Response Time by Caching Mode")
    plt.ylabel("Response Time (seconds)")
    plt.grid(True, linestyle="--", alpha=0.5, axis="y")

    for i, v in enumerate([no_cache, cache_write, cache_read]):
        plt.text(i, v + 0.1, f"{v:.2f}s", ha="center")

    # Calculate improvement percentage
    improvement = (1 - cache_read / no_cache) * 100
    plt.figtext(
        0.5, 0.01,
        f"Response time improvement with caching: {improvement:.1f}%",
        ha="center",
        fontsize=12
    )

    # Save visualization
    plt_file = f"benchmark/results/prompt_caching_benchmark_{timestamp}.png"
    plt.savefig(plt_file)

    print(f"Results saved to {csv_file}")
    print(f"Visualization saved to {plt_file}")

    return df


def benchmark_combined_optimizations():
    """
    Benchmark combined optimizations
    """
    print("Benchmarking combined optimizations...")

    client = create_optimized_client()

    llms = {
        "Standard (No Optimizations)": get_standard_llm(client),
        "Balanced Thinking Only": get_balanced_thinking_llm(client),
        "Caching Only": get_cached_llm(client),
        "Optimized (Thinking + Caching)": get_optimal_llm(client)
    }

    # Test with all financial questions
    results = compare_llms(llms, FINANCIAL_QUESTIONS, display_results=False)

    # Prepare results for DataFrame
    df_data = []

    for prompt_idx, prompt in enumerate(FINANCIAL_QUESTIONS):
        prompt_name = f"Prompt {prompt_idx + 1}"
        for llm_name, llm_results in results.items():
            result = llm_results[prompt_name]
            df_data.append({
                "LLM": llm_name,
                "Prompt": prompt_name,
                "PromptComplexity": prompt_idx + 1,  # 1 = simple, 4 = complex
                "ResponseTime": result["average_time"]
            })

    df = pd.DataFrame(df_data)

    # Create results directory if it doesn't exist
    os.makedirs("benchmark/results", exist_ok=True)

    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"benchmark/results/combined_optimizations_benchmark_{timestamp}.csv"
    df.to_csv(csv_file, index=False)

    # Generate visualization
    plt.figure(figsize=(12, 8))

    # Calculate average time per LLM
    avg_times = df.groupby("LLM")["ResponseTime"].mean().reset_index()

    plt.bar(
        avg_times["LLM"],
        avg_times["ResponseTime"]
    )

    plt.title("Average Response Time by Optimization Strategy")
    plt.ylabel("Response Time (seconds)")
    plt.xticks(rotation=15, ha="right")
    plt.grid(True, linestyle="--", alpha=0.5, axis="y")

    for i, v in enumerate(avg_times["ResponseTime"]):
        plt.text(i, v + 0.1, f"{v:.2f}s", ha="center")

    # Calculate improvement percentage
    baseline = avg_times[avg_times["LLM"] == "Standard (No Optimizations)"]["ResponseTime"].values[0]
    best = avg_times[avg_times["LLM"] == "Optimized (Thinking + Caching)"]["ResponseTime"].values[0]
    improvement = (1 - best / baseline) * 100

    plt.figtext(
        0.5, 0.01,
        f"Overall response time improvement with combined optimizations: {improvement:.1f}%",
        ha="center",
        fontsize=12
    )

    # Save visualization
    plt_file = f"benchmark/results/combined_optimizations_benchmark_{timestamp}.png"
    plt.savefig(plt_file)

    print(f"Results saved to {csv_file}")
    print(f"Visualization saved to {plt_file}")

    return df


def run_all_benchmarks():
    """
    Run all benchmarks
    """
    print("Running all benchmarks...")

    # Create results directory
    os.makedirs("benchmark/results", exist_ok=True)

    # Run benchmarks
    reasoning_results = benchmark_reasoning_budgets()
    print("\n" + "=" * 50 + "\n")

    caching_results = benchmark_prompt_caching()
    print("\n" + "=" * 50 + "\n")

    combined_results = benchmark_combined_optimizations()
    print("\n" + "=" * 50 + "\n")

    print("All benchmarks completed.")
    print("Results saved to the benchmark/results directory.")


if __name__ == "__main__":
    run_all_benchmarks()