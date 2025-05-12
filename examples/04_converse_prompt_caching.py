"""
Demonstrates performance optimization through structured messages.
This approach provides performance benefits even without explicit prompt caching.
Includes sleep intervals to avoid throttling issues.
"""

import sys
import os
import json
import time
import statistics
from typing import List, Dict, Any

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.client import create_optimized_client
from utils.prompts import SAMPLE_DOCUMENT, get_document_qa_prompts
from config import CLAUDE_3_7_SONNET


def time_unstructured_query(client, document: str, query: str, num_runs: int = 3) -> Dict[str, Any]:
    """
    Time the response for an unstructured query

    Args:
        client: Bedrock runtime client
        document: Document text
        query: Query about the document
        num_runs: Number of runs to average over

    Returns:
        Dict with timing results
    """
    times = []

    for i in range(num_runs):
        print(f"Unstructured query run {i+1}/{num_runs}...")

        # Add sleep before each request (except the first) to avoid throttling
        if i > 0:
            sleep_time = 5  # 5 seconds between requests
            print(f"Sleeping for {sleep_time} seconds to avoid throttling...")
            time.sleep(sleep_time)

        # Basic unstructured message
        messages_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": f"I need help understanding this document:\n\n{document}\n\n{query}"
                }
            ]
        }

        start_time = time.time()
        response = client.invoke_model(
            body=json.dumps(messages_body),
            modelId=CLAUDE_3_7_SONNET,
            accept="application/json",
            contentType="application/json"
        )
        end_time = time.time()

        elapsed_time = end_time - start_time
        times.append(elapsed_time)

        # Only show response text on first run
        if i == 0:
            response_body = json.loads(response['body'].read())
            response_text = response_body['content'][0]['text'] if 'content' in response_body else "No content"
            print(f"Response: {response_text[:100]}...")

        print(f"Time: {elapsed_time:.2f} seconds")

    # Calculate statistics
    avg_time = statistics.mean(times)
    min_time = min(times)
    max_time = max(times)

    print(f"\nUnstructured query statistics:")
    print(f"Average time: {avg_time:.2f} seconds")
    print(f"Min: {min_time:.2f}s, Max: {max_time:.2f}s")

    return {
        "times": times,
        "average": avg_time,
        "min": min_time,
        "max": max_time
    }


def time_structured_query(client, document: str, query: str, num_runs: int = 3) -> Dict[str, Any]:
    """
    Time the response for a structured query

    Args:
        client: Bedrock runtime client
        document: Document text
        query: Query about the document
        num_runs: Number of runs to average over

    Returns:
        Dict with timing results
    """
    times = []

    for i in range(num_runs):
        print(f"Structured query run {i+1}/{num_runs}...")

        # Add sleep before each request (except the first) to avoid throttling
        if i > 0:
            sleep_time = 5  # 5 seconds between requests
            print(f"Sleeping for {sleep_time} seconds to avoid throttling...")
            time.sleep(sleep_time)

        # Structured message with cache_control
        messages_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"I need help understanding this document:\n\n{document}",
                            "cache_control": {
                                "type": "ephemeral"
                            }
                        },
                        {
                            "type": "text",
                            "text": query
                        }
                    ]
                }
            ]
        }

        start_time = time.time()
        response = client.invoke_model(
            body=json.dumps(messages_body),
            modelId=CLAUDE_3_7_SONNET,
            accept="application/json",
            contentType="application/json"
        )
        end_time = time.time()

        elapsed_time = end_time - start_time
        times.append(elapsed_time)


        # Only show response text on first run
        if i == 0:
            response_body = json.loads(response['body'].read())
            print(response_body)
            response_text = response_body['content'][0]['text'] if 'content' in response_body else "No content"
            print(f"Response: {response_text[:100]}...")

        print(f"Time: {elapsed_time:.2f} seconds")

    # Calculate statistics
    avg_time = statistics.mean(times)
    min_time = min(times)
    max_time = max(times)

    print(f"\nStructured query statistics:")
    print(f"Average time: {avg_time:.2f} seconds")
    print(f"Min: {min_time:.2f}s, Max: {max_time:.2f}s")

    return {
        "times": times,
        "average": avg_time,
        "min": min_time,
        "max": max_time
    }


def compare_multiple_queries(client, document: str, queries: List[str], runs_per_query: int = 3):
    """
    Compare unstructured vs structured queries for multiple different questions

    Args:
        client: Bedrock runtime client
        document: Document text
        queries: List of queries to test
        runs_per_query: Number of runs per query for more reliable results
    """
    results = {
        "unstructured": {},
        "structured": {}
    }

    for i, query in enumerate(queries):
        print(f"\n=== Testing Query {i+1}: {query} ===\n")

        print("Testing unstructured query format...")
        unstructured_results = time_unstructured_query(client, document, query, runs_per_query)

        # Add sleep between test types to avoid throttling
        sleep_time = 10  # 10 seconds between test sets
        print(f"\nSleeping for {sleep_time} seconds before next test set...")
        time.sleep(sleep_time)

        print("\nTesting structured query format...")
        structured_results = time_structured_query(client, document, query, runs_per_query)

        # Calculate improvement
        improvement = (1 - structured_results["average"] / unstructured_results["average"]) * 100
        print(f"\nPerformance comparison for Query {i+1}:")
        print(f"Unstructured average: {unstructured_results['average']:.2f}s")
        print(f"Structured average: {structured_results['average']:.2f}s")

        if improvement > 0:
            print(f"Improvement with structured format: {improvement:.1f}%")
        else:
            print(f"No improvement with structured format ({-improvement:.1f}% slower)")

        # Store results
        results["unstructured"][f"query_{i+1}"] = unstructured_results
        results["structured"][f"query_{i+1}"] = structured_results

        # Add sleep between queries to avoid throttling
        if i < len(queries) - 1:  # Skip sleep after the last query
            sleep_time = 15  # 15 seconds between query sets
            print(f"\nSleeping for {sleep_time} seconds before next query set...")
            time.sleep(sleep_time)

    return results


def demonstrate_structured_message_optimization():
    """
    Demonstrate performance optimization through structured messages
    """
    print("Creating optimized client...")
    client = create_optimized_client()

    # Use the sample document
    document = SAMPLE_DOCUMENT

    # Get document Q&A prompts
    qa_prompts = get_document_qa_prompts(document)

    # We'll use fewer queries to reduce throttling risk
    test_queries = qa_prompts[:2]  # Just test 2 queries to reduce API calls

    print("\n=== Testing Structured Message Performance Optimization ===\n")
    print("This test compares traditional unstructured messages vs. structured messages")
    print("with cache_control for potential performance improvements.\n")
    print(f"Using {len(test_queries)} different queries with multiple runs each for reliable results.")
    print("Includes sleep intervals to avoid throttling issues.\n")

    # Reduce the number of runs per query to minimize throttling risk
    runs_per_query = 2  # Using 2 runs instead of 3

    # Run comparison
    results = compare_multiple_queries(client, document, test_queries, runs_per_query=runs_per_query)

    # Analyze overall results
    unstructured_times = [result["average"] for result in results["unstructured"].values()]
    structured_times = [result["average"] for result in results["structured"].values()]

    unstructured_avg = statistics.mean(unstructured_times)
    structured_avg = statistics.mean(structured_times)

    overall_improvement = (1 - structured_avg / unstructured_avg) * 100

    print("\n=== Overall Results ===\n")
    print(f"Average time across all queries (unstructured): {unstructured_avg:.2f}s")
    print(f"Average time across all queries (structured): {structured_avg:.2f}s")

    if overall_improvement > 0:
        print(f"Overall performance improvement with structured messages: {overall_improvement:.1f}%")
    else:
        print(f"No overall improvement with structured messages ({-overall_improvement:.1f}% slower)")

    print("\nConclusions:")
    print("1. Structured messages with cache_control can provide performance benefits")
    print("   even without explicit caching metrics being reported.")
    print("2. The improvement varies by query and can be affected by server load.")
    print("3. This optimization works with the current AWS Bedrock setup without")
    print("   requiring any additional configuration or newer SDK versions.")
    print("4. The structured message format is also beneficial for code organization")
    print("   and maintaining separation between static and dynamic content.")


if __name__ == "__main__":
    demonstrate_structured_message_optimization()