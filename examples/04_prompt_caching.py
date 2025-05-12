"""
Demonstrates the effect of prompt caching on response time.
"""

import sys
import os
import json
import time

import boto3

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.client import create_optimized_client
from utils.prompts import SAMPLE_DOCUMENT, get_document_qa_prompts
from config import CLAUDE_3_7_SONNET


def time_response(bedrock_client, document, query, use_caching=False):
    """
    Time the response for a query, with or without caching
    """
    if use_caching:
        print(f"Using content structure for caching")

        # Use structured content with cache control
        messages_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "I need help understanding this document:\n\n",
                            "cache_control": {
                                "type": "ephemeral"
                            }
                        },
                        {
                            "type": "text",
                            "text": document,
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
    else:
        # Non-cached version (unchanged)
        print(f"Not using caching")
        messages_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "messages": [
                {
                    "role": "user",
                    "content": f"I need help understanding this document:\n\n{document}\n\n{query}"
                }
            ]
        }

    start_time = time.time()
    bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
    response = bedrock.invoke_model(
        body=json.dumps(messages_body),
        modelId=CLAUDE_3_7_SONNET,
        accept="application/json",
        contentType="application/json"
    )
    end_time = time.time()

    response_body = json.loads(response['body'].read())
    response_text = response_body['content'][0]['text'] if 'content' in response_body else "No content"
    print(json.dumps(response_body, indent=2))

    return {
        "time": end_time - start_time,
        "text": response_text,
        "usage": response_body.get('usage', {})
    }


def test_basic_caching(client):
    """
    A very basic test of caching functionality
    """
    print("\n=== BASIC CACHING TEST ===")

    # Simple content for caching test
    cached_content = "This is a cached text that will be reused across requests."

    # First request without caching
    print("1. Without cache:")
    no_cache_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": f"{cached_content}\n\nWhat does the text say?"
            }
        ]
    }

    start_time = time.time()
    response = client.invoke_model(
        body=json.dumps(no_cache_body),
        modelId=CLAUDE_3_7_SONNET,
        accept="application/json",
        contentType="application/json"
    )
    end_time = time.time()

    response_body = json.loads(response['body'].read())
    result1_time = end_time - start_time
    result1_usage = response_body.get('usage', {})

    print(f"Time: {result1_time:.2f}s")
    print(f"Usage: {result1_usage}")

    # Second request with structured content for potential caching
    print("\n2. With structured content:")
    structured_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": cached_content,
                        "cache_control": {
                            "type": "ephemeral"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Tell me about the text."
                    }
                ]
            }
        ]
    }

    start_time = time.time()
    try:
        response = client.invoke_model(
            body=json.dumps(structured_body),
            modelId=CLAUDE_3_7_SONNET,
            accept="application/json",
            contentType="application/json"
        )
        end_time = time.time()

        response_body = json.loads(response['body'].read())
        result2_time = end_time - start_time
        result2_usage = response_body.get('usage', {})

        print(f"Time: {result2_time:.2f}s")
        print(f"Usage: {result2_usage}")

        # Wait a moment
        print("Waiting a moment...")
        time.sleep(2)

        # Third request with same structured content
        print("\n3. Second attempt with structured content:")

        start_time = time.time()
        response = client.invoke_model(
            body=json.dumps(structured_body),
            modelId=CLAUDE_3_7_SONNET,
            accept="application/json",
            contentType="application/json"
        )
        end_time = time.time()

        response_body = json.loads(response['body'].read())
        result3_time = end_time - start_time
        result3_usage = response_body.get('usage', {})

        print(f"Time: {result3_time:.2f}s")
        print(f"Usage: {result3_usage}")

        # Check if any caching metrics are present
        if 'cacheReadInputTokens' in result3_usage and result3_usage['cacheReadInputTokens'] > 0:
            print("✅ Cache hit detected in basic test")
        else:
            print("❌ No cache hit detected in basic test")

    except Exception as e:
        print(f"Error during basic caching test: {str(e)}")
        print("This may indicate that prompt caching is not supported in your environment.")


def demonstrate_prompt_caching():
    """
    Demonstrate the effect of prompt caching on response time
    """
    print("Creating optimized client...")
    bedrock_client = create_optimized_client()

    # Test for API version support
    try:
        print("Testing API compatibility...")
        test_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 10,
            "messages": [
                {
                    "role": "user",
                    "content": "Hello"
                }
            ]
        }

        bedrock_client.invoke_model(
            body=json.dumps(test_body),
            modelId=CLAUDE_3_7_SONNET,
            accept="application/json",
            contentType="application/json"
        )
        print("✅ Basic API test successful")
    except Exception as e:
        print(f"❌ API test failed: {str(e)}")
        print("Please check your AWS credentials and region settings.")
        return

    # Use the sample document as context
    document = SAMPLE_DOCUMENT

    # Get document Q&A prompts
    qa_prompts = get_document_qa_prompts(document)

    # First query without caching
    print("\n1. First query (without cache)...")
    first_query = qa_prompts[0]
    print(f"Query: {first_query}")
    first_result = time_response(bedrock_client, document, first_query, use_caching=False)
    print(f"Response time: {first_result['time']:.2f} seconds")
    print(f"Response: {first_result['text'][:100]}...")

    # Second query with caching disabled
    print("\n2. Second query (still without cache)...")
    second_query = qa_prompts[1]
    print(f"Query: {second_query}")
    second_result = time_response(bedrock_client, document, second_query, use_caching=False)
    print(f"Response time: {second_result['time']:.2f} seconds")
    print(f"Response: {second_result['text'][:100]}...")

    try:
        # Third query with caching structure
        print("\n3. Third query (with caching structure - attempt)...")
        third_query = qa_prompts[2]
        print(f"Query: {third_query}")
        third_result = time_response(bedrock_client, document, third_query, use_caching=True)
        print(f"Response time: {third_result['time']:.2f} seconds")
        print(f"Response: {third_result['text'][:100]}...")
        print(f"Usage statistics: {third_result['usage']}")

        # Add a delay
        print("\nWaiting a moment...")
        time.sleep(2)

        # Fourth query with same caching structure
        print("\n4. Fourth query (same caching structure)...")
        fourth_query = qa_prompts[2]  # Using the same query as third for potential cache hit
        print(f"Query: {fourth_query}")
        fourth_result = time_response(bedrock_client, document, fourth_query, use_caching=True)
        print(f"Response time: {fourth_result['time']:.2f} seconds")
        print(f"Response: {fourth_result['text'][:100]}...")

        # Display cache statistics
        print("Usage statistics:")
        cache_stats = fourth_result['usage']
        if 'cacheReadInputTokens' in cache_stats and cache_stats['cacheReadInputTokens'] > 0:
            print(f"✅ Cache hit detected: {cache_stats['cacheReadInputTokens']} tokens read from cache")
        else:
            print("❌ No cache hit detected")

        # Try one more with a different query
        print("\n5. Fifth query (different query, same caching structure)...")
        fifth_query = qa_prompts[3] if len(qa_prompts) > 3 else "What are the fees associated with this account?"
        print(f"Query: {fifth_query}")
        fifth_result = time_response(bedrock_client, document, fifth_query, use_caching=True)
        print(f"Response time: {fifth_result['time']:.2f} seconds")
        print(f"Response: {fifth_result['text'][:100]}...")

        # Display cache statistics
        print("Usage statistics:")
        cache_stats = fifth_result['usage']
        if 'cacheReadInputTokens' in cache_stats and cache_stats['cacheReadInputTokens'] > 0:
            print(f"✅ Cache hit detected: {cache_stats['cacheReadInputTokens']} tokens read from cache")
        else:
            print("❌ No cache hit detected")

        # Summary of results
        print("\nSummary:")
        print(f"Without caching: {first_result['time']:.2f}s, {second_result['time']:.2f}s")
        print(f"With caching structure: {third_result['time']:.2f}s, {fourth_result['time']:.2f}s, {fifth_result['time']:.2f}s")

        # Calculate performance comparison
        avg_no_cache = (first_result['time'] + second_result['time']) / 2
        avg_with_caching = (third_result['time'] + fourth_result['time'] + fifth_result['time']) / 3

        if avg_with_caching < avg_no_cache:
            improvement = (1 - avg_with_caching / avg_no_cache) * 100
            print(f"Performance improvement with structured content: {improvement:.1f}%")
        else:
            print("No performance improvement observed with structured content.")

    except Exception as e:
        print(f"Error during caching tests: {str(e)}")
        print("This may indicate that prompt caching is not supported in your environment.")

    # Add basic caching test for verification
    test_basic_caching(bedrock_client)

    print("\nConclusions about prompt caching:")
    print("1. The performance impact of structured content vs. regular content can vary.")
    print("2. If caching is not working, it may not be available in your AWS Bedrock configuration.")
    print("3. Check the AWS Bedrock documentation for the latest information on prompt caching.")
    print("   You may need to update the boto3 version or use a different Claude model version.")
    print("4. Boto3 version: 1.38.11 might not fully support all Claude 3.7 Sonnet features.")
    print("   Consider updating to the latest boto3 version if possible.")


if __name__ == "__main__":
    demonstrate_prompt_caching()