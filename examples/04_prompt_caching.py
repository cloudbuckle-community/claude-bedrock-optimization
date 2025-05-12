"""
Comparative prompt caching test between non-sensitive content (book/literature) and sensitive (financial) content.
"""

import sys
import os
import json
import time
import requests
from bs4 import BeautifulSoup
import boto3

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.prompts import SAMPLE_DOCUMENT, get_document_qa_prompts
from config import CLAUDE_3_7_SONNET


def fetch_book_content(url="https://www.gutenberg.org/cache/epub/1342/pg1342.txt"):
    """Fetch and clean book content like in your working example"""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()

    # Get text
    text = soup.get_text()

    # Clean up text
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)

    # To ensure we're not exceeding token limits, we'll truncate
    return text[:20000]  # Truncate to avoid token limits


def test_content_caching(content, content_name, client, query="What is this text about?"):
    """Test caching for a specific content"""
    print(f"\n=== Testing Caching for {content_name} ===")
    print(f"Content length: {len(content)} characters")

    # Create cached payload
    cached_payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": content,
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

    # Convert to JSON once to ensure consistency
    cached_json = json.dumps(cached_payload)

    # First call to set up the cache
    print(f"1. First {content_name} query (cache setup)...")
    start_time = time.time()
    response = client.invoke_model(
        body=cached_json,
        modelId=CLAUDE_3_7_SONNET,
        accept="application/json",
        contentType="application/json"
    )
    end_time = time.time()

    response_body = json.loads(response['body'].read())
    first_time = end_time - start_time
    print(f"Response time: {first_time:.2f} seconds")
    first_usage = response_body.get('usage', {})
    print(f"Usage statistics: {first_usage}")

    # Wait for cache to establish
    print(f"Waiting 10 seconds to ensure cache is established...")
    time.sleep(10)

    # Second call that should use the cache
    print(f"2. Second {content_name} query (should use cache)...")
    start_time = time.time()
    response = client.invoke_model(
        body=cached_json,  # Same JSON string
        modelId=CLAUDE_3_7_SONNET,
        accept="application/json",
        contentType="application/json"
    )
    end_time = time.time()

    response_body = json.loads(response['body'].read())
    second_time = end_time - start_time
    second_usage = response_body.get('usage', {})
    print(f"Response time: {second_time:.2f} seconds")
    print(f"Usage statistics: {second_usage}")

    # Check for cache hit
    cache_hit = False
    if 'cache_read_input_tokens' in second_usage and second_usage['cache_read_input_tokens'] > 0:
        cache_hit = True
        print(f"✅ Cache hit detected: {second_usage['cache_read_input_tokens']} tokens read from cache")
    else:
        print(f"❌ No cache hit detected for {content_name}")

    return {
        "content_name": content_name,
        "first_time": first_time,
        "second_time": second_time,
        "cache_hit": cache_hit,
        "first_usage": first_usage,
        "second_usage": second_usage
    }


def run_comparative_test():
    """Run a comparative test between book and financial content"""
    print("Creating client...")
    client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

    # Get the two types of content
    print("Fetching book content...")
    book_content = fetch_book_content()
    financial_content = SAMPLE_DOCUMENT

    # Test book content first
    book_results = test_content_caching(
        book_content,
        "Book Content",
        client,
        "What is the main theme of this book?"
    )

    # Test financial content second
    financial_results = test_content_caching(
        financial_content,
        "Financial Content",
        client,
        "What are the interest rates mentioned in this text?"
    )

    # Print summary comparison
    print("\n=== COMPARATIVE RESULTS ===")
    print(f"Book Content:")
    print(f"  First call: {book_results['first_time']:.2f}s")
    print(f"  Second call: {book_results['second_time']:.2f}s")
    print(f"  Cache hit: {'Yes' if book_results['cache_hit'] else 'No'}")

    print(f"\nFinancial Content:")
    print(f"  First call: {financial_results['first_time']:.2f}s")
    print(f"  Second call: {financial_results['second_time']:.2f}s")
    print(f"  Cache hit: {'Yes' if financial_results['cache_hit'] else 'No'}")

    # Analysis
    print("\n=== ANALYSIS ===")
    if book_results['cache_hit'] and not financial_results['cache_hit']:
        print("CONFIRMED: Book content is cached but financial content is not.")
        print("This suggests Claude is treating the content types differently for caching.")
    elif financial_results['cache_hit'] and not book_results['cache_hit']:
        print("UNEXPECTED: Financial content is cached but book content is not.")
    elif book_results['cache_hit'] and financial_results['cache_hit']:
        print("BOTH are cached. Content type doesn't affect caching.")
    else:
        print("NEITHER is cached. The caching feature might not be available in your environment.")

    # Performance analysis
    book_speedup = book_results['first_time'] - book_results['second_time']
    financial_speedup = financial_results['first_time'] - financial_results['second_time']

    print(f"\nPerformance speedup:")
    print(f"  Book content: {book_speedup:.2f}s faster on second call")
    print(f"  Financial content: {financial_speedup:.2f}s faster on second call")

    print("\nRecommendations:")
    if book_results['cache_hit'] and not financial_results['cache_hit']:
        print("1. Use caching primarily for non-sensitive, non-financial content")
        print("2. For financial documents, consider that caching may not be available")
        print("3. You may want to contact AWS support to confirm if this is expected behavior")
    elif not book_results['cache_hit'] and not financial_results['cache_hit']:
        print("1. Check if your AWS Bedrock setup supports caching features")
        print("2. Verify boto3 version is up to date")
        print("3. Ensure 'cache_control' is properly formatted in your requests")
        print("4. Consider using 'x-amzn-bedrock-trace' header to help diagnose API issues")


if __name__ == "__main__":
    run_comparative_test()
