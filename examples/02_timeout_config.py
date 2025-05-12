"""
Demonstrates the importance of timeout configuration for Claude 3.7 Sonnet.
"""

import sys
import os
import boto3
from botocore.config import Config
import json
import time

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.client import create_optimized_client, create_claude_llm
from utils.prompts import FINANCIAL_QUESTIONS
from config import CLAUDE_3_7_SONNET


def create_default_client():
    """
    Create a boto3 client with default timeout settings
    """
    boto_config = Config(
        read_timeout=60,
        connect_timeout=10,
        retries={'max_attempts': 0}
    )
    return boto3.client(
        'bedrock-runtime',
        region_name="us-east-1" ,
        config=boto_config
    )


def create_custom_timeout_client(timeout_seconds):
    """
    Create a boto3 client with custom timeout
    """
    boto_config = Config(
        read_timeout=timeout_seconds,
        connect_timeout=10,
        retries={'max_attempts': 0}
    )

    return boto3.client(
        'bedrock-runtime',
        region_name="us-east-1",
        config=boto_config
    )


def invoke_with_timeout(client, prompt, thinking_budget=3000):
    """
    Invoke Claude with specific prompt and measure time
    Using extended thinking to make processing take longer
    """
    # Ensure thinking budget is at least 1024
    thinking_budget = max(1024, thinking_budget)

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": thinking_budget + 1024,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "thinking": {
            "type": "enabled",
            "budget_tokens": thinking_budget
        }
    }

    start_time = time.time()

    try:
        response = client.invoke_model(
            body=json.dumps(body),
            modelId=CLAUDE_3_7_SONNET,
            accept="application/json",
            contentType="application/json"
        )

        response_body = json.loads(response['body'].read())
        end_time = time.time()

        # Properly access the response text - this was likely causing the 'text' error
        response_text = ""
        if 'content' in response_body and len(response_body['content']) > 0:
            if 'text' in response_body['content'][0]:
                response_text = response_body['content'][0]['text']
            else:
                response_text = "Response content found but no text field"
        else:
            response_text = "No content in response"

        return {
            "success": True,
            "time": end_time - start_time,
            "response": response_text
        }
    except Exception as e:
        end_time = time.time()
        return {
            "success": False,
            "time": end_time - start_time,
            "error": str(e)
        }


def demonstrate_timeout_config():
    """
    Demonstrate the impact of timeout configuration
    """
    print("Testing timeout configuration on a complex financial query...")

    # Use a complex financial question to ensure longer processing time
    prompt = FINANCIAL_QUESTIONS[3]

    # Test with default timeout (60 seconds)
    print("\n1. Using default client (60-second timeout):")
    default_client = create_default_client()
    default_result = invoke_with_timeout(default_client, prompt, thinking_budget=8000)

    if default_result["success"]:
        print(f"✅ Success: Response received in {default_result['time']:.2f} seconds")
    else:
        print(f"❌ Error: {default_result['error']}")
        print(f"Request ran for {default_result['time']:.2f} seconds before failing")

    # Test with extended timeout (2 minutes)
    print("\n2. Using extended timeout client (120-second timeout):")
    extended_client = create_custom_timeout_client(120)
    extended_result = invoke_with_timeout(extended_client, prompt, thinking_budget=8000)

    if extended_result["success"]:
        print(f"✅ Success: Response received in {extended_result['time']:.2f} seconds")
    else:
        print(f"❌ Error: {extended_result['error']}")
        print(f"Request ran for {extended_result['time']:.2f} seconds before failing")

    # Test with optimized client (3-minute timeout)
    print("\n3. Using optimized client (180-second timeout):")
    optimized_client = create_optimized_client()
    optimized_result = invoke_with_timeout(optimized_client, prompt, thinking_budget=8000)

    if optimized_result["success"]:
        print(f"✅ Success: Response received in {optimized_result['time']:.2f} seconds")
    else:
        print(f"❌ Error: {optimized_result['error']}")
        print(f"Request ran for {optimized_result['time']:.2f} seconds before failing")

    print("\nKey takeaway:")
    print("AWS SDK clients have a default timeout of just 60 seconds, which can")
    print("cause premature termination of Claude 3.7 requests, especially when")
    print("using extended thinking mode or processing complex queries.")
    print("Always configure appropriate timeouts for your use case.")


if __name__ == "__main__":
    demonstrate_timeout_config()