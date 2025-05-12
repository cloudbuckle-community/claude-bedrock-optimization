"""
Utilities for creating optimized AWS Bedrock clients.
"""

import boto3
from botocore.config import Config
from langchain_aws import ChatBedrockConverse
import sys
import os

# Add the project root to the path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    AWS_REGION,
    DEFAULT_TIMEOUT,
    CONNECTION_TIMEOUT,
    CLAUDE_3_5_SONNET,
    CLAUDE_3_7_SONNET,
    BUDGET_BALANCED
)


def create_optimized_client(region_name=AWS_REGION, timeout_seconds=DEFAULT_TIMEOUT):
    """
    Create an optimized boto3 client for Bedrock with extended timeout

    Args:
        region_name (str): AWS region
        timeout_seconds (int): Read timeout in seconds

    Returns:
        boto3.client: Configured Bedrock runtime client
    """
    boto_config = Config(
        read_timeout=timeout_seconds,
        connect_timeout=CONNECTION_TIMEOUT,
        retries={'max_attempts': 1}
    )

    client = boto3.client(
        'bedrock-runtime',
        region_name=region_name,
        config=boto_config
    )

    return client


def create_claude_llm(
        client=None,
        region_name=AWS_REGION,
        model_id=CLAUDE_3_7_SONNET,
        thinking_budget=None,
        temperature=0,
        max_tokens=1024,
        enable_caching=False
):
    """
    Create a properly configured Claude 3.7 LLM with optimized settings

    Args:
        client (boto3.client, optional): Pre-configured boto3 client
        region_name (str): AWS region if client not provided
        model_id (str): Claude model ID
        thinking_budget (int, optional): Token budget for reasoning
        temperature (float): Temperature for generation
        max_tokens (int): Maximum tokens in response
        enable_caching (bool): Whether to enable prompt caching

    Returns:
        ChatBedrockConverse: Configured LLM
    """
    if client is None:
        client = create_optimized_client(region_name)

    # Validate and adjust max_tokens if needed
    if thinking_budget is not None:
        if max_tokens <= thinking_budget:
            max_tokens = thinking_budget + 1024
            print(f"Warning: Adjusted max_tokens to {max_tokens} to exceed thinking budget")

        # Set temperature to 1 when thinking is enabled
        # This is required by Claude's API
        temperature = 1
        print(f"Note: Setting temperature to 1 as required with thinking mode")

    # Ensure max_tokens is greater than thinking_budget if thinking is enabled
    if thinking_budget is not None and max_tokens <= thinking_budget:
        # Add a buffer to ensure it's larger
        max_tokens = thinking_budget + 1024

    # Build additional model request fields
    additional_fields = {}

    # Add thinking configuration if specified
    if thinking_budget is not None:
        additional_fields["thinking"] = {
            "type": "enabled",
            "budget_tokens": thinking_budget
        }

    # Add prompt caching if enabled
    if enable_caching:
        additional_fields["prompt_caching"] = {
            "enabled": True
        }

    # Create the LLM
    llm = ChatBedrockConverse(
        model=model_id,
        client=client,
        temperature=temperature,
        max_tokens=max_tokens,
        additional_model_request_fields=additional_fields
    )

    return llm


def get_standard_llm(client=None):
    """Get Claude 3.7 in standard mode (no reasoning)"""
    return create_claude_llm(client=client)


def get_fast_thinking_llm(client=None):
    """Get Claude 3.7 with minimal reasoning budget"""
    from config import BUDGET_MINIMAL
    return create_claude_llm(
        client=client,
        thinking_budget=BUDGET_MINIMAL,
        max_tokens=BUDGET_MINIMAL + 1024  # Explicitly set max_tokens
    )


def get_balanced_thinking_llm(client=None):
    """Get Claude 3.7 with balanced reasoning budget"""
    from config import BUDGET_BALANCED
    return create_claude_llm(
        client=client,
        thinking_budget=BUDGET_BALANCED,
        max_tokens=BUDGET_BALANCED + 1024
    )


def get_deep_thinking_llm(client=None):
    """Get Claude 3.7 with deep reasoning budget"""
    from config import BUDGET_DEEP
    return create_claude_llm(
        client=client,
        thinking_budget=BUDGET_DEEP,
        max_tokens=BUDGET_DEEP + 1024
    )


def get_cached_llm(client=None):
    """Get Claude 3.7 with prompt caching enabled"""
    return create_claude_llm(client=client, enable_caching=True)


def get_optimal_llm(client=None):
    """Get Claude 3.7 with balanced reasoning and caching"""
    from config import BUDGET_BALANCED
    return create_claude_llm(
        client=client,
        thinking_budget=BUDGET_BALANCED,
        enable_caching=True
    )