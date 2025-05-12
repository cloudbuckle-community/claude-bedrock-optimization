"""
Performance measurement utilities for Claude 3.7 Sonnet.
"""

import time
import json
from langchain_aws import ChatBedrockConverse
import boto3


def time_llm_response(
        llm,
        prompt,
        num_runs=1,
        verbose=True
):
    """
    Time the response of an LLM for a given prompt

    Args:
        llm (ChatBedrockConverse): LangChain LLM
        prompt (str): Prompt to send
        num_runs (int): Number of runs to average over
        verbose (bool): Whether to print results

    Returns:
        dict: Timing results
    """

    times = []

    for i in range(num_runs):
        if verbose and num_runs > 1:
            print(f"Run {i + 1}/{num_runs}...")

        llm_response = None
        start_time = time.time()
        llm_response = llm.invoke(prompt)
        end_time = time.time()

        # print("Raw response type:", type(llm_response))
        # print("Raw response content:", llm_response)

        elapsed = end_time - start_time
        times.append(elapsed)

        if verbose:
            print(f"Response time: {elapsed:.2f} seconds")

    avg_time = sum(times) / len(times)

    result = {
        "prompt": prompt,
        "times": times,
        "average_time": avg_time,
        "min_time": min(times),
        "max_time": max(times),
        "response": llm_response
    }

    if verbose and num_runs > 1:
        print(f"\nSummary:")
        print(f"Average response time: {avg_time:.2f} seconds")
        print(f"Min: {min(times):.2f}s, Max: {max(times):.2f}s")

    return result


def time_direct_invoke(
        client,
        body,
        model_id,
        num_runs=1,
        verbose=True
):
    """
    Time direct invocation of Bedrock model

    Args:
        client (boto3.client): Bedrock runtime client
        body (dict): Request body
        model_id (str): Model ID
        num_runs (int): Number of runs to average over
        verbose (bool): Whether to print results

    Returns:
        dict: Timing results with usage statistics
    """
    times = []
    usages = []

    body_str = json.dumps(body)

    for i in range(num_runs):
        if verbose and num_runs > 1:
            print(f"Run {i + 1}/{num_runs}...")

        start_time = time.time()
        response = client.invoke_model(
            body=body_str,
            modelId=model_id,
            accept="application/json",
            contentType="application/json"
        )
        end_time = time.time()

        elapsed = end_time - start_time
        times.append(elapsed)

        # Extract usage information
        response_body = json.loads(response['body'].read())
        if 'usage' in response_body:
            usages.append(response_body['usage'])

        if verbose:
            print(f"Response time: {elapsed:.2f} seconds")
            if usages:
                if 'cacheReadInputTokens' in usages[-1]:
                    print(f"Cache read tokens: {usages[-1].get('cacheReadInputTokens', 0)}")
                if 'cacheWriteInputTokens' in usages[-1]:
                    print(f"Cache write tokens: {usages[-1].get('cacheWriteInputTokens', 0)}")

    avg_time = sum(times) / len(times)

    result = {
        "times": times,
        "average_time": avg_time,
        "min_time": min(times),
        "max_time": max(times),
        "usages": usages
    }

    if verbose and num_runs > 1:
        print(f"\nSummary:")
        print(f"Average response time: {avg_time:.2f} seconds")
        print(f"Min: {min(times):.2f}s, Max: {max(times):.2f}s")

    return result


def compare_llms(llms, prompts, display_results=True):
    """
    Compare multiple LLMs on multiple prompts

    Args:
        llms (dict): Dictionary of {name: llm}
        prompts (list): List of prompts to test
        display_results (bool): Whether to print results

    Returns:
        dict: Comparison results
    """
    results = {}

    for llm_name, llm in llms.items():
        results[llm_name] = {}

        if display_results:
            print(f"\n--- Testing {llm_name} ---")

        for i, prompt in enumerate(prompts):
            prompt_name = f"Prompt {i + 1}" if isinstance(prompt, str) else prompt.get("name", f"Prompt {i + 1}")
            prompt_text = prompt if isinstance(prompt, str) else prompt.get("text", "")

            if display_results:
                print(f"\n{prompt_name}:")

            result = time_llm_response(
                llm=llm,
                prompt=prompt_text,
                verbose=display_results
            )

            results[llm_name][prompt_name] = result

    return results