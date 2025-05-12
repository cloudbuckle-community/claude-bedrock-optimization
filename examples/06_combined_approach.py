"""
Demonstrates a combined approach with all optimizations.
"""

import sys
import os
import json
import time

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.client import create_optimized_client, create_claude_llm
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_react_agent
from config import BUDGET_BALANCED, CLAUDE_3_7_SONNET
from utils.prompts import SAMPLE_DOCUMENT


def simple_calculator(expression):
    """Simple calculator tool"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {str(e)}"


def query_document(query):
    """Tool to query a document"""
    # In a real implementation, this would do RAG or similar
    return f"Querying document for: {query}"


def demonstrate_combined_approach():
    """
    Demonstrate all optimizations combined
    """
    # 1. Create optimized client with proper timeout
    client = create_optimized_client()

    # 2. Create LLM with balanced reasoning budget
    llm = create_claude_llm(
        client=client,
        thinking_budget=BUDGET_BALANCED,
        enable_caching=True  # 3. Enable prompt caching
    )

    # 4. Create optimized agent with minimal instructions
    tools = [
        Tool(
            name="calculator",
            func=simple_calculator,
            description="Calculator for math operations. Input: mathematical expression."
        ),
        Tool(
            name="document_query",
            func=query_document,
            description="Query document for information. Input: specific question about the document."
        )
    ]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You're a financial advisor. Use tools when needed. Be concise."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=3,  # Limit iterations to prevent overthinking
        early_stopping_method="generate"
    )

    # 5. Implement document caching for RAG
    def query_with_document_context(query):
        """
        Query with document context using prompt caching
        """
        # Structure for prompt caching
        messages_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "I need help understanding this financial document:\n\n" + SAMPLE_DOCUMENT,
                            "cache_control": {
                                "type": "ephemeral"  # Cache this part
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

        response_body = json.loads(response['body'].read())
        response_text = response_body['content'][0]['text'] if 'content' in response_body else "No content"

        return {
            "time": end_time - start_time,
            "text": response_text,
            "usage": response_body.get('usage', {})
        }

    # Test combined approach with different query types

    # Test agent with calculation query
    print("1. Testing optimized agent with calculation query...")
    calculation_query = "If I invest $10,000 at 5% annual interest compounded monthly, how much will I have after 10 years?"
    start_time = time.time()
    agent_result = agent_executor.invoke({"input": calculation_query})
    agent_time = time.time() - start_time
    print(f"\nAgent response time: {agent_time:.2f} seconds")

    # Test document query with caching - first request (cache write)
    print("\n2. Testing document query with caching (first request)...")
    doc_query1 = "What are the interest rates for this account?"
    doc_result1 = query_with_document_context(doc_query1)
    print(f"Document query response time: {doc_result1['time']:.2f} seconds")
    print(f"Response: {doc_result1['text'][:100]}...")

    # Test document query with caching - second request (cache hit)
    print("\n3. Testing document query with caching (second request)...")
    doc_query2 = "What fees are associated with this account?"
    doc_result2 = query_with_document_context(doc_query2)
    print(f"Document query response time: {doc_result2['time']:.2f} seconds")
    print(f"Response: {doc_result2['text'][:100]}...")
    print(f"Cache statistics: {doc_result2['usage']}")

    print("\nCombined approach benefits:")
    print("1. Proper timeout configuration prevents connection errors")
    print("2. Balanced reasoning budget optimizes performance vs. quality tradeoff")
    print("3. Prompt caching accelerates document queries by caching static content")
    print("4. Optimized agent design prevents overthinking and excessive iterations")
    print("5. Early stopping prevents unnecessary reasoning steps")

    # Calculate document query improvement
    doc_improvement = (1 - doc_result2['time'] / doc_result1['time']) * 100
    print(f"\nDocument query latency reduction with caching: {doc_improvement:.1f}%")


if __name__ == "__main__":
    demonstrate_combined_approach()