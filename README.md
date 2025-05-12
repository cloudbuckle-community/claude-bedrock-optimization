# Claude 3.7 Sonnet Optimization on AWS Bedrock

This repository contains practical code examples demonstrating how to optimize performance and reduce response time when using Claude 3.7 Sonnet on AWS Bedrock with LangChain.

## Key Optimizations Demonstrated

1. **Proper API Integration** - Using the correct LangChain integration (ChatBedrockConverse) for Claude 3.7 models
2. **Timeout Configuration** - Configuring appropriate AWS SDK timeouts to prevent premature connection termination
3. **Reasoning Budget Control** - Optimizing Claude's extended thinking capabilities with appropriate token budgets
4. **Prompt Caching** - Implementing caching for static context to dramatically reduce response time
5. **Agent Optimization** - Creating efficient LangChain agents with proper tool configuration

## Getting Started

### Prerequisites
- Python 3.9+
- AWS account with access to Claude 3.7 Sonnet on Bedrock
- AWS credentials configured locally

### Installation

1. Clone this repository:
   ```bash
   git clone git@github.com-cbkl-comm:cloudbuckle-community/claude-bedrock-optimization.git
   cd claude-bedrock-optimization
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure your AWS credentials:
   ```bash
   aws configure
   ```

### Running Examples

Each example in the `examples/` directory demonstrates a specific optimization:

```bash
python examples/01_basic_setup.py        # Proper API integration
python examples/02_timeout_config.py     # Timeout configuration
python examples/03_reasoning_budget.py   # Reasoning budget control
python examples/04_prompt_caching.py     # Prompt caching implementation
python examples/05_agent_optimization.py # Agent optimization
python examples/06_combined_approach.py  # All optimizations together
```

## Optimization Details

### 1. Proper API Integration

Claude 3.7 Sonnet requires the newer Messages API format rather than the legacy completion API. This repository demonstrates the proper LangChain integration using the `ChatBedrockConverse` class, which avoids the common error:

```
ValidationException: "claude-3-7-sonnet-20250219" is not supported on this API. Please use the Messages API instead.
```

### 2. Timeout Configuration

AWS SDK clients have a default timeout of just 60 seconds, which causes premature termination of Claude 3.7 requests, especially when using extended thinking mode. Our examples show how to configure appropriate timeouts:

```python
boto_config = Config(
    read_timeout=3600,  # 60 minutes in seconds
    connect_timeout=10,
    retries={'max_attempts': 3}
)
```

### 3. Reasoning Budget Control

Claude 3.7 Sonnet's unique "extended thinking" capability can be optimized by controlling the token budget. We demonstrate how different budgets affect performance and quality:

- **Standard mode**: Fastest responses, good for simple queries
- **Fast thinking (1,000 tokens)**: Quick responses with basic reasoning
- **Balanced thinking (2,500 tokens)**: Good balance of speed and quality
- **Deep thinking (5,000 tokens)**: Thorough analysis for complex questions

### 4. Prompt Caching

AWS Bedrock's prompt caching feature can reduce response latency by up to 85% for repeated content. Our implementation shows how to structure prompts to maximize caching benefits:

```python
{
    "type": "text",
    "text": "Document content that doesn't change...",
    "cache_control": {
        "type": "ephemeral"
    }
}
```

### 5. Agent Optimization

LangChain agents can be optimized for better performance by:
- Using minimal instructions (reduced token count)
- Limiting iterations (prevents overthinking)
- Implementing early stopping (avoids unnecessary steps)
- Providing focused tool descriptions

## Benchmarking

You can run comprehensive benchmarks to compare performance across different configurations:

```bash
python benchmark/benchmark.py
```

This will generate performance comparison charts in the `benchmark/results/` directory.

## Related Blog Post

For a detailed explanation of these optimization techniques, check out the accompanying blog post: [Optimizing Claude 3.7 Sonnet Response Time with LangChain and AWS Bedrock](https://medium.com/@cloudbuckle/optimizing-claude-3-7-sonnet-response-time-with-langchain-and-aws-bedrock)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Anthropic for developing Claude 3.7 Sonnet
- AWS for providing the Bedrock platform
- LangChain for their excellent framework

---

Built with ❤️ by [CloudBuckle](https://github.com/cloudbuckle-community)