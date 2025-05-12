# Claude 3.7 Sonnet Optimization on AWS Bedrock

This repository contains practical code examples demonstrating how to optimize performance and reduce response time when using Claude 3.7 Sonnet on AWS Bedrock with LangChain. Through these optimizations, we've achieved a 10-30x improvement in response times (from 95+ seconds to 3-10 seconds).

## Key Optimizations Demonstrated

1. **Proper API Integration** - Using the correct LangChain integration (`ChatBedrockConverse`) for Claude 3.7 models
2. **Timeout Configuration** - Configuring appropriate AWS SDK timeouts to prevent premature connection termination
3. **Reasoning Budget Control** - Optimizing Claude's extended thinking capabilities with appropriate token budgets
4. **Prompt Caching** - Implementing caching for static context to dramatically reduce response time
5. **Agent Optimization** - Creating efficient LangChain agents with proper tool configuration (68.6% performance improvement)

## Getting Started

### Prerequisites
- Python 3.9+
- AWS account with access to Claude 3.7 Sonnet on Bedrock
- AWS credentials configured locally

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/cloudbuckle-community/claude-bedrock-optimization.git
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

Claude 3.7 Sonnet requires the newer Messages API format rather than the legacy completion API. This repository demonstrates the proper LangChain integration using the `ChatBedrockConverse` class, which avoids common errors:

```python
from langchain_aws import ChatBedrockConverse

llm = ChatBedrockConverse(
    model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    client=bedrock_client,
    temperature=0,
    max_tokens=1024
)
```

### 2. Timeout Configuration

AWS SDK clients have a default timeout of just 60 seconds, which causes premature termination of Claude 3.7 requests, especially when using extended thinking mode. Our examples show how to configure appropriate timeouts:

```python
from botocore.config import Config

boto_config = Config(
    read_timeout=3600,  # 60 minutes in seconds
    connect_timeout=10,
    retries={'max_attempts': 3}
)

bedrock_client = boto3.client(
    'bedrock-runtime',
    region_name='us-east-1',
    config=boto_config
)
```

### 3. Reasoning Budget Control

Claude 3.7 Sonnet's unique "extended thinking" capability can be optimized by controlling the token budget. We demonstrate how different budgets affect performance and quality:

```python
llm = ChatBedrockConverse(
    model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    client=bedrock_client,
    temperature=0,
    max_tokens=1024,
    additional_model_request_fields={
        "thinking": {
            "type": "enabled",
            "budget_tokens": 2000  # Balanced default
        }
    }
)
```

- **Standard mode**: Fastest responses, good for simple queries
- **Fast thinking (1,000 tokens)**: Quick responses with basic reasoning
- **Balanced thinking (2,500 tokens)**: Good balance of speed and quality
- **Deep thinking (5,000 tokens)**: Thorough analysis for complex questions

### 4. Prompt Caching

AWS Bedrock's prompt caching feature can reduce response latency by up to 85% for repeated content. Our implementation shows how to structure prompts to maximize caching benefits:

```python
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": document_content,
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
```

**Important note**: Our testing revealed that caching works effectively for literary content but not for financial/markdown-formatted content. Content type matters for caching effectiveness.

### 5. Agent Optimization

LangChain agents can be optimized for better performance by:
- Using minimal instructions (reduced token count)
- Limiting iterations (prevents overthinking)
- Implementing early stopping (avoids unnecessary steps)
- Providing focused tool descriptions
- Designing robust tools that handle various input formats

```python
# Create optimized agent with minimal instructions
agent_executor = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    max_iterations=3,
    early_stopping_method="generate",
    handle_parsing_errors=True,
    agent_kwargs={
        "prefix": """
        You're a financial advisor. Use tools when needed. Be concise.
        The calculator tool accepts standard math syntax and handles ^ notation.
        """
    }
)
```

Our testing showed a dramatic 68.6% improvement in response time with these optimizations.

## Performance Results

Our optimization journey delivered dramatic improvements:

| Implementation Stage | Response Time |
|----------------------|---------------|
| Initial implementation | 95-120 seconds |
| After timeout config | 75-95 seconds |
| With proper integration | 40-60 seconds |
| With reasoning budget | 20-30 seconds |
| With prompt caching | 5-15 seconds (for cached content) |
| With optimized agents | 3-10 seconds (overall average) |

## Benchmarking

You can run comprehensive benchmarks to compare performance across different configurations:

```bash
python benchmark/benchmark.py
```

This will generate performance comparison charts in the `benchmark/results/` directory.

## Related Blog Post

For a detailed explanation of these optimization techniques, check out the accompanying blog post: [Optimizing Claude 3.7 Sonnet Response Time with LangChain and AWS Bedrock](https://blog.cloudbuckle.com/80b09c71fee0)

## Key Lessons Learned

1. **Integration matters**: Using the right LangChain class makes a huge difference
2. **AWS config is critical**: Simple timeout settings prevent many failures
3. **Reasoning budget is powerful**: Tuning the thinking tokens has dramatic effects
4. **Prompt structure affects performance**: How you structure content impacts caching effectiveness
5. **Simpler agents work better**: Less complexity = faster responses (68.6% improvement)
6. **Tool design is crucial**: Building robust tools reduces errors and correction cycles

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
