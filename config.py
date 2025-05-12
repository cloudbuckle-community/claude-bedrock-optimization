"""
Configuration settings for AWS Bedrock and Claude 3.7 Sonnet.
"""

# AWS configuration
AWS_REGION = "us-east-1"  # Change to your preferred region

# Claude model IDs
CLAUDE_3_7_SONNET = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
CLAUDE_3_5_SONNET = "us.anthropic.claude-3-5-sonnet-20240620-v1:0"

# Default timeout settings
DEFAULT_TIMEOUT = 3600  # 60 minutes in seconds
CONNECTION_TIMEOUT = 10  # 10 seconds

# Reasoning budget presets
BUDGET_MINIMAL = 1024    # Fast responses
BUDGET_BALANCED = 2500   # Good balance of speed and quality
BUDGET_DEEP = 5000       # In-depth analysis

# Prompt caching settings
CACHE_TTL = 300  # 5 minutes in seconds