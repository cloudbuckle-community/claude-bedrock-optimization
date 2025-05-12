"""
Sample prompts for testing and benchmarking.
"""

# Simple to complex financial questions for testing
FINANCIAL_QUESTIONS = [
    # Simple factual query
    "What is the difference between a savings account and a checking account?",

    # Moderate complexity
    "If I have $10,000 to invest for retirement and I'm 35 years old, what are my options and their pros and cons?",

    # Complex question requiring reasoning
    """
    I'm considering a 30-year mortgage at 5.5% APR for $350,000. 
    My property tax is estimated at $5,000 per year, and insurance at $1,200 annually.
    I can make a down payment of 15% or 20%.
    What would my monthly payment be in each scenario, and how much would I save over the life of the loan with the 20% down payment?
    Additionally, analyze how each investment would perform in high-inflation (7%+)
    and low-inflation (1-2%) scenarios. Calculate the risk-adjusted return for each
    option using a Sharpe ratio analysis. Finally, recommend the optimal portfolio
    allocation across these investments for someone in their early 40s with moderate
    risk tolerance.
    Finally create an amortization chart.
    """,

    # Very complex multi-part question
    """
    I'm comparing three investment options:
    1. A rental property costing $300,000 with expected rental income of $2,000/month and expenses of $600/month
    2. An S&P 500 index fund with historical average returns of 10% annually
    3. A small business investment requiring $300,000 with projected annual profit of $45,000

    Assuming a 30-year timeline, 3% inflation, 25% tax rate on profits, and that the property appreciates at 3% annually, 
    which investment would likely yield the highest after-tax return? Show your calculations.
    """
]

# System prompts of varying complexity
SYSTEM_PROMPTS = {
    "simple": "You are a helpful financial assistant.",

    "moderate": """
    You are a financial advisor who helps users understand financial products and make investment decisions.
    Provide clear, accurate information and always consider the user's best interests.
    """,

    "complex": """
    You are a senior financial advisor with the following responsibilities:

    1. Provide accurate, up-to-date information about financial products and services
    2. Help users understand complex financial concepts in simple terms
    3. Analyze user's financial situations and provide personalized advice
    4. Compare different financial options objectively with pros and cons
    5. Calculate returns, payments, and other financial metrics when requested
    6. Maintain compliance with financial regulations in your answers
    7. Clearly state when information might be dependent on market conditions
    8. Avoid making specific investment recommendations but provide general guidance

    When analyzing mortgage options, use standard amortization formulas and consider:
    - Principal and interest payments
    - Property taxes and insurance (if in escrow)
    - PMI for down payments under 20%
    - Total interest over the life of the loan

    For investment analyses, consider:
    - Initial investment amount
    - Expected returns (historical and projected)
    - Time horizon
    - Tax implications
    - Inflation effects
    - Risk factors

    Always present calculations clearly and explain your methodology.
    """
}

# Sample document for testing document Q&A
SAMPLE_DOCUMENT = """
# Premium Savings Account - Terms and Conditions

## Account Overview
The Premium Savings Account is a high-yield savings account designed for customers who maintain a minimum balance of $5,000. This account offers tiered interest rates based on your balance and provides monthly interest compounding.

## Interest Rates
Current rates as of January 1, 2025:
- $5,000 - $24,999: 2.50% APY
- $25,000 - $99,999: 3.25% APY
- $100,000+: 3.75% APY

Interest is calculated daily and compounded monthly. Rates are variable and may change at any time based on market conditions.

## Fees and Charges
- Monthly maintenance fee: $15 (waived if minimum daily balance of $5,000 is maintained)
- Excessive withdrawal fee: $10 per withdrawal after 6 withdrawals per month
- Account closure fee: $25 if closed within 90 days of opening
- Paper statement fee: $3 per month (waived with e-statements)

## Eligibility Requirements
- Must be 18 years or older
- Valid Social Security Number or Individual Taxpayer Identification Number
- US citizen or permanent resident
- Minimum opening deposit of $5,000

## Access and Features
- 24/7 online banking access
- Mobile banking app with mobile check deposit
- ATM card available upon request
- Automatic transfer capabilities
- Direct deposit available

## Early Withdrawal Penalties
While this is not a time-deposit account like a CD, excessive withdrawals beyond the allowed 6 per month will incur a $10 fee per transaction as noted in the Fees section.

## Account Protection
Funds in this account are FDIC insured up to $250,000 per depositor, per ownership category.

## Contact Information
For questions or support, please contact our customer service:
- Phone: 1-800-555-1234
- Email: support@bankexample.com
- Online: www.bankexample.com/contact

## Amendments
The bank reserves the right to change these terms and conditions at any time. Customers will be notified at least 30 days in advance of any changes.
"""


def get_document_qa_prompts(document=SAMPLE_DOCUMENT):
    """Generate document Q&A prompts using the sample document"""
    return [
        f"What are the interest rates for the Premium Savings Account?",
        f"What is the minimum balance requirement and what happens if I don't meet it?",
        f"How many withdrawals can I make per month without paying a fee?",
        f"Is there an early withdrawal penalty for this account?",
    ]