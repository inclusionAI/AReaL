"""
Yahoo finance MCP server.
Adapted from https://github.com/shzhiqi/yahoo-finance-mcp/blob/main/server.py.
"""
# pylint: disable=broad-exception-caught,too-many-statements,line-too-long,too-many-return-statements,invalid-name
import json
from enum import Enum

import click
import pandas as pd
import yfinance as yf
from mcp.server.fastmcp import FastMCP
from mcpuniverse.common.logger import get_logger


# Define an enum for the type of financial statement
class FinancialType(str, Enum):
    """Financial type."""
    income_stmt = "income_stmt"
    quarterly_income_stmt = "quarterly_income_stmt"
    balance_sheet = "balance_sheet"
    quarterly_balance_sheet = "quarterly_balance_sheet"
    cashflow = "cashflow"
    quarterly_cashflow = "quarterly_cashflow"


class HolderType(str, Enum):
    """Holder type."""
    major_holders = "major_holders"
    institutional_holders = "institutional_holders"
    mutualfund_holders = "mutualfund_holders"
    insider_transactions = "insider_transactions"
    insider_purchases = "insider_purchases"
    insider_roster_holders = "insider_roster_holders"


class RecommendationType(str, Enum):
    """Recommendation type."""
    recommendations = "recommendations"
    upgrades_downgrades = "upgrades_downgrades"


def build_server(port: int) -> FastMCP:
    """
    Initializes the MCP server.

    :param port: Port for SSE.
    :return: The MCP server.
    """

    yfinance_server = FastMCP(
        "yfinance",
        port=port,
        instructions="""
    # Yahoo Finance MCP Server
    
    This server is used to get information about a given ticker symbol from yahoo finance.
    
    Available tools:
    - get_historical_stock_prices: Get historical stock prices for a given ticker symbol from yahoo finance. Include the following information: Date, Open, High, Low, Close, Volume, Adj Close.
    - get_stock_info: Get stock information for a given ticker symbol from yahoo finance. Include the following information: Stock Price & Trading Info, Company Information, Financial Metrics, Earnings & Revenue, Margins & Returns, Dividends, Balance Sheet, Ownership, Analyst Coverage, Risk Metrics, Other.
    - get_yahoo_finance_news: Get news for a given ticker symbol from yahoo finance.
    - get_stock_actions: Get stock dividends and stock splits for a given ticker symbol from yahoo finance.
    - get_financial_statement: Get financial statement for a given ticker symbol from yahoo finance. You can choose from the following financial statement types: income_stmt, quarterly_income_stmt, balance_sheet, quarterly_balance_sheet, cashflow, quarterly_cashflow.
    - get_holder_info: Get holder information for a given ticker symbol from yahoo finance. You can choose from the following holder types: major_holders, institutional_holders, mutualfund_holders, insider_transactions, insider_purchases, insider_roster_holders.
    - get_option_expiration_dates: Fetch the available options expiration dates for a given ticker symbol.
    - get_option_chain: Fetch the option chain for a given ticker symbol, expiration date, and option type.
    - get_recommendations: Get recommendations or upgrades/downgrades for a given ticker symbol from yahoo finance. You can also specify the number of months back to get upgrades/downgrades for, default is 12.
    """,
    )

    @yfinance_server.tool(
        name="get_historical_stock_prices",
        description="""Get historical stock prices for a given ticker symbol from yahoo finance. Include the following information: Date, Open, High, Low, Close, Volume, Adj Close.
    Args:
        ticker: str
            The ticker symbol of the stock to get historical prices for, e.g. "AAPL"
        start_date: str
            format: yyyy-mm-dd
        end_date: str
            format: yyyy-mm-dd
        interval: str
            Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
            Intraday data cannot extend last 60 days
            Default is "1d"
    """,
    )
    async def get_historical_stock_prices(
            ticker: str, start_date: str, end_date: str, interval: str = "1d"
    ) -> str:
        """Get historical stock prices for a given ticker symbol

        Args:
            ticker: str
                The ticker symbol of the stock to get historical prices for, e.g. "AAPL"
            start_date: str
                format: yyyy-mm-dd
            end_date: str
                format: yyyy-mm-dd
            interval: str
                Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
                Intraday data cannot extend last 60 days
                Default is "1d"
        """
        company = yf.Ticker(ticker)
        try:
            if company.isin is None:
                print(f"Company ticker {ticker} not found.")
                return f"Company ticker {ticker} not found."
        except Exception as e:
            print(f"Error: getting historical stock prices for {ticker}: {e}")
            return f"Error: getting historical stock prices for {ticker}: {e}"

        # If the company is found, get the historical data
        hist_data = company.history(start=start_date, end=end_date, interval=interval)
        hist_data = hist_data.reset_index(names="Date")
        hist_data = hist_data.to_json(orient="records", date_format="iso")
        return hist_data

    @yfinance_server.tool(
        name="get_stock_info",
        description="""Get stock information for a given ticker symbol from yahoo finance. Include the following information:
    Stock Price & Trading Info, Company Information, Financial Metrics, Earnings & Revenue, Margins & Returns, Dividends, Balance Sheet, Ownership, Analyst Coverage, Risk Metrics, Other.
    
    Args:
        ticker: str
            The ticker symbol of the stock to get information for, e.g. "AAPL"
    """,
    )
    async def get_stock_info(ticker: str) -> str:
        """Get stock information for a given ticker symbol"""
        company = yf.Ticker(ticker)
        try:
            if company.isin is None:
                print(f"Company ticker {ticker} not found.")
                return f"Company ticker {ticker} not found."
        except Exception as e:
            print(f"Error: getting stock information for {ticker}: {e}")
            return f"Error: getting stock information for {ticker}: {e}"
        info = company.info
        return json.dumps(info)

    @yfinance_server.tool(
        name="get_yahoo_finance_news",
        description="""Get news for a given ticker symbol from yahoo finance.
    
    Args:
        ticker: str
            The ticker symbol of the stock to get news for, e.g. "AAPL"
    """,
    )
    async def get_yahoo_finance_news(ticker: str) -> str:
        """Get news for a given ticker symbol

        Args:
            ticker: str
                The ticker symbol of the stock to get news for, e.g. "AAPL"
        """
        company = yf.Ticker(ticker)
        try:
            if company.isin is None:
                print(f"Company ticker {ticker} not found.")
                return f"Company ticker {ticker} not found."
        except Exception as e:
            print(f"Error: getting news for {ticker}: {e}")
            return f"Error: getting news for {ticker}: {e}"

        # If the company is found, get the news
        try:
            news = company.news
        except Exception as e:
            print(f"Error: getting news for {ticker}: {e}")
            return f"Error: getting news for {ticker}: {e}"

        news_list = []
        for news in company.news:
            if news.get("content", {}).get("contentType", "") == "STORY":
                title = news.get("content", {}).get("title", "")
                summary = news.get("content", {}).get("summary", "")
                description = news.get("content", {}).get("description", "")
                url = news.get("content", {}).get("canonicalUrl", {}).get("url", "")
                news_list.append(
                    f"Title: {title}\nSummary: {summary}\nDescription: {description}\nURL: {url}"
                )
        if not news_list:
            print(f"No news found for company that searched with {ticker} ticker.")
            return f"No news found for company that searched with {ticker} ticker."
        return "\n\n".join(news_list)

    @yfinance_server.tool(
        name="get_stock_actions",
        description="""Get stock dividends and stock splits for a given ticker symbol from yahoo finance.
    
    Args:
        ticker: str
            The ticker symbol of the stock to get stock actions for, e.g. "AAPL"
    """,
    )
    async def get_stock_actions(ticker: str) -> str:
        """Get stock dividends and stock splits for a given ticker symbol"""
        try:
            company = yf.Ticker(ticker)
        except Exception as e:
            print(f"Error: getting stock actions for {ticker}: {e}")
            return f"Error: getting stock actions for {ticker}: {e}"
        actions_df = company.actions
        actions_df = actions_df.reset_index(names="Date")
        return actions_df.to_json(orient="records", date_format="iso")

    @yfinance_server.tool(
        name="get_financial_statement",
        description="""Get financial statement for a given ticker symbol from yahoo finance. You can choose from the following financial statement types: income_stmt, quarterly_income_stmt, balance_sheet, quarterly_balance_sheet, cashflow, quarterly_cashflow.
    
    Args:
        ticker: str
            The ticker symbol of the stock to get financial statement for, e.g. "AAPL"
        financial_type: str
            The type of financial statement to get. You can choose from the following financial statement types: income_stmt, quarterly_income_stmt, balance_sheet, quarterly_balance_sheet, cashflow, quarterly_cashflow.
    """,
    )
    async def get_financial_statement(ticker: str, financial_type: str) -> str:
        """Get financial statement for a given ticker symbol"""

        company = yf.Ticker(ticker)
        try:
            if company.isin is None:
                print(f"Company ticker {ticker} not found.")
                return f"Company ticker {ticker} not found."
        except Exception as e:
            print(f"Error: getting financial statement for {ticker}: {e}")
            return f"Error: getting financial statement for {ticker}: {e}"

        if financial_type == FinancialType.income_stmt:
            financial_statement = company.income_stmt
        elif financial_type == FinancialType.quarterly_income_stmt:
            financial_statement = company.quarterly_income_stmt
        elif financial_type == FinancialType.balance_sheet:
            financial_statement = company.balance_sheet
        elif financial_type == FinancialType.quarterly_balance_sheet:
            financial_statement = company.quarterly_balance_sheet
        elif financial_type == FinancialType.cashflow:
            financial_statement = company.cashflow
        elif financial_type == FinancialType.quarterly_cashflow:
            financial_statement = company.quarterly_cashflow
        else:
            return f"Error: invalid financial type {financial_type}. Please use one of the following: {FinancialType.income_stmt}, {FinancialType.quarterly_income_stmt}, {FinancialType.balance_sheet}, {FinancialType.quarterly_balance_sheet}, {FinancialType.cashflow}, {FinancialType.quarterly_cashflow}."

        # Create a list to store all the json objects
        result = []

        # Loop through each column (date)
        for column in financial_statement.columns:
            if isinstance(column, pd.Timestamp):
                date_str = column.strftime("%Y-%m-%d")  # Format as YYYY-MM-DD
            else:
                date_str = str(column)

            # Create a dictionary for each date
            date_obj = {"date": date_str}

            # Add each metric as a key-value pair
            for index, value in financial_statement[column].items():
                # Add the value, handling NaN values
                date_obj[index] = None if pd.isna(value) else value

            result.append(date_obj)

        return json.dumps(result)

    @yfinance_server.tool(
        name="get_holder_info",
        description="""Get holder information for a given ticker symbol from yahoo finance. You can choose from the following holder types: major_holders, institutional_holders, mutualfund_holders, insider_transactions, insider_purchases, insider_roster_holders.
    
    Args:
        ticker: str
            The ticker symbol of the stock to get holder information for, e.g. "AAPL"
        holder_type: str
            The type of holder information to get. You can choose from the following holder types: major_holders, institutional_holders, mutualfund_holders, insider_transactions, insider_purchases, insider_roster_holders.
    """,
    )
    async def get_holder_info(ticker: str, holder_type: str) -> str:
        """Get holder information for a given ticker symbol"""

        company = yf.Ticker(ticker)
        try:
            if company.isin is None:
                print(f"Company ticker {ticker} not found.")
                return f"Company ticker {ticker} not found."
        except Exception as e:
            print(f"Error: getting holder info for {ticker}: {e}")
            return f"Error: getting holder info for {ticker}: {e}"

        if holder_type == HolderType.major_holders:
            return company.major_holders.reset_index(names="metric").to_json(orient="records")
        if holder_type == HolderType.institutional_holders:
            return company.institutional_holders.to_json(orient="records")
        if holder_type == HolderType.mutualfund_holders:
            return company.mutualfund_holders.to_json(orient="records", date_format="iso")
        if holder_type == HolderType.insider_transactions:
            return company.insider_transactions.to_json(orient="records", date_format="iso")
        if holder_type == HolderType.insider_purchases:
            return company.insider_purchases.to_json(orient="records", date_format="iso")
        if holder_type == HolderType.insider_roster_holders:
            return company.insider_roster_holders.to_json(orient="records", date_format="iso")
        return f"Error: invalid holder type {holder_type}. Please use one of the following: {HolderType.major_holders}, {HolderType.institutional_holders}, {HolderType.mutualfund_holders}, {HolderType.insider_transactions}, {HolderType.insider_purchases}, {HolderType.insider_roster_holders}."

    @yfinance_server.tool(
        name="get_option_expiration_dates",
        description="""Fetch the available options expiration dates for a given ticker symbol.
    
    Args:
        ticker: str
            The ticker symbol of the stock to get option expiration dates for, e.g. "AAPL"
    """,
    )
    async def get_option_expiration_dates(ticker: str) -> str:
        """Fetch the available options expiration dates for a given ticker symbol."""

        company = yf.Ticker(ticker)
        try:
            if company.isin is None:
                print(f"Company ticker {ticker} not found.")
                return f"Company ticker {ticker} not found."
        except Exception as e:
            print(f"Error: getting option expiration dates for {ticker}: {e}")
            return f"Error: getting option expiration dates for {ticker}: {e}"
        return json.dumps(company.options)

    @yfinance_server.tool(
        name="get_option_chain",
        description="""Fetch the option chain for a given ticker symbol, expiration date, and option type.
    
    Args:
        ticker: str
            The ticker symbol of the stock to get option chain for, e.g. "AAPL"
        expiration_date: str
            The expiration date for the options chain (format: 'YYYY-MM-DD')
        option_type: str
            The type of option to fetch ('calls' or 'puts')
    """,
    )
    async def get_option_chain(ticker: str, expiration_date: str, option_type: str) -> str:
        """Fetch the option chain for a given ticker symbol, expiration date, and option type.

        Args:
            ticker: The ticker symbol of the stock
            expiration_date: The expiration date for the options chain (format: 'YYYY-MM-DD')
            option_type: The type of option to fetch ('calls' or 'puts')

        Returns:
            str: JSON string containing the option chain data
        """

        company = yf.Ticker(ticker)
        try:
            if company.isin is None:
                print(f"Company ticker {ticker} not found.")
                return f"Company ticker {ticker} not found."
        except Exception as e:
            print(f"Error: getting option chain for {ticker}: {e}")
            return f"Error: getting option chain for {ticker}: {e}"

        # Check if the expiration date is valid
        if expiration_date not in company.options:
            return f"Error: No options available for the date {expiration_date}. You can use `get_option_expiration_dates` to get the available expiration dates."

        # Check if the option type is valid
        if option_type not in ["calls", "puts"]:
            return "Error: Invalid option type. Please use 'calls' or 'puts'."

        # Get the option chain
        option_chain = company.option_chain(expiration_date)
        if option_type == "calls":
            return option_chain.calls.to_json(orient="records", date_format="iso")
        if option_type == "puts":
            return option_chain.puts.to_json(orient="records", date_format="iso")
        return f"Error: invalid option type {option_type}. Please use one of the following: calls, puts."

    @yfinance_server.tool(
        name="get_recommendations",
        description="""Get recommendations or upgrades/downgrades for a given ticker symbol from yahoo finance. You can also specify the number of months back to get upgrades/downgrades for, default is 12.
    
    Args:
        ticker: str
            The ticker symbol of the stock to get recommendations for, e.g. "AAPL"
        recommendation_type: str
            The type of recommendation to get. You can choose from the following recommendation types: recommendations, upgrades_downgrades.
        months_back: int
            The number of months back to get upgrades/downgrades for, default is 12.
    """,
    )
    async def get_recommendations(ticker: str, recommendation_type: str, months_back: int = 12) -> str:
        """Get recommendations or upgrades/downgrades for a given ticker symbol"""
        company = yf.Ticker(ticker)
        try:
            if company.isin is None:
                print(f"Company ticker {ticker} not found.")
                return f"Company ticker {ticker} not found."
        except Exception as e:
            print(f"Error: getting recommendations for {ticker}: {e}")
            return f"Error: getting recommendations for {ticker}: {e}"
        try:
            if recommendation_type == RecommendationType.recommendations:
                return company.recommendations.to_json(orient="records")
            if recommendation_type == RecommendationType.upgrades_downgrades:
                # Get the upgrades/downgrades based on the cutoff date
                upgrades_downgrades = company.upgrades_downgrades.reset_index()
                cutoff_date = pd.Timestamp.now() - pd.DateOffset(months=months_back)
                upgrades_downgrades = upgrades_downgrades[
                    upgrades_downgrades["GradeDate"] >= cutoff_date
                    ]
                upgrades_downgrades = upgrades_downgrades.sort_values("GradeDate", ascending=False)
                # Get the first occurrence (most recent) for each firm
                latest_by_firm = upgrades_downgrades.drop_duplicates(subset=["Firm"])
                return latest_by_firm.to_json(orient="records", date_format="iso")
        except Exception as e:
            print(f"Error: getting recommendations for {ticker}: {e}")
            return f"Error: getting recommendations for {ticker}: {e}"

    return yfinance_server


@click.command()
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse"]),
    default="stdio",
    help="Transport type",
)
@click.option("--port", default="8000", help="Port to listen on for SSE")
def main(transport: str, port: str):
    """
    Starts the initialized MCP server.

    :param port: Port for SSE.
    :param transport: The transport type, e.g., `stdio` or `sse`.
    """
    assert transport.lower() in ["stdio", "sse"], \
        "Transport should be `stdio` or `sse`"
    logger = get_logger("Service:yahoo_finance")
    logger.info("Starting the MCP server")
    mcp = build_server(int(port))
    mcp.run(transport=transport.lower())
