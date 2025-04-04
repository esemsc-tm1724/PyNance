import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt


class TradingStrategy:
    """Base class for backtesting trading strategies on financial time series data.

    This class provides a general framework for calculating returns, plotting results,
    managing indicators, handling transaction costs, and evaluating performance metrics.

    Subclasses must implement the `_generate_signals()` method.
    """

    def __init__(self, data, period, interval):

        """Initialize the strategy with historical price data and time resolution.

        Args:
            data (pd.DataFrame): Historical price data (must contain 'Close').
            period (str): The data period (e.g., '1y', '6mo').
            interval (str): The time interval (e.g., '1d', '1h').
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data attribute expects a pandas DataFrame")
        self.data = data
        self.period = period
        self.interval = interval
        self.sma_list = []
        self.strategy_returns = False
        self.risk_free_rate = self._get_risk_free_rate()
        self.total_time = (self.data.index[-1] - self.data.index[0]).total_seconds()

        self._calc_returns()

    def _calc_returns(self):

        """Calculate log returns and cumulative returns for the price series."""

        self.data["Log Returns"] = np.log(
            self.data["Close"] / self.data["Close"].shift(1)
        )
        self.data["Cum Log Returns"] = self.data["Log Returns"].cumsum()
        self.data["Cum Returns"] = np.exp(self.data["Cum Log Returns"])
        self.data.dropna(inplace=True)
        return

    def add_sma(self, SMA):

        """Add a simple moving average (SMA) column to the dataset.

        Args:
            SMA (int): The window length of the SMA.
        """

        self.data[f"SMA {SMA}"] = self.data["Close"].rolling(SMA).mean()
        self.sma_list.append(f"SMA {SMA}")
        return
    
    def plot_close(self, title, show_SMA=False):

        """Plot historical close price data and technical indicators
    
        Args:
            title (str): Title of the plot.
            show_SMA (bool): Choice to plot the SMA's.
        """

        plt.figure(figsize=(14, 6)).add_axes([0.1, 0.1, 0.85, 0.8])
        plt.plot(
            self.data.index,
            self.data["Close"],
            label="Closing Price",
            linewidth=2,
            color="red",
        )

        if show_SMA:
            for sma in self.sma_list:
                plt.plot(
                self.data.index,
                self.data[f"{sma}"],
                label=f"{sma}",
                linewidth=2,
                linestyle="--",
                )


        plt.title(title, fontsize=16)
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()
        return


    def plot_returns(self, title):

        """Plot cumulative market and strategy returns.

        Args:
            title (str): Title of the plot.
        """

        plt.figure(figsize=(14, 6)).add_axes([0.1, 0.1, 0.85, 0.8])
        plt.plot(
            self.data.index,
            self.data["Cum Returns"],
            label="Market Returns",
            linewidth=2,
            color="red",
        )

        if self.strategy_returns:
            # plt.plot(self.data.index, self.data['Strat Cum Returns'], label='Strategy Returns', linewidth=2, color='green')
            plt.plot(
                self.data.index,
                self.data["Strat Cum Returns Adj"],
                label="Strategy Returns Adjusted for Transaction Costs",
                linewidth=2,
                color="blue",
            )

        plt.title(title, fontsize=16)
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()
        return

    def _generate_signals(self):

        """Abstract method to generate trading signals. Should be implemented by subclasses."""

        return NotImplementedError

    def plot_signals(self):

        """Plot the position signal as a step function over time, with horizontal reference lines."""

        plt.figure(figsize=(14, 6)).add_axes([0.1, 0.1, 0.85, 0.8])

        # Plot the position as a step function
        plt.step(
            self.data.index,
            self.data["Position"],
            where="mid",
            label="Position Signal",
            color="purple",
        )

        # Add horizontal lines for reference
        plt.axhline(y=1, color="green", linestyle="--", linewidth=1, label="Long")
        plt.axhline(y=0, color="gray", linestyle="--", linewidth=1, label="Neutral")
        plt.axhline(y=-1, color="red", linestyle="--", linewidth=1, label="Short")

        # Styling
        plt.title("Trading Position Over Time")
        plt.xlabel("Date")
        plt.ylabel("Position")
        plt.yticks([-1, 0, 1])
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        return

    def back_test(self, transaction_cost, print_results=True):

        """Run a full backtest on the strategy including metrics and transaction costs.

        Args:
            transaction_cost (float): Proportional transaction fee per trade (e.g., 0.01 for 1%).
            print_results (bool): Whether to print performance metrics to console.
        """

        self._generate_signals()

        # Calculate strategy returns
        self.data["Strat Log Returns"] = self.data["Position"] * self.data[
            "Log Returns"
        ].shift(1)
        self.data["Strat Returns"] = self.data["Strat Log Returns"].apply(np.exp)

        # Calculate strategy returns adjusted for transaction costs
        self.data["Position Change"] = self.data["Position"].diff().abs()
        self.data["Strat Log Returns Adj"] = self.data["Strat Log Returns"] + self.data[
            "Position Change"
        ] * np.log(1 - transaction_cost)
        self.data["Strat Returns Adj"] = np.exp(self.data["Strat Log Returns Adj"])

        # Generate cumulative returns
        self.data["Strat Cum Log Returns"] = self.data["Strat Log Returns"].cumsum()
        self.data["Strat Cum Returns"] = np.exp(self.data["Strat Cum Log Returns"])
        self.data["Strat Cum Log Returns Adj"] = self.data[
            "Strat Log Returns Adj"
        ].cumsum()
        self.data["Strat Cum Returns Adj"] = np.exp(
            self.data["Strat Cum Log Returns Adj"]
        )
        self.strategy_returns = True

        # Calculate performance metrics (Adjusted)
        self.pct_return = (
            self.data["Strat Cum Returns Adj"].iloc[-1]
            - self.data["Strat Cum Returns Adj"].dropna().iloc[0]
        )
        self.win_rate = (self.data["Strat Log Returns Adj"] > 0).sum() / self.data[
            "Strat Log Returns Adj"
        ].count()
        self.average_return = self.data["Strat Returns Adj"].mean()
        self.average_gain = self.data["Strat Log Returns Adj"][
            self.data["Strat Log Returns Adj"] > 0
        ].mean()
        self.average_loss = self.data["Strat Log Returns Adj"][
            self.data["Strat Log Returns Adj"] < 0
        ].mean()

        gains = (
            self.data["Strat Returns Adj"][self.data["Strat Returns Adj"] > 1]
            .dropna()
            .sum()
        )
        losses = (
            self.data["Strat Returns Adj"][self.data["Strat Returns Adj"] < 1]
            .dropna()
            .sum()
        )
        self.profit_factor = gains / losses - 1 if losses != 0 else np.nan

        self.max_drawdown = (
            self.data["Strat Cum Returns Adj"].cummax()
            - self.data["Strat Cum Returns Adj"]
        ).max()
        self.trade_count = (self.data["Position"] != 0).sum()

        # Calculate Volatility and Sharpe Ratios (NOTE They are being calculated in log space)
        avg_interval = self.total_time / (len(self.data) - 1)
        seconds_per_year = 365.25 * 24 * 60 * 60
        periods_per_year = seconds_per_year / avg_interval
        self.volatility = self.data["Strat Log Returns"].std() * np.sqrt(
            periods_per_year
        )

        log_risk_free_rate = np.log(1 + self.risk_free_rate)
        self.sharpe_ratio = (
            (self.data["Strat Log Returns Adj"].mean() - log_risk_free_rate)
            / self.data["Strat Log Returns Adj"].std()
        ) * np.sqrt(periods_per_year)

        if print_results:
            print(f"{'Metric':<25}Value")
            print("-" * 40)
            print(f"{'Percentage Return':<25}{self.pct_return:.2%}")
            print(f"{'Win Rate':<25}{self.win_rate:.2%}")
            print(f"{'Average Gain':<25}{self.average_gain:.4%}")
            print(f"{'Average Loss':<25}{self.average_loss:.4%}")
            print(f"{'Profit Factor':<25}{self.profit_factor:.2f}")
            print(f"{'Sharpe Ratio':<25}{self.sharpe_ratio:.2f}")
            print(f"{'Volatility (Ann)':<25}{self.volatility:.2%}")
            print(f"{'Max Drawdown':<25}{self.max_drawdown:.2%}")
            print(f"{'Trade Count':<25}{self.trade_count}")

    def _get_risk_free_rate(self, annual_rf=0.03):

        """Calculate the per-period risk-free rate based on backtest frequency.

        Args:
            annual_rf (float): Annualized risk-free rate (default: 0.03).

        Returns:
            float: Risk-free rate per period based on `self.interval`.
        """

        freq_map = {
            "1d": 252,
            "1h": 252 * 6.5,
            "30m": 252 * 13,
            "15m": 252 * 26,
            "5m": 252 * 78,
            "1m": 252 * 390,
        }

        periods = freq_map.get(self.interval)
        if periods is None:
            raise ValueError(f"Unsupported frequency: {self.interval}")

        return annual_rf / periods

    def _create_lags(self, lags):

        """Create lagged return columns for machine learning strategies.

        Args:
            lags (int): Number of lag periods to create.
        """
         
        # Create columns with lagged returns, this will be needed for training some ML models
        for lag in range(1, lags + 1):
            self.data[f"Lag {lag}"] = self.data["Log Returns"].shift(lag)

        # Remove nan values
        self.data.dropna(inplace=True)
