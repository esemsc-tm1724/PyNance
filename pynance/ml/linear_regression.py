from pynance.ml import MLStrategy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 

class LRStrategy(MLStrategy):

    """
    A Linear Regression-based trading strategy.

    This strategy uses lagged log returns as features to predict future returns using
    a linear regression model. Trading signals are generated based on the sign of the predicted returns.
    """
     
    def __init__(self, data, period, interval, lags):

        """
        Initialize the Linear Regression strategy.

        Parameters:
        - data (pd.DataFrame): Historical price data.
        - period (str): Period string used when pulling data (e.g., '1y').
        - interval (str): Data resolution (e.g., '1d').
        - lags (int): Number of lagged return columns to use as features.
        """

        super().__init__(data, period, interval, lags)
        self.lags = lags

        # Create lag colums
        self._create_lags(self.lags)

        # Instantiate linear regression model
        self.model = LinearRegression()

        # Train the model
        self.train_model(self.model)

    def _generate_signals(self):

        """
        Generate trading signals based on the predicted returns from the regression model.

        A long (1) position is taken when the predicted return is positive,
        and a short (-1) position when it is negative.
        """

        # Generate signals from predicted returns
        self.data['Position'] = np.sign(self.data['Predicted Log Returns'])


