from pynance.base import TradingStrategy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 

class LRStrategy(TradingStrategy):

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

        super().__init__(data, period, interval)
        self.lags = lags

        # Create lag colums
        self._create_lags(self.lags)

        # Get Trained model
        self.model = self.train_model()

    def train_model(self):

        """
        Train a Linear Regression model using lagged log returns as features.

        Returns:
        - model (LinearRegression): Trained regression model.
        """

        # Initialise the Linear Regression model
        model = LinearRegression()

        # Turn all the lag columns into a numpy array of shape (n_samples, lags)
        self.X =  self.data[[f'Lag {lag}' for lag in range(1, self.lags + 1)]].values

        # Extract the returns as the targey variable
        self.y = self.data['Log Returns']

        # Fit the model to the data
        model.fit(self.X, self.y)

        # Return the trained model
        return model


    def _generate_signals(self):

        """
        Generate trading signals based on the predicted returns from the regression model.

        A long (1) position is taken when the predicted return is positive,
        and a short (-1) position when it is negative.
        """

        # Use the model to predict the returns
        self.data['Predicted Returns'] = self.model.predict(self.X)
        
        # Generate signals from predicted returns
        self.data['Position'] = np.sign(self.data['Predicted Returns'])


