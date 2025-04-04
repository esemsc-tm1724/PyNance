from pynance.base import TradingStrategy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MLStrategy(TradingStrategy):

    def __init__(self, data, period, interval, lags):
        super().__init__(data, period, interval)

        # Amount of time lag to pass to the model
        self.lags = lags


    def train_model(self, model, train=0.7):

        """
        Performs walk-forward validation to simulate live trading.

        Parameters:
        - model: Any sklearn-like model with fit/predict methods.
        - train_size (int): Initial number of observations to train on.

        Populates:
        - self.data['Predicted Returns']: One-step-ahead predicted returns.
        """

        predictions = []
        true_returns = []

        train_size = int(len(self.data) * train)

        for i in range(train_size, len(self.data)):
            # Slice up to current day for training
            X_train = self.data[[f"Lag {lag}" for lag in range(1, self.lags + 1)]].iloc[i - train_size:i].values
            y_train = self.data['Log Returns'].iloc[i - train_size:i].values

            # Train model
            model.fit(X_train, y_train)

            # Predict 1 day ahead
            X_test = self.data[[f"Lag {lag}" for lag in range(1, self.lags + 1)]].iloc[i].values.reshape(1, -1)
            pred = model.predict(X_test)[0]

            predictions.append(pred)
            true_returns.append(self.data['Log Returns'].iloc[i])

        # Truncate the time series up to the point where the model starts predicting 
        self.data = self.data.iloc[train_size:].copy()
        self.data['Predicted Log Returns'] = predictions
        # Recalculate market returns from the point the model starts predicting
        self._calc_returns()


    def _generate_signals(self):

        """Abstract method to generate trading signals. Should be implemented by subclasses."""

        return NotImplementedError