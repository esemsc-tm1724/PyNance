from pynance.base import TradingStrategy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MLStrategy(TradingStrategy):

    def __init__(self, data, period, interval, lags):
        super().__init__(data, period, interval)

        # Amount of time lag to pass to the model
        self.lags = lags


    def train_model(self, model):

        """
        Train a machine learning model using lagged log returns as features.

        Returns:
        - model : Trained model.
        """


        # Turn all the lag columns into a numpy array of shape (n_samples, lags)
        self.X =  self.data[[f'Lag {lag}' for lag in range(1, self.lags + 1)]].values

        # Extract the returns as the targey variable
        self.y = self.data['Log Returns']

        # Fit the model to the data
        model.fit(self.X, self.y)

        # Return the trained model
        return model


    def _generate_signals(self):

         """Abstract method to generate trading signals. Should be implemented by subclasses."""
         
        return NotImplementedError