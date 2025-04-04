from pynance.base import TradingStrategy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MRStrategy(TradingStrategy):

    """
    Mean Reversion Strategy.

    This strategy assumes that prices will revert to a mean value over time.
    It compares the current price to a Simple Moving Average (SMA) and takes
    a position when the deviation exceeds a defined standard deviation threshold.
    
    - Long when price is significantly below the SMA
    - Short when price is significantly above the SMA
    - Neutral otherwise
    """

    def __init__(self, data, period, interval, SMA = 25):

        """
        Initializes the Mean Reversion strategy.

        Parameters:
        - data (pd.DataFrame): Historical price data.
        - period (str): Duration of historical data (e.g., '1y').
        - interval (str): Data frequency (e.g., '1d').
        - SMA (int): Window for the Simple Moving Average.
        """

        super().__init__(data, period, interval)
     
        # Add an SMA to check deviation of the price
        self.add_sma(SMA)
        
        # Calculate price deviation
        self.data['Deviation'] = self.data[f'SMA {SMA}'] - self.data['Close']
        self.boundary = self.data['Deviation'].std() # Calculate boundary as 1 SD

    def plot_deviation(self):

        """
        Plots the deviation from the SMA along with upper and lower thresholds.

        Visually highlights the zones where trades will be entered or avoided
        based on how far price has deviated from its mean.
        """

        plt.figure(figsize=(14, 6)).add_axes([0.1, 0.1, 0.85, 0.8]) 
        plt.plot(self.data['Deviation'], label='Deviation', color='blue')

        # Horizontal lines
        plt.axhline(y=0, color='red', linestyle='--', label='Mean')
        plt.axhline(y=self.boundary, color='green', linestyle='--', label='Upper Bound')
        plt.axhline(y=-self.boundary, color='green', linestyle='--', label='Lower Bound')

        # Shading between bounds 
        plt.fill_between(self.data.index, -self.boundary, self.boundary, color='green', alpha=0.1, label='No Trade Zone')

        # Titles and labels
        plt.title('Deviation from Mean with Thresholds')
        plt.xlabel('Date')
        plt.ylabel('Deviation')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    def _generate_signals(self):

        """
        Generates trading signals based on price deviation from SMA.

        Signal logic:
        - Long (+1) if price is below SMA by more than 1 std deviation.
        - Short (-1) if price is above SMA by more than 1 std deviation.
        - Flat (0) otherwise.
        """

        conditions = [
            self.data['Deviation'] > self.boundary,    
            self.data['Deviation'] < -self.boundary    
        ]
        choices = [-1, 1]

        self.data['Position'] = np.select(conditions, choices, default=0)
        self.data['Position'].fillna(0, inplace=True)  
        self.signals_generated = True