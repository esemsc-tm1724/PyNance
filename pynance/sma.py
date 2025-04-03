from pynance.base import TradingStrategy
import numpy as np
import pandas as pd


class SMAStrategy(TradingStrategy):

    """
    Simple Moving Average Crossover Strategy.
    
    Buys when short SMA crosses above long SMA, sells when it crosses below.
    """
    
    def __init__(self, data, period, interval, SMA1, SMA2):
        super().__init__(data, period, interval)
        # Ensure both SMA's are in size order
        if SMA1 >= SMA2:
            SMA1, SMA2 = SMA2, SMA1

        # Add 2 SMA's using the method from the parent class
        self.add_sma(SMA1)
        self.add_sma(SMA2)


    def _generate_signals(self):
        """
        Generates trading signals based on SMA crossover.
        Long when short SMA > long SMA, short when opposite.
        """
      
        self.data['Position'] = np.where(self.data[f'{self.sma_list[0]}'] < self.data[f'{self.sma_list[1]}'], 1, -1)
        self.signals_generated = True