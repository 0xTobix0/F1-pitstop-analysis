"""
F1 Pit Stop Strategy Optimizer

A comprehensive pit stop strategy optimization system for Formula 1 races.
The package provides real-time strategy recommendations based on track conditions,
tire performance, and race situations.

Key Components:
- Track Strategy: Core strategy calculation engine
- FastF1 Integration: Real-time data processing
- Track Characteristics: Circuit-specific parameters
- Stop Window: Pit stop timing optimization

Project Structure:
/src/models/
  - track_strategy.py: Main strategy logic
  - fastf1_loader.py: Data integration
/cache/
  - 2024/: Current season data
  - 2023/: Reference race data

Dependencies:
- FastF1 >= 3.0.0
- NumPy >= 1.24.0
- Pandas >= 2.0.0

Author: 0xTobix0
"""

from src.models.track_strategy import TrackStrategyOptimizer
from src.models.fastf1_loader import FastF1DataLoader

__version__ = '1.0.0'
__all__ = ['TrackStrategyOptimizer', 'FastF1DataLoader']
