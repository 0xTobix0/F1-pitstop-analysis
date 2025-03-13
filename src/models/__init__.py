"""
F1 Strategy Models Package

Core models and data processing components for F1 pit stop strategy optimization.

Components:
1. Track Strategy Model:
   - Strategy calculation engine
   - Track characteristics modeling
   - Stop window optimization
   - Position-based adjustments

2. FastF1 Data Integration:
   - Real-time session data
   - Weather analysis
   - Tire performance tracking
   - Track evolution modeling

Cache Structure:
- 2024/: Current season data
- 2023/: Key reference races (Monaco, Monza, Spa)
- cache_backup/: Historical data archive

For usage examples and detailed documentation, see the main package README.
"""

from .track_strategy import TrackStrategyOptimizer
from .fastf1_loader import FastF1DataLoader

__all__ = ['TrackStrategyOptimizer', 'FastF1DataLoader']
