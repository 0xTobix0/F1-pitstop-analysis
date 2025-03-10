"""
Track Strategy Optimizer for F1 Pitstop Analysis.
This module provides tools for analyzing and optimizing race strategies for specific tracks.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class TrackCharacteristics:
    """Data class to store track-specific characteristics."""
    name: str
    avg_lap_time: float  # in seconds
    tire_degradation: float  # in ms/lap
    pit_loss_time: float  # in seconds
    overtaking_difficulty: float  # scale 0-1
    sc_probability: float  # probability of safety car appearance

class TrackStrategyOptimizer:
    def __init__(self, track_data: pd.DataFrame):
        """Initialize the track strategy optimizer.
        
        Args:
            track_data: DataFrame containing track-specific race data
        """
        self.track_data = track_data
        self.track_characteristics = self._analyze_track_characteristics()

    def _analyze_track_characteristics(self) -> TrackCharacteristics:
        """Analyze track data to determine key characteristics."""
        # Calculate average lap time
        avg_lap_time = self.track_data['milliseconds_x'].mean() / 1000
        
        # Calculate tire degradation
        tire_deg = self.track_data['lap_degradation'].mean()
        
        # Calculate pit loss time (average pit duration + delta to racing line)
        pit_loss = self.track_data[self.track_data['is_pit_stop'] == 1]['duration'].mean()
        
        # Calculate overtaking difficulty based on position changes
        pos_changes = abs(self.track_data['position_x'].diff()).mean()
        overtaking_diff = 1 / (1 + pos_changes)  # Normalize to 0-1
        
        # Calculate safety car probability
        total_races = len(self.track_data['raceId'].unique())
        sc_races = len(self.track_data[
            self.track_data['milliseconds_x'] > 
            self.track_data['milliseconds_x'].median() * 1.4
        ]['raceId'].unique())
        sc_prob = sc_races / total_races if total_races > 0 else 0
        
        return TrackCharacteristics(
            name=self.track_data['name'].iloc[0],
            avg_lap_time=avg_lap_time,
            tire_degradation=tire_deg,
            pit_loss_time=pit_loss,
            overtaking_difficulty=overtaking_diff,
            sc_probability=sc_prob
        )

    def recommend_strategy(self, race_distance: int, 
                         weather_condition: str = 'dry') -> Dict:
        """Recommend optimal pit stop strategy for the track.
        
        Args:
            race_distance: Race distance in laps
            weather_condition: Current weather conditions ('dry' or 'wet')
            
        Returns:
            Dictionary containing recommended strategy
        """
        # Base number of stops based on tire degradation and race distance
        base_stops = int(np.ceil(
            (race_distance * abs(self.track_characteristics.tire_degradation)) / 
            (1000 * 60)  # Threshold of 1 minute degradation
        ))
        
        # Adjust for track characteristics
        if self.track_characteristics.overtaking_difficulty > 0.7:
            # Hard to overtake -> minimize stops
            base_stops = max(1, base_stops - 1)
        
        if self.track_characteristics.sc_probability > 0.6:
            # High SC probability -> keep flexibility for opportunistic stops
            base_stops = min(base_stops + 1, 3)
        
        # Calculate optimal stop windows
        window_size = race_distance // (base_stops + 1)
        stop_windows = [(i * window_size, (i + 1) * window_size) 
                       for i in range(base_stops)]
        
        return {
            'recommended_stops': base_stops,
            'stop_windows': stop_windows,
            'strategy_notes': self._generate_strategy_notes(base_stops),
            'track_characteristics': self.track_characteristics
        }
    
    def _generate_strategy_notes(self, num_stops: int) -> List[str]:
        """Generate strategy-specific notes based on track characteristics."""
        notes = []
        tc = self.track_characteristics
        
        if tc.overtaking_difficulty > 0.7:
            notes.append("Track position critical - prioritize clean air")
        
        if tc.sc_probability > 0.6:
            notes.append("High SC probability - maintain strategy flexibility")
        
        if tc.tire_degradation < -500:
            notes.append("High tire degradation - monitor tire life closely")
        
        if tc.pit_loss_time > 25:
            notes.append("Long pit loss time - extend stints if possible")
            
        return notes
