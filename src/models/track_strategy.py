"""
Track strategy optimizer for F1 races.
Focuses on tire performance and degradation analysis.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
import logging

@dataclass
class TrackCharacteristics:
    """Track characteristics data class."""
    tire_degradation: float = 0.0
    track_evolution: float = 0.0
    safety_car_probability: float = 0.3
    traffic_impact: float = 0.5
    max_stops: Optional[int] = None
    pit_window_margin: int = 5  # Default ±5 laps
    overtaking_difficulty: float = 0.5  # Scale 0-1
    track_length: float = 5.0  # km
    track_type: str = 'standard'  # standard, street, high_speed
    pit_loss_time: float = 20.0  # seconds

@dataclass
class StopWindow:
    """Pit stop window data class."""
    start: float
    optimal: float
    end: float

class TrackStrategyOptimizer:
    """Track strategy optimizer class."""
    
    # Track-specific configurations
    TRACK_CONFIGS = {
        'monaco': {
            'max_stops': 2,
            'tire_degradation_mod': 0.8,  # 20% lower degradation
            'track_evolution': 0.003,
            'safety_car_prob': 0.6,
            'traffic_impact': 0.8,
            'pit_window_margin': 2,
            'overtaking_difficulty': 0.9,
            'track_length': 3.337,
            'track_type': 'street',
            'pit_loss_time': 23.5
        },
        'spa': {
            'max_stops': 3,
            'tire_degradation_mod': 1.2,  # Higher degradation
            'track_evolution': 0.012,
            'safety_car_prob': 0.4,
            'traffic_impact': 0.4,
            'pit_window_margin': 5,
            'overtaking_difficulty': 0.3,
            'track_length': 7.004,
            'track_type': 'high_speed',
            'pit_loss_time': 18.5
        },
        'monza': {
            'max_stops': 3,
            'tire_degradation_mod': 1.3,  # Higher degradation
            'track_evolution': 0.008,
            'safety_car_prob': 0.3,
            'traffic_impact': 0.3,
            'pit_window_margin': 5,
            'overtaking_difficulty': 0.2,
            'track_length': 5.793,
            'track_type': 'high_speed',
            'pit_loss_time': 21.0
        },
        'australia': {
            'max_stops': 3,
            'tire_degradation_mod': 1.0,
            'track_evolution': 0.013,
            'safety_car_prob': 0.3,
            'traffic_impact': 0.5,
            'pit_window_margin': 4,
            'overtaking_difficulty': 0.6,
            'track_length': 5.278,
            'track_type': 'standard',
            'pit_loss_time': 22.0
        }
    }
    
    def __init__(self, track_name: str, year: int):
        """Initialize track strategy optimizer.
        
        Args:
            track_name: Name of the track
            year: Season year
        """
        self.track_name = track_name.lower()
        self.year = year
        self.track_characteristics = TrackCharacteristics()
        
        # Apply track-specific configurations
        self._configure_track_characteristics()
    
    def _configure_track_characteristics(self):
        """Configure track-specific characteristics."""
        config = self.TRACK_CONFIGS.get(self.track_name, {
            'max_stops': 3,
            'tire_degradation_mod': 1.0,
            'track_evolution': 0.01,
            'safety_car_prob': 0.3,
            'traffic_impact': 0.5,
            'pit_window_margin': 5,
            'overtaking_difficulty': 0.5,
            'track_length': 5.0,
            'track_type': 'standard',
            'pit_loss_time': 20.0
        })
        
        self.track_characteristics.max_stops = config['max_stops']
        self.track_characteristics.track_evolution = config['track_evolution']
        self.track_characteristics.safety_car_probability = config['safety_car_prob']
        self.track_characteristics.traffic_impact = config['traffic_impact']
        self.track_characteristics.pit_window_margin = config['pit_window_margin']
        self.track_characteristics.overtaking_difficulty = config['overtaking_difficulty']
        self.track_characteristics.track_length = config['track_length']
        self.track_characteristics.track_type = config['track_type']
        self.track_characteristics.pit_loss_time = config['pit_loss_time']
    
    def update_characteristics(self, tire_data: Dict, weather_data: Dict):
        """Update track characteristics based on real data.
        
        Args:
            tire_data: Tire performance data
            weather_data: Weather condition data
        """
        # Update tire degradation based on real data
        if tire_data and 'compound_performance' in tire_data:
            deg_values = []
            for compound_data in tire_data['compound_performance'].values():
                if compound_data['avg_tire_life'] > 0:
                    deg = (compound_data['fastest_lap'] - compound_data['avg_lap_time']) / compound_data['avg_tire_life']
                    deg_values.append(abs(deg))
            
            if deg_values:
                base_degradation = np.mean(deg_values)
                # Apply track-specific modifiers
                config = self.TRACK_CONFIGS.get(self.track_name, {'tire_degradation_mod': 1.0})
                self.track_characteristics.tire_degradation = base_degradation * config['tire_degradation_mod']
        
        # Update safety car probability based on weather and track type
        if weather_data:
            base_probability = self.track_characteristics.safety_car_probability
            
            # Weather impact
            if weather_data.get('rainfall', False):
                base_probability *= 1.5  # 50% higher in wet conditions
            if weather_data.get('track_temp', 25) > 40:
                base_probability *= 1.2  # 20% higher in very hot conditions
            
            # Track type impact
            if self.track_characteristics.track_type == 'street':
                base_probability *= 1.3  # 30% higher on street circuits
            elif self.track_characteristics.track_type == 'high_speed':
                base_probability *= 1.1  # 10% higher on high-speed circuits
            
            self.track_characteristics.safety_car_probability = min(base_probability, 1.0)
    
    def _calculate_stop_windows(self, race_distance: float, n_stops: int) -> List[StopWindow]:
        """Calculate optimal pit stop windows.
        
        Args:
            race_distance: Total race distance in km
            n_stops: Number of planned stops
        
        Returns:
            List of stop windows
        """
        windows = []
        stint_length = race_distance / (n_stops + 1)
        margin = self.track_characteristics.pit_window_margin * self.track_characteristics.track_length
        
        for i in range(n_stops):
            optimal_point = (i + 1) * stint_length
            
            # Adjust window based on track characteristics
            if self.track_characteristics.track_type == 'street':
                # Tighter windows for street circuits
                margin *= 0.8
            elif self.track_characteristics.track_type == 'high_speed':
                # Wider windows for high-speed circuits
                margin *= 1.2
            
            window = StopWindow(
                start=max(0, optimal_point - margin),
                optimal=optimal_point,
                end=min(race_distance, optimal_point + margin)
            )
            windows.append(window)
        
        return windows
    
    def _generate_strategy_notes(self, n_stops: int, weather_condition: str) -> List[str]:
        """Generate strategy notes based on track characteristics.
        
        Args:
            n_stops: Number of planned stops
            weather_condition: Current weather condition
        
        Returns:
            List of strategy notes
        """
        notes = []
        
        # Track-specific notes
        if self.track_characteristics.track_type == 'street':
            notes.extend([
                "Track position is critical - prioritize clean air over optimal stop timing",
                f"Safety car probability is {self.track_characteristics.safety_car_probability:.1%} - prepare offset strategy",
                "Overtaking difficult - track position priority over tire management"
            ])
            
            if self.track_name == 'monaco':
                notes.extend([
                    "Consider overcut opportunities due to high track evolution",
                    f"{n_stops}-stop strategy allows for more aggressive tire usage",
                    "Maximum 2 planned stops due to track position importance"
                ])
        
        elif self.track_characteristics.track_type == 'high_speed':
            notes.extend([
                "Multiple overtaking opportunities - focus on optimal stop timing",
                "DRS effect significant - consider undercut opportunities",
                f"Higher tire degradation expected ({self.track_characteristics.tire_degradation:.4f})"
            ])
        
        # General strategy notes
        if n_stops >= 2:
            notes.append(f"{n_stops}-stop strategy allows for more aggressive tire usage")
        
        if self.track_characteristics.safety_car_probability > 0.5:
            notes.append(f"High safety car probability ({self.track_characteristics.safety_car_probability:.1%}) - consider offset strategy")
        
        if self.track_characteristics.track_evolution > 0.01:
            notes.append("High track evolution - monitor grip improvement")
        
        # Weather-specific notes
        if weather_condition == 'wet':
            notes.append("Monitor track evolution for crossover point to dry tires")
            if self.track_characteristics.safety_car_probability > 0.4:
                notes.append("Higher risk of safety car in wet conditions")
        
        return notes
    
    def recommend_strategy(self, race_distance: float, weather_condition: str = 'dry') -> Dict:
        """Recommend pit stop strategy.
        
        Args:
            race_distance: Total race distance in km
            weather_condition: Weather condition ('dry' or 'wet')
        
        Returns:
            Strategy recommendation including number of stops and stop windows
        """
        # Calculate optimal number of stops based on multiple factors
        base_stops = 1
        
        # Tire degradation impact
        if self.track_characteristics.tire_degradation > 0.0006:
            base_stops += 1
        if self.track_characteristics.tire_degradation > 0.001:
            base_stops += 1
        
        # Weather impact
        if weather_condition == 'wet':
            base_stops += 1
        
        # Track type impact
        if self.track_characteristics.track_type == 'high_speed':
            base_stops = min(base_stops + 1, 3)  # More stops for high-speed circuits
        elif self.track_characteristics.track_type == 'street':
            base_stops = max(1, min(base_stops, 2))  # Limit stops for street circuits
        
        # Apply track-specific stop limits
        if self.track_characteristics.max_stops is not None:
            base_stops = min(base_stops, self.track_characteristics.max_stops)
        
        # Calculate stop windows
        stop_windows = self._calculate_stop_windows(race_distance, base_stops)
        
        # Generate strategy notes
        strategy_notes = self._generate_strategy_notes(base_stops, weather_condition)
        
        return {
            'recommended_stops': base_stops,
            'stop_windows': [
                {
                    'start': window.start,
                    'optimal': window.optimal,
                    'end': window.end
                }
                for window in stop_windows
            ],
            'strategy_notes': strategy_notes,
            'track_characteristics': self.track_characteristics
        }
