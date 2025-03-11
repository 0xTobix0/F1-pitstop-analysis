"""
FastF1 data loader for F1 strategy optimization.
Handles loading and processing of F1 telemetry and session data.
"""

import fastf1
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
import logging
from dataclasses import dataclass

# Enable FastF1 cache
fastf1.Cache.enable_cache('cache')

# Track-specific configurations
TRACK_INFO = {
    'monaco': {
        'length': 3.337,
        'laps': 78,
        'tire_deg_modifier': 0.8,  # 20% lower degradation due to slower speeds
        'traffic_impact': 0.8,     # High traffic impact
        'sc_probability': 0.4,     # Base safety car probability
        'pit_window_margin': 2,    # Tighter pit windows (±2 laps)
        'track_position_weight': 0.9,  # Very high importance of track position
        'base_tire_deg': 0.03      # Base tire degradation rate
    },
    'monza': {
        'length': 5.793,
        'laps': 53,
        'tire_deg_modifier': 1.2,  # Higher deg due to high speeds
        'traffic_impact': 0.3,
        'sc_probability': 0.2,
        'pit_window_margin': 4,
        'track_position_weight': 0.4,
        'base_tire_deg': 0.04
    },
    'spa': {
        'length': 7.004,
        'laps': 44,
        'tire_deg_modifier': 1.1,
        'traffic_impact': 0.4,
        'sc_probability': 0.25,
        'pit_window_margin': 4,
        'track_position_weight': 0.5,
        'base_tire_deg': 0.035
    },
    'australia': {
        'length': 5.278,
        'laps': 58,
        'tire_deg_modifier': 1.0,
        'traffic_impact': 0.5,
        'sc_probability': 0.3,
        'pit_window_margin': 3,
        'track_position_weight': 0.6,
        'base_tire_deg': 0.03
    }
}

@dataclass
class TireStintData:
    """Data for a single tire stint."""
    compound: str
    start_lap: int
    end_lap: int
    avg_lap_time: float
    degradation: float
    tire_life: float

class FastF1DataLoader:
    """Enhanced F1 data loader using FastF1 integration."""
    
    def __init__(self):
        """Initialize the data loader."""
        self.session = None
        self.track_name = None
        self.track_info = None
    
    def load_session(self, year: int, track_name: str, session_type: str = 'R') -> Dict:
        """Load session data for a specific race.
        
        Args:
            year: Season year
            track_name: Name of the track (lowercase)
            session_type: Session type (R for Race, Q for Qualifying)
            
        Returns:
            Dictionary containing processed session data
        """
        try:
            self.track_name = track_name.lower()
            self.track_info = TRACK_INFO.get(self.track_name, {
                'length': 5.0,
                'laps': 50,
                'tire_deg_modifier': 1.0,
                'traffic_impact': 0.5,
                'sc_probability': 0.3,
                'pit_window_margin': 3,
                'track_position_weight': 0.6,
                'base_tire_deg': 0.03
            })
            
            self.session = fastf1.get_session(year, track_name, session_type)
            self.session.load()
            
            # Get race distance
            race_distance = self.track_info['length'] * self.track_info['laps']
            
            return {
                'weather': self._get_weather_data(),
                'tire_performance': self._get_tire_performance(),
                'track_evolution': self._calculate_track_evolution(),
                'race_distance': race_distance,
                'track_length': self.track_info['length']
            }
            
        except Exception as e:
            logging.error(f"Error loading session data: {e}")
            raise
    
    def _get_weather_data(self) -> Dict:
        """Get detailed weather data including track conditions."""
        try:
            weather = self.session.weather_data
            
            # Calculate weather stability
            temp_variance = weather['TrackTemp'].std()
            wind_variance = weather['WindSpeed'].std()
            
            # Adjust safety car probability based on weather
            base_sc_prob = self.track_info['sc_probability']
            weather_sc_modifier = 1.0
            if weather['Rainfall'].any():
                weather_sc_modifier = 1.5  # 50% higher in wet conditions
            elif weather['TrackTemp'].mean() > 40:
                weather_sc_modifier = 1.2  # 20% higher in very hot conditions
            
            conditions = {
                'condition': 'wet' if weather['Rainfall'].any() else 'dry',
                'track_temp': float(weather['TrackTemp'].mean()),
                'air_temp': float(weather['AirTemp'].mean()),
                'humidity': float(weather['Humidity'].mean()),
                'rainfall': bool(weather['Rainfall'].any()),
                'wind_speed': float(weather['WindSpeed'].mean()),
                'temp_variance': float(temp_variance),
                'wind_variance': float(wind_variance),
                'weather_stability': 1.0 - min(1.0, (temp_variance/10 + wind_variance/20)),
                'safety_car_probability': min(0.8, base_sc_prob * weather_sc_modifier)
            }
            
            return conditions
            
        except Exception as e:
            logging.warning(f"Error getting weather data: {e}")
            return {
                'condition': 'dry',
                'track_temp': 25.0,
                'air_temp': 20.0,
                'humidity': 50.0,
                'rainfall': False,
                'wind_speed': 10.0,
                'temp_variance': 0.0,
                'wind_variance': 0.0,
                'weather_stability': 1.0,
                'safety_car_probability': self.track_info['sc_probability']
            }
    
    def _get_tire_performance(self) -> Dict:
        """Get comprehensive tire performance analysis."""
        try:
            compounds = self.session.laps['Compound'].unique()
            compound_performance = {}
            
            for compound in compounds:
                compound_laps = self.session.laps[self.session.laps['Compound'] == compound]
                if not compound_laps.empty:
                    # Calculate degradation from lap time progression
                    lap_times = compound_laps.sort_values('LapNumber')['LapTime'].dt.total_seconds()
                    if len(lap_times) > 1:
                        # Calculate degradation as percentage increase per lap
                        initial_times = lap_times.head(3).mean()
                        final_times = lap_times.tail(3).mean()
                        laps_between = len(lap_times)
                        if laps_between > 0 and initial_times > 0:
                            degradation = ((final_times - initial_times) / initial_times) / laps_between
                        else:
                            degradation = self.track_info['base_tire_deg']
                    else:
                        degradation = self.track_info['base_tire_deg']
                    
                    # Apply track-specific tire degradation modifier
                    degradation *= self.track_info['tire_deg_modifier']
                    
                    compound_performance[compound] = {
                        'avg_lap_time': float(compound_laps['LapTime'].mean().total_seconds()),
                        'fastest_lap': float(compound_laps['LapTime'].min().total_seconds()),
                        'avg_tire_life': float(compound_laps['TyreLife'].mean()),
                        'degradation': float(degradation),
                        'stint_count': int(len(compound_laps['Stint'].unique())),
                        'total_laps': int(len(compound_laps))
                    }
            
            # Analyze stint data
            stints = self._analyze_stints()
            
            return {
                'compound_performance': compound_performance,
                'stints': stints,
                'pit_stops': len(self.session.laps[self.session.laps['PitOutTime'].notna()]),
                'avg_stint_length': float(self.session.laps['TyreLife'].mean()),
                'traffic_impact': self.track_info['traffic_impact']
            }
            
        except Exception as e:
            logging.warning(f"Error analyzing tire performance: {e}")
            base_deg = self.track_info['base_tire_deg'] * self.track_info['tire_deg_modifier']
            return {
                'compound_performance': {
                    'SOFT': {
                        'avg_lap_time': 80.0,
                        'fastest_lap': 78.0,
                        'avg_tire_life': 20.0,
                        'degradation': base_deg * 1.2,
                        'stint_count': 1,
                        'total_laps': 20
                    },
                    'MEDIUM': {
                        'avg_lap_time': 81.0,
                        'fastest_lap': 79.0,
                        'avg_tire_life': 30.0,
                        'degradation': base_deg,
                        'stint_count': 1,
                        'total_laps': 30
                    },
                    'HARD': {
                        'avg_lap_time': 82.0,
                        'fastest_lap': 80.0,
                        'avg_tire_life': 40.0,
                        'degradation': base_deg * 0.8,
                        'stint_count': 1,
                        'total_laps': 40
                    }
                },
                'stints': [],
                'pit_stops': 2,
                'avg_stint_length': 25.0,
                'traffic_impact': self.track_info['traffic_impact']
            }
    
    def _analyze_stints(self) -> List[TireStintData]:
        """Analyze individual tire stints."""
        stints = []
        
        try:
            for driver in self.session.drivers:
                driver_laps = self.session.laps.pick_driver(driver)
                
                for stint in driver_laps['Stint'].unique():
                    stint_laps = driver_laps[driver_laps['Stint'] == stint]
                    if not stint_laps.empty:
                        lap_times = stint_laps['LapTime'].dt.total_seconds()
                        
                        # Calculate degradation for this stint
                        if len(lap_times) > 1:
                            initial_times = lap_times.head(3).mean()
                            final_times = lap_times.tail(3).mean()
                            laps_between = len(lap_times)
                            if laps_between > 0 and initial_times > 0:
                                degradation = ((final_times - initial_times) / initial_times) / laps_between
                            else:
                                degradation = self.track_info['base_tire_deg']
                        else:
                            degradation = self.track_info['base_tire_deg']
                        
                        # Apply track-specific modifier
                        degradation *= self.track_info['tire_deg_modifier']
                        
                        stints.append(TireStintData(
                            compound=stint_laps['Compound'].iloc[0],
                            start_lap=int(stint_laps['LapNumber'].min()),
                            end_lap=int(stint_laps['LapNumber'].max()),
                            avg_lap_time=float(lap_times.mean()),
                            degradation=float(degradation),
                            tire_life=float(stint_laps['TyreLife'].mean())
                        ))
        except Exception as e:
            logging.warning(f"Error analyzing stints: {e}")
        
        return stints
    
    def _calculate_track_evolution(self) -> float:
        """Calculate track evolution rate."""
        try:
            # Get lap times for each sector
            lap_times = self.session.laps['LapTime'].dt.total_seconds()
            
            if len(lap_times) > 10:
                # Calculate improvement rate over the session
                initial_times = lap_times.head(5).mean()
                final_times = lap_times.tail(5).mean()
                if initial_times > 0:
                    evolution_rate = (initial_times - final_times) / initial_times
                    return float(max(0.0, min(0.03, evolution_rate)))  # Cap between 0% and 3%
            
            return self.track_info.get('base_evolution', 0.01)
            
        except Exception as e:
            logging.warning(f"Error calculating track evolution: {e}")
            return self.track_info.get('base_evolution', 0.01)
