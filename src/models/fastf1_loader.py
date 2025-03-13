"""
FastF1 Data Integration Module

This module handles the integration with FastF1 API for real-time F1 race data.
It provides optimized data loading, caching, and processing for strategy calculations.

Key Features:
1. Real-time session data loading
2. Weather and track condition analysis
3. Tire performance calculations
4. Track evolution modeling
5. Efficient caching system

The module is designed to work with the 2024 season data and key reference
races from 2023 (Monaco, Monza, Spa) for historical analysis.
"""

import fastf1
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
import logging
from dataclasses import dataclass

# Enable FastF1 cache with optimized structure
# Current season (2024) and key reference races (2023)
fastf1.Cache.enable_cache('cache')

# Track-specific configurations and characteristics
TRACK_INFO = {
    'monaco': {
        'length': 3.337,          # Track length in kilometers
        'laps': 78,              # Standard race length
        'tire_deg_modifier': 0.8,  # 20% lower degradation due to slower speeds
        'traffic_impact': 0.8,     # High impact due to narrow track
        'sc_probability': 0.4,     # Higher safety car probability
        'pit_window_margin': 2,    # Narrow pit windows due to track nature
        'track_position_weight': 0.9,  # Track position critical
        'base_tire_deg': 0.03      # Base tire wear rate
    },
    'monza': {
        'length': 5.793,
        'laps': 53,
        'tire_deg_modifier': 1.2,  # Higher wear from high speeds
        'traffic_impact': 0.3,     # Low due to long straights
        'sc_probability': 0.2,     # Lower due to run-off areas
        'pit_window_margin': 4,    # Flexible due to overtaking opportunities
        'track_position_weight': 0.4,  # Less critical
        'base_tire_deg': 0.04      # Higher base wear
    },
    'spa': {
        'length': 7.004,
        'laps': 44,
        'tire_deg_modifier': 1.1,  # Moderate-high wear
        'traffic_impact': 0.4,     # Moderate
        'sc_probability': 0.25,    # Moderate
        'pit_window_margin': 4,    # Standard
        'track_position_weight': 0.5,  # Balanced
        'base_tire_deg': 0.035     # Moderate wear
    },
    'australia': {
        'length': 5.278,
        'laps': 58,
        'tire_deg_modifier': 1.0,  # Standard wear
        'traffic_impact': 0.5,     # Moderate
        'sc_probability': 0.3,     # Moderate
        'pit_window_margin': 3,    # Standard
        'track_position_weight': 0.6,  # Moderately important
        'base_tire_deg': 0.03      # Standard wear
    }
}

@dataclass
class TireStintData:
    """
    Data structure for analyzing individual tire stints.
    
    Attributes:
        compound (str): Tire compound (soft/medium/hard/inter/wet)
        start_lap (int): First lap of the stint
        end_lap (int): Last lap of the stint
        avg_lap_time (float): Average lap time during stint
        degradation (float): Tire degradation rate
        tire_life (float): Total life of tire in laps
    """
    compound: str
    start_lap: int
    end_lap: int
    avg_lap_time: float
    degradation: float
    tire_life: float

class FastF1DataLoader:
    """
    Enhanced F1 data loader using FastF1 integration.
    
    This class handles:
    1. Session data loading and caching
    2. Weather condition analysis
    3. Tire performance calculations
    4. Track evolution modeling
    5. Race distance and timing calculations
    
    The loader maintains an optimized cache structure:
    - 2024 season: All races
    - 2023 season: Key reference races
    - Historical data: Archived in cache_backup/
    """
    
    def __init__(self):
        """Initialize the data loader with empty session state."""
        self.session = None
        self.track_name = None
        self.track_info = None
    
    def load_session(self, year: int, track_name: str, session_type: str = 'R') -> Dict:
        """
        Load and process F1 session data.
        
        This method:
        1. Loads session data from FastF1
        2. Processes weather conditions
        3. Analyzes tire performance
        4. Calculates track evolution
        5. Determines race characteristics
        
        Args:
            year (int): Season year (primarily 2024, some 2023)
            track_name (str): Name of the track (lowercase)
            session_type (str): Session type (R=Race, Q=Qualifying)
            
        Returns:
            Dict: Processed session data including:
                - weather: Current and predicted conditions
                - tire_performance: Compound-specific analysis
                - track_evolution: Grip improvement rate
                - race_distance: Total race distance
                - track_length: Circuit length
                
        Raises:
            Exception: If session data cannot be loaded
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
            
            # Calculate total race distance
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
        """
        Analyze and process weather conditions.
        
        This method calculates:
        1. Track and air temperatures
        2. Weather stability metrics
        3. Safety car probability adjustments
        4. Rain impact on strategy
        
        Returns:
            Dict: Weather analysis including:
                - condition: Current track condition (wet/dry)
                - track_temp: Average track temperature
                - air_temp: Average air temperature
                - humidity: Average humidity
                - rainfall: Presence of rain
                - wind_speed: Average wind speed
                - weather_stability: Track condition stability
                - safety_car_probability: Adjusted SC probability
        """
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
        """
        Analyze tire performance and degradation.
        
        This method calculates:
        1. Compound-specific degradation rates
        2. Average and fastest lap times
        3. Tire life expectations
        4. Stint performance patterns
        
        The analysis considers:
        - Track-specific wear characteristics
        - Temperature impact on compounds
        - Historical performance data
        - Real-time degradation patterns
        
        Returns:
            Dict: Tire performance data by compound:
                - avg_lap_time: Mean lap time on compound
                - fastest_lap: Best lap time achieved
                - avg_tire_life: Expected tire life
                - degradation: Wear rate per lap
                - stint_count: Number of stints
                - total_laps: Laps completed
        """
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
        """
        Analyze individual tire stints.
        
        This method calculates:
        1. Stint duration
        2. Average lap time
        3. Tire degradation
        4. Tire life
        
        Returns:
            List[TireStintData]: List of analyzed stints
        """
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
        """
        Calculate track evolution rate.
        
        This method calculates:
        1. Lap time improvement rate
        2. Grip improvement rate
        
        Returns:
            float: Track evolution rate
        """
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
