"""
F1 Pit Stop Strategy Optimizer

This module provides a comprehensive pit stop strategy optimization system for Formula 1 races.
It uses track characteristics, tire data, and real-time race information to calculate optimal
pit stop windows and tire compound recommendations.

Key Components:
- TrackCharacteristics: Manages track-specific parameters and conditions
- StopWindow: Defines pit stop timing and compound recommendations
- TrackStrategyOptimizer: Core strategy calculation engine
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from src.models.fastf1_loader import FastF1DataLoader

@dataclass
class TrackCharacteristics:
    """
    Track characteristics and conditions that influence pit stop strategy.
    
    Attributes:
        tire_degradation (float): Base tire wear rate (1.0 = normal, >1.0 = higher wear)
        track_evolution (float): How quickly track grip improves (0.003-0.017 per lap)
        safety_car_probability (float): Likelihood of safety car appearance (0.2-0.65)
        traffic_impact (float): Effect of traffic on lap times (0.4-0.8)
        pit_window_margin (int): Flexible laps around optimal pit window
        overtaking_difficulty (float): How hard it is to pass (0.4-0.8)
        track_length (float): Circuit length in kilometers
        track_type (str): Circuit category (street/technical/high-speed/standard)
        pit_loss_time (float): Time lost for pit stop in seconds
        race_laps (int): Total number of race laps
        max_stops (int): Maximum recommended pit stops
    """
    tire_degradation: float = 1.0
    track_evolution: float = 0.005
    safety_car_probability: float = 0.3
    traffic_impact: float = 0.5
    pit_window_margin: int = 3
    overtaking_difficulty: float = 0.5
    track_length: float = 5.0
    track_type: str = 'standard'
    pit_loss_time: float = 20.0
    race_laps: int = 0
    max_stops: int = 3
    
    def __post_init__(self):
        """Initialize available tire compounds after instance creation."""
        self.tire_compounds = ['soft', 'medium', 'hard']
    
    @property
    def high_degradation(self) -> bool:
        """Check if track has high tire degradation (>1.1x base rate)."""
        return self.tire_degradation > 1.1

    @property
    def high_evolution(self) -> bool:
        """Check if track has significant grip improvement (>0.008 per lap)."""
        return self.track_evolution > 0.008

    @property
    def high_safety_car(self) -> bool:
        """Check if track has high safety car risk (>35% probability)."""
        return self.safety_car_probability > 0.35

    @property
    def difficult_overtaking(self) -> bool:
        """Check if track has limited overtaking opportunities (>0.7 difficulty)."""
        return self.overtaking_difficulty > 0.7

@dataclass
class StopWindow:
    """
    Defines a pit stop window with timing and compound recommendations.
    
    A pit stop window represents a range of laps where pitting would be optimal,
    considering track position, tire wear, and race conditions.
    
    Attributes:
        start_lap (int): Earliest recommended lap to pit
        optimal_lap (int): Ideal lap for pit stop
        end_lap (int): Latest recommended lap to pit
        compound (str): Recommended tire compound for next stint
    """
    start_lap: int = field(metadata={"description": "Start lap of the pit window"})
    optimal_lap: int = field(metadata={"description": "Optimal lap to pit"})
    end_lap: int = field(metadata={"description": "End lap of the pit window"})
    compound: str = field(metadata={"description": "Recommended tire compound"})

class TrackStrategyOptimizer:
    """
    Core strategy optimization engine for F1 pit stops.
    
    This class handles:
    1. Track configuration management
    2. Real-time data integration via FastF1
    3. Pit stop window calculations
    4. Tire compound recommendations
    5. Strategy point generation
    
    The optimizer considers:
    - Track characteristics (type, length, degradation)
    - Race situation (position, lap, tire age)
    - Historical data (previous races, weather patterns)
    """
    
    # Track configurations with pre-defined characteristics
    TRACK_CONFIGS = {
        'australian_gp': {
            'track_length': 5.278,
            'track_type': 'standard',
            'race_laps': 58,
            'pit_loss_time': 21.5,
            'tire_compounds': ['soft', 'medium', 'hard'],
            'max_pit_stops': 2,
            'difficult_overtaking': False,
            'tire_degradation': 1.0,
            'track_evolution': 0.013,
            'safety_car_probability': 0.3,
            'traffic_impact': 0.5,
            'pit_window_margin': 4
        },
        'chinese_gp': {
            'track_length': 5.451,
            'track_type': 'technical',
            'race_laps': 56,
            'pit_loss_time': 21.5,
            'tire_compounds': ['soft', 'medium', 'hard'],
            'max_pit_stops': 3,
            'difficult_overtaking': False,
            'tire_degradation': 1.28,
            'track_evolution': 0.013,
            'safety_car_probability': 0.35,
            'traffic_impact': 0.5,
            'pit_window_margin': 4
        },
        'japanese_gp': {
            'track_length': 5.807,
            'track_type': 'technical',
            'race_laps': 53,
            'pit_loss_time': 21.5,
            'tire_compounds': ['soft', 'medium', 'hard'],
            'max_pit_stops': 2,
            'difficult_overtaking': True,
            'tire_degradation': 1.2,
            'track_evolution': 0.014,
            'safety_car_probability': 0.4,
            'traffic_impact': 0.6,
            'pit_window_margin': 4
        },
        'bahrain_gp': {
            'track_length': 5.412,
            'track_type': 'standard',
            'race_laps': 57,
            'pit_loss_time': 21.0,
            'tire_compounds': ['soft', 'medium', 'hard'],
            'max_pit_stops': 3,
            'difficult_overtaking': False,
            'tire_degradation': 1.25,
            'track_evolution': 0.015,
            'safety_car_probability': 0.25,
            'traffic_impact': 0.4,
            'pit_window_margin': 4
        },
        'saudi_arabia_gp': {
            'track_length': 6.174,
            'track_type': 'street',
            'race_laps': 50,
            'pit_loss_time': 22.5,
            'tire_compounds': ['soft', 'medium', 'hard'],
            'max_pit_stops': 2,
            'difficult_overtaking': True,
            'tire_degradation': 1.15,
            'track_evolution': 0.016,
            'safety_car_probability': 0.65,
            'traffic_impact': 0.6,
            'pit_window_margin': 4
        },
        'miami_gp': {
            'track_length': 5.412,
            'track_type': 'standard',
            'race_laps': 57,
            'pit_loss_time': 21.5,
            'tire_compounds': ['soft', 'medium', 'hard'],
            'max_pit_stops': 2,
            'difficult_overtaking': False,
            'tire_degradation': 1.18,
            'track_evolution': 0.017,
            'safety_car_probability': 0.45,
            'traffic_impact': 0.5,
            'pit_window_margin': 4
        },
        'emilia_romagna_gp': {
            'track_length': 4.909,
            'track_type': 'technical',
            'race_laps': 63,
            'pit_loss_time': 22.0,
            'tire_compounds': ['soft', 'medium', 'hard'],
            'max_pit_stops': 2,
            'difficult_overtaking': True,
            'tire_degradation': 1.2,
            'track_evolution': 0.012,
            'safety_car_probability': 0.5,
            'traffic_impact': 0.7,
            'pit_window_margin': 4
        },
        'monaco_gp': {
            'track_length': 3.337,
            'track_type': 'street',
            'race_laps': 78,
            'pit_loss_time': 23.5,
            'tire_compounds': ['soft', 'medium', 'hard'],
            'max_pit_stops': 1,
            'difficult_overtaking': True,
            'tire_degradation': 0.8,
            'track_evolution': 0.003,
            'safety_car_probability': 0.6,
            'traffic_impact': 0.8,
            'pit_window_margin': 2
        },
        'spanish_gp': {
            'track_length': 4.675,
            'track_type': 'technical',
            'race_laps': 66,
            'pit_loss_time': 21.5,
            'tire_compounds': ['soft', 'medium', 'hard'],
            'max_pit_stops': 3,
            'difficult_overtaking': True,
            'tire_degradation': 1.22,
            'track_evolution': 0.012,
            'safety_car_probability': 0.2,
            'traffic_impact': 0.7,
            'pit_window_margin': 4
        },
        'canadian_gp': {
            'track_length': 4.361,
            'track_type': 'street',
            'race_laps': 70,
            'pit_loss_time': 22.0,
            'tire_compounds': ['soft', 'medium', 'hard'],
            'max_pit_stops': 2,
            'difficult_overtaking': False,
            'tire_degradation': 1.2,
            'track_evolution': 0.014,
            'safety_car_probability': 0.5,
            'traffic_impact': 0.6,
            'pit_window_margin': 4
        },
        'austrian_gp': {
            'track_length': 4.318,
            'track_type': 'standard',
            'race_laps': 71,
            'pit_loss_time': 20.5,
            'tire_compounds': ['soft', 'medium', 'hard'],
            'max_pit_stops': 2,
            'difficult_overtaking': False,
            'tire_degradation': 1.15,
            'track_evolution': 0.011,
            'safety_car_probability': 0.25,
            'traffic_impact': 0.3,
            'pit_window_margin': 4
        },
        'british_gp': {
            'track_length': 5.891,
            'track_type': 'high_speed',
            'race_laps': 52,
            'pit_loss_time': 22.0,
            'tire_compounds': ['soft', 'medium', 'hard'],
            'max_pit_stops': 2,
            'difficult_overtaking': False,
            'tire_degradation': 1.25,
            'track_evolution': 0.015,
            'safety_car_probability': 0.35,
            'traffic_impact': 0.4,
            'pit_window_margin': 5
        },
        'belgian_gp': {
            'track_length': 7.004,
            'track_type': 'high_speed',
            'race_laps': 44,
            'pit_loss_time': 18.5,
            'tire_compounds': ['soft', 'medium', 'hard'],
            'max_pit_stops': 3,
            'difficult_overtaking': False,
            'tire_degradation': 1.2,
            'track_evolution': 0.012,
            'safety_car_probability': 0.4,
            'traffic_impact': 0.4,
            'pit_window_margin': 5
        },
        'hungarian_gp': {
            'track_length': 4.381,
            'track_type': 'technical',
            'race_laps': 70,
            'pit_loss_time': 22.0,
            'tire_compounds': ['soft', 'medium', 'hard'],
            'max_pit_stops': 2,
            'difficult_overtaking': True,
            'tire_degradation': 1.2,
            'track_evolution': 0.015,
            'safety_car_probability': 0.3,
            'traffic_impact': 0.8,
            'pit_window_margin': 4
        },
        'dutch_gp': {
            'track_length': 4.259,
            'track_type': 'technical',
            'race_laps': 72,
            'pit_loss_time': 22.0,
            'tire_compounds': ['soft', 'medium', 'hard'],
            'max_pit_stops': 2,
            'difficult_overtaking': True,
            'tire_degradation': 1.25,
            'track_evolution': 0.013,
            'safety_car_probability': 0.4,
            'traffic_impact': 0.7,
            'pit_window_margin': 4
        },
        'italian_gp': {
            'track_length': 5.793,
            'track_type': 'high_speed',
            'race_laps': 53,
            'pit_loss_time': 21.0,
            'tire_compounds': ['soft', 'medium', 'hard'],
            'max_pit_stops': 2,
            'difficult_overtaking': False,
            'tire_degradation': 1.3,
            'track_evolution': 0.008,
            'safety_car_probability': 0.3,
            'traffic_impact': 0.3,
            'pit_window_margin': 5
        },
        'azerbaijan_gp': {
            'track_length': 6.003,
            'track_type': 'street',
            'race_laps': 51,
            'pit_loss_time': 22.5,
            'tire_compounds': ['soft', 'medium', 'hard'],
            'max_pit_stops': 2,
            'difficult_overtaking': False,
            'tire_degradation': 1.0,
            'track_evolution': 0.017,
            'safety_car_probability': 0.7,
            'traffic_impact': 0.75,
            'pit_window_margin': 3
        },
        'singapore_gp': {
            'track_length': 4.940,
            'track_type': 'street',
            'race_laps': 61,
            'pit_loss_time': 23.0,
            'tire_compounds': ['soft', 'medium', 'hard'],
            'max_pit_stops': 2,
            'difficult_overtaking': True,
            'tire_degradation': 1.1,
            'track_evolution': 0.018,
            'safety_car_probability': 0.8,
            'traffic_impact': 0.9,
            'pit_window_margin': 3
        },
        'united_states_gp': {
            'track_length': 5.513,
            'track_type': 'standard',
            'race_laps': 56,
            'pit_loss_time': 21.5,
            'tire_compounds': ['soft', 'medium', 'hard'],
            'max_pit_stops': 2,
            'difficult_overtaking': False,
            'tire_degradation': 1.15,
            'track_evolution': 0.014,
            'safety_car_probability': 0.35,
            'traffic_impact': 0.5,
            'pit_window_margin': 4
        },
        'mexican_gp': {
            'track_length': 4.304,
            'track_type': 'standard',
            'race_laps': 71,
            'pit_loss_time': 22.0,
            'tire_compounds': ['soft', 'medium', 'hard'],
            'max_pit_stops': 2,
            'difficult_overtaking': False,
            'tire_degradation': 1.1,
            'track_evolution': 0.012,
            'safety_car_probability': 0.3,
            'traffic_impact': 0.6,
            'pit_window_margin': 4
        },
        'brazilian_gp': {
            'track_length': 4.309,
            'track_type': 'technical',
            'race_laps': 71,
            'pit_loss_time': 21.5,
            'tire_compounds': ['soft', 'medium', 'hard'],
            'max_pit_stops': 2,
            'difficult_overtaking': False,
            'tire_degradation': 1.18,
            'track_evolution': 0.016,
            'safety_car_probability': 0.45,
            'traffic_impact': 0.5,
            'pit_window_margin': 4
        },
        'las_vegas_gp': {
            'track_length': 6.201,
            'track_type': 'street',
            'race_laps': 50,
            'pit_loss_time': 22.0,
            'tire_compounds': ['soft', 'medium', 'hard'],
            'max_pit_stops': 2,
            'difficult_overtaking': False,
            'tire_degradation': 1.1,
            'track_evolution': 0.02,
            'safety_car_probability': 0.55,
            'traffic_impact': 0.4,
            'pit_window_margin': 4
        },
        'qatar_gp': {
            'track_length': 5.419,
            'track_type': 'high_speed',
            'race_laps': 57,
            'pit_loss_time': 21.5,
            'tire_compounds': ['soft', 'medium', 'hard'],
            'max_pit_stops': 3,
            'difficult_overtaking': False,
            'tire_degradation': 1.35,
            'track_evolution': 0.015,
            'safety_car_probability': 0.3,
            'traffic_impact': 0.5,
            'pit_window_margin': 4
        },
        'abu_dhabi_gp': {
            'track_length': 5.281,
            'track_type': 'technical',
            'race_laps': 55,
            'pit_loss_time': 22.0,
            'tire_compounds': ['soft', 'medium', 'hard'],
            'max_pit_stops': 2,
            'difficult_overtaking': False,
            'tire_degradation': 1.05,
            'track_evolution': 0.01,
            'safety_car_probability': 0.25,
            'traffic_impact': 0.6,
            'pit_window_margin': 4
        }
    }
    
    TRACK_NAMES = {
        'australian_gp': 'Australian Grand Prix',
        'chinese_gp': 'Chinese Grand Prix',
        'japanese_gp': 'Japanese Grand Prix',
        'bahrain_gp': 'Bahrain Grand Prix',
        'saudi_arabia_gp': 'Saudi Arabian Grand Prix',
        'miami_gp': 'Miami Grand Prix',
        'emilia_romagna_gp': 'Emilia Romagna Grand Prix',
        'monaco_gp': 'Monaco Grand Prix',
        'spanish_gp': 'Spanish Grand Prix',
        'canadian_gp': 'Canadian Grand Prix',
        'austrian_gp': 'Austrian Grand Prix',
        'british_gp': 'British Grand Prix',
        'belgian_gp': 'Belgian Grand Prix',
        'hungarian_gp': 'Hungarian Grand Prix',
        'dutch_gp': 'Dutch Grand Prix',
        'italian_gp': 'Italian Grand Prix',
        'azerbaijan_gp': 'Azerbaijan Grand Prix',
        'singapore_gp': 'Singapore Grand Prix',
        'united_states_gp': 'United States Grand Prix',
        'mexican_gp': 'Mexican Grand Prix',
        'brazilian_gp': 'Brazilian Grand Prix',
        'las_vegas_gp': 'Las Vegas Grand Prix',
        'qatar_gp': 'Qatar Grand Prix',
        'abu_dhabi_gp': 'Abu Dhabi Grand Prix'
    }
    
    # Tire compound characteristics and performance data
    TIRE_COMPOUNDS = {
        'soft': {
            'min_life': 10,  # Minimum expected laps before significant degradation
            'max_life': 20,  # Maximum usable life in optimal conditions
            'pace_advantage': -0.5,  # Pace delta vs medium compound (seconds)
            'optimal_window': (1, 15),  # Best performance window (start_lap, end_lap)
            'high_deg_tracks': ['singapore_gp', 'spanish_gp'],  # Tracks with high wear
            'preferred_phase': 'early'  # Best race phase for this compound
        },
        'medium': {
            'min_life': 20,
            'max_life': 35,
            'pace_advantage': 0.0,
            'optimal_window': (5, 25),  # Best performance between laps 5-25
            'high_deg_tracks': ['silverstone', 'suzuka'],
            'preferred_phase': 'middle'
        },
        'hard': {
            'min_life': 30,
            'max_life': 45,
            'pace_advantage': 0.0,
            'optimal_window': (10, 40),  # Best performance between laps 10-40
            'high_deg_tracks': ['spain', 'france'],
            'preferred_phase': 'start'
        },
        'inter': {
            'min_life': 15,
            'max_life': 30,
            'pace_advantage': 0.0,
            'optimal_window': (0, 30),
            'high_deg_tracks': [],
            'preferred_phase': 'any'
        },
        'wet': {
            'min_life': 20,
            'max_life': 40,
            'pace_advantage': 0.0,
            'optimal_window': (0, 40),
            'high_deg_tracks': [],
            'preferred_phase': 'any'
        }
    }
    
    # Special weather-specific tire compounds
    WEATHER_COMPOUNDS = {
        'intermediate': {
            'color': 'Green',
            'min_life': 25,
            'max_life': 40,
            'characteristics': 'For light rain and drying tracks'
        },
        'wet': {
            'color': 'Blue',
            'min_life': 15,
            'max_life': 25,
            'characteristics': 'For heavy rain conditions'
        }
    }
    
    def __init__(self, track_name: str, year: int):
        """
        Initialize the pit stop strategy optimizer.
        
        Args:
            track_name (str): Name of the F1 track (e.g., 'monaco_gp')
            year (int): Season year for historical data reference
        
        The initializer:
        1. Loads track configuration from predefined settings
        2. Attempts to fetch live session data via FastF1
        3. Updates track characteristics based on real-time data
        """
        self.track_name = track_name.lower()
        self.year = year
        self.track_characteristics = TrackCharacteristics()
        self.fastf1_loader = FastF1DataLoader()
        self.load_track_config()
        
        # Try to load live data
        try:
            self.session_data = self.fastf1_loader.load_session(
                year=self.year,
                track_name=self.track_name
            )
            self.update_characteristics(
                self.session_data.get('tire_performance', {}),
                self.session_data.get('weather', {})
            )
        except Exception as e:
            print(f"\nNote: Using default track data (no live data available: {e})")
    
    def load_track_config(self):
        """
        Load and set track-specific configuration parameters.
        
        This method:
        1. Retrieves track config from TRACK_CONFIGS dictionary
        2. Sets basic track parameters (length, type, pit loss time)
        3. Sets race parameters (laps, compounds, max stops)
        4. Sets behavioral characteristics (degradation, evolution, safety car)
        """
        config = self.TRACK_CONFIGS[self.track_name]
        
        # Basic track characteristics
        self.track_characteristics.track_length = config['track_length']
        self.track_characteristics.track_type = config['track_type']
        self.track_characteristics.pit_loss_time = config['pit_loss_time']
        self.track_characteristics.tire_compounds = config['tire_compounds']
        self.track_characteristics.race_laps = config['race_laps']
        self.track_characteristics.max_stops = config['max_pit_stops']
        
        # Track behavior characteristics
        self.track_characteristics.tire_degradation = config['tire_degradation']
        self.track_characteristics.track_evolution = config['track_evolution']
        self.track_characteristics.safety_car_probability = config['safety_car_probability']
        self.track_characteristics.traffic_impact = config['traffic_impact']
        self.track_characteristics.pit_window_margin = config['pit_window_margin']
        self.track_characteristics.overtaking_difficulty = 0.8 if config['difficult_overtaking'] else 0.4
    
    def update_characteristics(self, tire_data: Dict, weather_data: Dict):
        """
        Update track characteristics using real-time session data.
        
        Args:
            tire_data (Dict): Real-time tire performance metrics
                Keys: compound names
                Values: degradation, avg life, fastest lap
            weather_data (Dict): Current weather conditions
                Keys: rainfall, track_temp, weather_stability
        
        This method dynamically adjusts:
        1. Tire degradation based on real performance
        2. Safety car probability based on conditions
        3. Track evolution rate
        4. Traffic impact factors
        """
        if tire_data:
            # Update tire degradation based on real data
            compounds_deg = [
                data['degradation']
                for data in tire_data.values()
                if 'degradation' in data
            ]
            if compounds_deg:
                avg_degradation = sum(compounds_deg) / len(compounds_deg)
                # Blend real data with base data (70% real, 30% base)
                self.track_characteristics.tire_degradation = (
                    0.7 * avg_degradation +
                    0.3 * self.track_characteristics.tire_degradation
                )
            
            # Update compound-specific data
            for compound, data in tire_data.items():
                if compound in self.TIRE_COMPOUNDS:
                    self.TIRE_COMPOUNDS[compound].update({
                        'min_life': max(5, data.get('avg_tire_life', 15) * 0.7),
                        'max_life': data.get('avg_tire_life', 30) * 1.2,
                        'pace_advantage': data.get('fastest_lap', 0) - min(
                            d.get('fastest_lap', float('inf'))
                            for d in tire_data.values()
                            if 'fastest_lap' in d
                        )
                    })
        
        if weather_data:
            # Update safety car probability based on conditions
            base_probability = self.track_characteristics.safety_car_probability
            
            # Weather impact factors
            if weather_data.get('rainfall', False):
                base_probability *= 1.5  # 50% higher in wet conditions
            if weather_data.get('track_temp', 25) > 40:
                base_probability *= 1.2  # 20% higher in very hot conditions
            
            # Track type impact factors
            if self.track_characteristics.track_type == 'street':
                base_probability *= 1.3  # 30% higher on street circuits
            elif self.track_characteristics.track_type == 'high_speed':
                base_probability *= 1.1  # 10% higher on high-speed circuits
            
            self.track_characteristics.safety_car_probability = min(base_probability, 1.0)
            
            # Update track evolution based on weather stability
            if 'weather_stability' in weather_data:
                self.track_characteristics.track_evolution *= weather_data['weather_stability']
            
            # Update traffic impact based on weather conditions
            if weather_data.get('rainfall', False):
                self.track_characteristics.traffic_impact *= 1.2  # 20% higher in wet conditions
    
    @staticmethod
    def get_track_details():
        """
        Interactive CLI to display track information and generate strategies.
        
        Features:
        1. Lists all available F1 tracks with key details
        2. Supports multiple track name formats (full, short, common names)
        3. Provides track-specific characteristics
        4. Generates customized pit stop strategies
        
        Track Types:
        - standard: Balanced characteristics
        - technical: High downforce, slower corners
        - high_speed: Long straights, fast corners
        - street: Narrow, walls, high risk
        """
        while True:
            print("\nAvailable F1 Tracks:")
            print("-" * 65)
            print(f"{'Track':<35} {'Laps':<6} {'Length':<8} {'Type'}")
            print("-" * 65)
            
            # Define track order
            track_order = [
                'australian_gp', 'chinese_gp', 'japanese_gp', 'bahrain_gp',
                'saudi_arabia_gp', 'miami_gp', 'emilia_romagna_gp', 'monaco_gp',
                'spanish_gp', 'canadian_gp', 'austrian_gp', 'british_gp',
                'belgian_gp', 'hungarian_gp', 'dutch_gp', 'italian_gp',
                'azerbaijan_gp', 'singapore_gp', 'united_states_gp', 'mexican_gp',
                'brazilian_gp', 'las_vegas_gp', 'qatar_gp', 'abu_dhabi_gp'
            ]
            
            # Track name aliases
            TRACK_ALIASES = {
                'australian': 'australian_gp',
                'australia': 'australian_gp',
                'chinese': 'chinese_gp',
                'china': 'chinese_gp',
                'japanese': 'japanese_gp',
                'japan': 'japanese_gp',
                'bahrain': 'bahrain_gp',
                'saudi': 'saudi_arabia_gp',
                'saudi arabia': 'saudi_arabia_gp',
                'miami': 'miami_gp',
                'emilia': 'emilia_romagna_gp',
                'imola': 'emilia_romagna_gp',
                'monaco': 'monaco_gp',
                'monte carlo': 'monaco_gp',
                'spanish': 'spanish_gp',
                'spain': 'spanish_gp',
                'canadian': 'canadian_gp',
                'canada': 'canadian_gp',
                'austrian': 'austrian_gp',
                'austria': 'austrian_gp',
                'british': 'british_gp',
                'britain': 'british_gp',
                'silverstone': 'british_gp',
                'belgian': 'belgian_gp',
                'belgium': 'belgian_gp',
                'spa': 'belgian_gp',
                'hungarian': 'hungarian_gp',
                'hungary': 'hungarian_gp',
                'dutch': 'dutch_gp',
                'netherlands': 'dutch_gp',
                'zandvoort': 'dutch_gp',
                'italian': 'italian_gp',
                'italy': 'italian_gp',
                'monza': 'italian_gp',
                'azerbaijan': 'azerbaijan_gp',
                'baku': 'azerbaijan_gp',
                'singapore': 'singapore_gp',
                'marina bay': 'singapore_gp',
                'united states': 'united_states_gp',
                'usa': 'united_states_gp',
                'cota': 'united_states_gp',
                'mexican': 'mexican_gp',
                'mexico': 'mexican_gp',
                'brazilian': 'brazilian_gp',
                'brazil': 'brazilian_gp',
                'interlagos': 'brazilian_gp',
                'las vegas': 'las_vegas_gp',
                'qatar': 'qatar_gp',
                'losail': 'qatar_gp',
                'abu dhabi': 'abu_dhabi_gp',
                'yas marina': 'abu_dhabi_gp'
            }
            
            # Display track info in order
            for track_id in track_order:
                config = TrackStrategyOptimizer.TRACK_CONFIGS[track_id]
                display_name = TrackStrategyOptimizer.TRACK_NAMES[track_id]
                print(f"{display_name:<35} {config['race_laps']:<6} {config['track_length']:<8.3f} {config['track_type'].replace('_', ' ').title()}")
            
            print("\nEnter track name (or 'exit' to quit): ", end='')
            track_input = input().strip().lower()
            
            if track_input in ['exit', 'quit']:
                print("\nGoodbye!")
                break
            
            # Check track aliases first
            track_key = TRACK_ALIASES.get(track_input)
            
            # If no alias found, try matching with track names
            if not track_key:
                for key, name in TrackStrategyOptimizer.TRACK_NAMES.items():
                    # Match full track name or key (e.g. "monaco grand prix" or "monaco_gp")
                    if track_input == name.lower() or track_input == key.replace('_', ' '):
                        track_key = key
                        break
            
            if track_key is None:
                print(f"\nError: Track not found. Please enter a valid track name.")
                continue
            
            optimizer = TrackStrategyOptimizer(track_key, 2024)
            
            print(f"\n{TrackStrategyOptimizer.TRACK_NAMES[track_key]} Configuration:")
            print("-" * 40)
            print(f"Track Length: {optimizer.track_characteristics.track_length:.3f} km")
            print(f"Track Type: {optimizer.track_characteristics.track_type.replace('_', ' ').title()}")
            print(f"Maximum Pit Stops: {optimizer.track_characteristics.max_stops}")
            print(f"Pit Loss Time: {optimizer.track_characteristics.pit_loss_time:.1f} seconds")
            
            print("\nTrack Characteristics:")
            print(f"- Tire Degradation: {optimizer.track_characteristics.tire_degradation:.2f}x")
            print(f"- Track Evolution: {optimizer.track_characteristics.track_evolution:.3f}")
            print(f"- Safety Car Probability: {optimizer.track_characteristics.safety_car_probability*100:.2f}%")
            print(f"- Traffic Impact: {optimizer.track_characteristics.traffic_impact:.2f}")
            print(f"- Overtaking Difficulty: {optimizer.track_characteristics.overtaking_difficulty:.2f}")
            print(f"- Pit Window Margin: {optimizer.track_characteristics.pit_window_margin} laps")
            
            print("\nEnter Race Details:")
            print("-" * 20)
            try:
                current_position = int(input("Current position: "))
                current_lap = int(input("Current lap: "))
                
                # Get current tire info
                print("\nCurrent Tire Details:")
                print("-" * 20)
                current_compound = input("Current tire compound (soft/medium/hard/inter/wet): ").lower()
                if current_compound not in ['soft', 'medium', 'hard', 'inter', 'wet']:
                    raise ValueError("Invalid tire compound")
                
                tire_age = int(input("Current tire age (laps): "))
                if tire_age < 0:
                    raise ValueError("Tire age must be non-negative")
                
                # Get previous pit stop info
                print("\nPrevious Pit Stops:")
                print("-" * 20)
                num_stops = int(input("Number of pit stops made (0-3): "))
                if num_stops < 0 or num_stops > 3:
                    raise ValueError("Number of stops must be between 0 and 3")
                
                previous_stops = []
                for i in range(num_stops):
                    stop_lap = int(input(f"Lap number of pit stop {i+1}: "))
                    if stop_lap < 1 or stop_lap >= current_lap:
                        raise ValueError(f"Invalid lap number for stop {i+1}")
                    previous_stops.append(stop_lap)
                
                if current_position <= 0 or current_lap < 0:
                    raise ValueError("Values must be positive")
                if current_position > 20:
                    raise ValueError("Position must be between 1 and 20")
                
                # Generate strategy with tire and pit stop history
                strategy = optimizer.generate_strategy(
                    current_lap=current_lap,
                    current_position=current_position,
                    current_compound=current_compound,
                    tire_age=tire_age,
                    previous_stops=previous_stops
                )
                
                optimizer.display_strategy(strategy)
                
            except ValueError as e:
                print(f"\nError: {e}")
                continue
            
    def generate_strategy(
        self,
        current_lap: int,
        current_position: int,
        current_compound: str = None,
        tire_age: int = 0,
        previous_stops: List[int] = None
    ) -> Dict:
        """
        Generate an optimal pit stop strategy based on current race conditions.
        
        This method analyzes multiple factors to determine the best strategy:
        1. Track characteristics (degradation, evolution, traffic)
        2. Current race situation (position, lap, tire condition)
        3. Historical pit stop data
        4. Weather and track conditions
        
        Args:
            current_lap (int): Current lap number in the race
            current_position (int): Current position (1-20)
            current_compound (str, optional): Current tire compound (soft/medium/hard/inter/wet)
            tire_age (int, optional): Age of current tires in laps
            previous_stops (List[int], optional): Lap numbers of previous pit stops
        
        Returns:
            Dict: Strategy recommendation containing:
                - recommended_stops: Number of remaining pit stops
                - pit_windows: List of StopWindow objects
                - strategy_notes: List of important strategy considerations
        """
        if previous_stops is None:
            previous_stops = []
        
        remaining_laps = self.track_characteristics.race_laps - current_lap
        stops_made = len(previous_stops)
        max_remaining_stops = self.track_characteristics.max_stops - stops_made
        
        # Get tire data and update with current state
        compounds_data = self.TIRE_COMPOUNDS.copy()
        if current_compound:
            # Update tire life expectancy based on current age and wear
            compound_data = compounds_data[current_compound]
            remaining_life = compound_data['max_life'] - tire_age
            if remaining_life < 0:
                remaining_life = 0
            compounds_data[current_compound]['remaining_life'] = remaining_life
        
        # Early return if race is finished
        if remaining_laps <= 0:
            return {
                'recommended_stops': 0,
                'pit_windows': [],
                'strategy_notes': ['Race is finished']
            }
        
        # Calculate optimal strategy considering:
        # 1. Remaining race distance
        # 2. Current tire condition and compound
        # 3. Track-specific characteristics
        # 4. Historical pit stop patterns
        base_stops = self._calculate_optimal_stops(
            remaining_laps=remaining_laps,
            current_compound=current_compound,
            tire_age=tire_age,
            track_evolution=self.track_characteristics.track_evolution,
            safety_car_prob=self.track_characteristics.safety_car_probability
        )
        
        # Adjust strategy based on race position and traffic
        adjusted_stops = self._adjust_strategy_for_position(
            base_stops=base_stops,
            current_position=current_position,
            traffic_impact=self.track_characteristics.traffic_impact,
            overtaking_difficulty=self.track_characteristics.overtaking_difficulty
        )
        
        # Generate pit windows for each remaining stop
        pit_windows = self._generate_pit_windows(
            remaining_laps=remaining_laps,
            num_stops=adjusted_stops,
            current_compound=current_compound,
            tire_age=tire_age
        )
        
        # Add strategy notes based on specific conditions
        strategy_notes = self._generate_strategy_notes(
            pit_windows=pit_windows,
            current_position=current_position,
            safety_car_prob=self.track_characteristics.safety_car_probability,
            track_evolution=self.track_characteristics.track_evolution
        )
        
        return {
            'recommended_stops': adjusted_stops,
            'pit_windows': pit_windows,
            'strategy_notes': strategy_notes
        }
    
    def _calculate_optimal_stops(
        self,
        remaining_laps: int,
        current_compound: str,
        tire_age: int,
        track_evolution: float,
        safety_car_prob: float
    ) -> int:
        """
        Calculate the optimal number of remaining pit stops.
        
        This method uses a weighted scoring system considering:
        1. Tire life vs remaining laps
        2. Track evolution impact on lap times
        3. Safety car probability
        4. Compound-specific characteristics
        
        Args:
            remaining_laps (int): Number of laps left in race
            current_compound (str): Current tire compound
            tire_age (int): Age of current tires
            track_evolution (float): Rate of track improvement
            safety_car_prob (float): Probability of safety car
            
        Returns:
            int: Recommended number of stops
        """
        # Base calculation from remaining laps
        base_stops = remaining_laps // 20  # Rough estimate: one stop every ~20 laps
        
        # Adjust based on current tire condition
        if current_compound and tire_age:
            compound_data = self.TIRE_COMPOUNDS[current_compound]
            remaining_life = compound_data['max_life'] - tire_age
            
            # If tires are near end of life, add a stop
            if remaining_life < remaining_laps * 0.7:  # Need 70% life for remaining laps
                base_stops += 1
        
        # Adjust based on track characteristics
        if track_evolution > 0.012:  # High evolution
            base_stops += 1
        elif track_evolution < 0.008:  # Low evolution
            base_stops = max(0, base_stops - 1)
        
        # Adjust based on safety car probability
        if safety_car_prob > 0.4:  # High safety car risk
            base_stops += 1
        elif safety_car_prob < 0.2:  # Low safety car risk
            base_stops = max(0, base_stops - 1)
        
        return base_stops
    
    def _adjust_strategy_for_position(
        self,
        base_stops: int,
        current_position: int,
        traffic_impact: float,
        overtaking_difficulty: float
    ) -> int:
        """
        Adjust the strategy based on the current race position.
        
        This method considers:
        1. Traffic impact on lap times
        2. Overtaking difficulty
        3. Current position
        
        Args:
            base_stops (int): Base number of stops
            current_position (int): Current position (1-20)
            traffic_impact (float): Effect of traffic on lap times
            overtaking_difficulty (float): How hard it is to pass
            
        Returns:
            int: Adjusted number of stops
        """
        # Adjust based on traffic impact
        if traffic_impact > 0.6:  # High traffic impact
            base_stops += 1
        elif traffic_impact < 0.4:  # Low traffic impact
            base_stops = max(0, base_stops - 1)
        
        # Adjust based on overtaking difficulty
        if overtaking_difficulty > 0.7:  # High overtaking difficulty
            base_stops += 1
        elif overtaking_difficulty < 0.5:  # Low overtaking difficulty
            base_stops = max(0, base_stops - 1)
        
        # Adjust based on current position
        if current_position > 10:  # Lower positions
            base_stops += 1
        elif current_position < 5:  # Higher positions
            base_stops = max(0, base_stops - 1)
        
        return base_stops
    
    def _generate_pit_windows(
        self,
        remaining_laps: int,
        num_stops: int,
        current_compound: str,
        tire_age: int
    ) -> List[Dict]:
        """
        Generate pit windows for each remaining stop.
        
        This method considers:
        1. Remaining laps
        2. Number of stops
        3. Current tire compound and age
        
        Args:
            remaining_laps (int): Number of laps left in race
            num_stops (int): Number of remaining stops
            current_compound (str): Current tire compound
            tire_age (int): Age of current tires
            
        Returns:
            List[Dict]: List of pit windows
        """
        pit_windows = []
        
        # Simple stint calculation - divide remaining laps evenly
        stint_length = remaining_laps // (num_stops + 1)
        
        for stop in range(num_stops):
            # Calculate optimal lap - never pit in last 2 laps
            optimal_lap = min(
                self.track_characteristics.race_laps - 2,
                stint_length * (stop + 1)
            )
            
            # Simple window margin based on track type
            margin = 3 if self.track_characteristics.track_type == 'street' else 5
            
            window = {
                'stop_number': stop + 1,
                'window': {
                    'start': max(stint_length * stop, optimal_lap - margin),
                    'optimal': optimal_lap,
                    'end': min(
                        self.track_characteristics.race_laps - 2,
                        optimal_lap + margin
                    )
                },
                'compound': self._recommend_compound(stop, num_stops, remaining_laps)
            }
            
            pit_windows.append(window)
        
        return pit_windows
    
    def _recommend_compound(
        self,
        stop: int,
        total_stops: int,
        remaining_laps: int
    ) -> str:
        """
        Recommend tire compound for the next stop.
        
        This method considers:
        1. Remaining laps
        2. Number of stops
        3. Current stop number
        
        Args:
            stop (int): Current stop number
            total_stops (int): Total number of stops
            remaining_laps (int): Number of laps left in race
            
        Returns:
            str: Recommended tire compound
        """
        # For last stint
        if stop == total_stops - 1:
            # Use softs for short final stints or street circuits
            if remaining_laps <= 15 or self.track_characteristics.track_type == 'street':
                return 'soft'
            return 'medium'
        
        # For early stints, use harder compounds
        return 'hard'
    
    def _generate_strategy_notes(
        self,
        pit_windows: List[Dict],
        current_position: int,
        safety_car_prob: float,
        track_evolution: float
    ) -> str:
        """
        Generate strategy notes based on specific conditions.
        
        This method considers:
        1. Pit windows
        2. Current position
        3. Safety car probability
        4. Track evolution
        
        Args:
            pit_windows (List[Dict]): List of pit windows
            current_position (int): Current position (1-20)
            safety_car_prob (float): Probability of safety car
            track_evolution (float): Rate of track improvement
            
        Returns:
            str: Strategy notes
        """
        notes = []
        
        # Basic strategy points
        if self.track_characteristics.track_type == 'street':
            notes.append("Track position is crucial")
        elif self.track_characteristics.track_type == 'high_speed':
            notes.append("Look for undercut opportunities")
            notes.append("Be aggressive in overtaking zones")
            
        # Tire management
        if self.track_characteristics.track_type in ['high_speed', 'technical']:
            notes.append("Manage tires in high-load corners")
            
        # Safety car
        if safety_car_prob >= 0.4:
            notes.append("Keep gaps under 20s for safety car")
            
        return "\n".join(f"- {note}" for note in notes) if notes else ""
    
    def display_strategy(self, strategy: Dict) -> None:
        """
        Display the recommended strategy.
        
        Args:
            strategy (Dict): Strategy recommendation
        """
        print("\nRecommended Strategy:")
        print("----------------------------------------")
        print(f"Current Tires: {strategy['current_compound'].title()} ({strategy['tire_age']} laps old)")
        print(f"Pit Stops Made: {strategy['pit_stops_made']}")
        
        if strategy['previous_stops']:
            stops = ', '.join(f"Lap {lap}" for lap in strategy['previous_stops'])
            print(f"Previous Stops: {stops}")
        print()
        
        print(f"Optimal Number of Remaining Stops: {strategy['recommended_stops']}")
        print()
        
        for window in strategy['pit_windows']:
            print(f"Stop {window['stop_number']} of {strategy['recommended_stops']}:")
            print(f"- Window: Lap {window['window']['start']} - {window['window']['end']}")
            print(f"- Optimal Lap: {window['window']['optimal']}")
            print(f"- Compound: {window['compound']}")
            print()
        
        if strategy['strategy_notes']:
            print("Key Strategy Points:")
            print(strategy['strategy_notes'])
            print()
    
if __name__ == "__main__":
    TrackStrategyOptimizer.get_track_details()
