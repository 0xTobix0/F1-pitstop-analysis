"""Track strategy optimizer module."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from src.models.fastf1_loader import FastF1DataLoader

@dataclass
class TrackCharacteristics:
    """Track characteristics data class."""
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
        self.tire_compounds = ['soft', 'medium', 'hard']  
    
    @property
    def high_degradation(self) -> bool:
        """Check if track has high tire degradation."""
        return self.tire_degradation > 1.1

    @property
    def high_evolution(self) -> bool:
        """Check if track has high evolution."""
        return self.track_evolution > 0.008

    @property
    def high_safety_car(self) -> bool:
        """Check if track has high safety car probability."""
        return self.safety_car_probability > 0.35

    @property
    def difficult_overtaking(self) -> bool:
        """Check if track has difficult overtaking."""
        return self.overtaking_difficulty > 0.7

@dataclass
class StopWindow:
    """Pit stop window data class."""
    start_lap: int = field(metadata={"description": "Start lap of the pit window"})
    optimal_lap: int = field(metadata={"description": "Optimal lap to pit"})
    end_lap: int = field(metadata={"description": "End lap of the pit window"})
    compound: str = field(metadata={"description": "Recommended tire compound"})

class TrackStrategyOptimizer:
    """Track strategy optimizer class."""
    
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
    
    TIRE_COMPOUNDS = {
        'soft': {
            'min_life': 10,
            'max_life': 20,
            'pace_advantage': 0.8,
            'optimal_window': (0, 15),  # Best performance in first 15 laps
            'high_deg_tracks': ['monaco_gp', 'singapore_gp'],  # Tracks where this compound degrades faster
            'preferred_phase': 'end'  # Best used at end of race
        },
        'medium': {
            'min_life': 20,
            'max_life': 35,
            'pace_advantage': 0.4,
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
        """Initialize with track name and year."""
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
        """Load track configuration."""
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
        """Update track characteristics based on real data."""
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
            # Update safety car probability
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
            
            # Update track evolution based on weather stability
            if 'weather_stability' in weather_data:
                self.track_characteristics.track_evolution *= weather_data['weather_stability']
            
            # Update traffic impact based on weather conditions
            if weather_data.get('rainfall', False):
                self.track_characteristics.traffic_impact *= 1.2  # 20% higher in wet conditions
    
    @staticmethod
    def get_track_details():
        """Get and display details for a specific F1 track and generate pit stop strategy."""
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
                'spa': 'belgian_gp',
                'monza': 'italian_gp',
                'silverstone': 'british_gp',
                'melbourne': 'australian_gp',
                'shanghai': 'chinese_gp',
                'suzuka': 'japanese_gp',
                'sakhir': 'bahrain_gp',
                'jeddah': 'saudi_arabia_gp',
                'imola': 'emilia_romagna_gp',
                'monte carlo': 'monaco_gp',
                'catalunya': 'spanish_gp',
                'montreal': 'canadian_gp',
                'spielberg': 'austrian_gp',
                'hungaroring': 'hungarian_gp',
                'zandvoort': 'dutch_gp',
                'baku': 'azerbaijan_gp',
                'marina bay': 'singapore_gp',
                'cota': 'united_states_gp',
                'austin': 'united_states_gp',
                'mexico city': 'mexican_gp',
                'interlagos': 'brazilian_gp',
                'sao paulo': 'brazilian_gp',
                'lusail': 'qatar_gp',
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
                
                print("\nRecommended Strategy:")
                print("-" * 40)
                print(f"Current Tires: {current_compound.title()} ({tire_age} laps old)")
                print(f"Pit Stops Made: {num_stops}")
                if previous_stops:
                    print("Previous Stops:", ", ".join(f"Lap {lap}" for lap in previous_stops))
                print(f"\nOptimal Number of Remaining Stops: {strategy['recommended_stops']}")
                print()
                
                for i, window in enumerate(strategy['pit_windows'], 1):
                    print(f"Stop {i} of {strategy['recommended_stops']}:")
                    print(f"- Window: Lap {window['start_lap']} - {window['end_lap']}")
                    print(f"- Optimal Lap: {window['optimal_lap']}")
                    print(f"- Compound: {window['compound']}")
                    print()
                
                print("Key Strategy Points:")
                for note in strategy['strategy_notes']:
                    print(f"- {note}")
                print()
                
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
        """Generate pit stop strategy.
        
        Args:
            current_lap: Current lap number
            current_position: Current race position
            current_compound: Current tire compound
            tire_age: Age of current tires in laps
            previous_stops: List of lap numbers where previous stops were made
        """
        if previous_stops is None:
            previous_stops = []
        
        remaining_laps = self.track_characteristics.race_laps - current_lap
        stops_made = len(previous_stops)
        max_remaining_stops = self.track_characteristics.max_stops - stops_made
        
        # Get tire data
        compounds_data = self.TIRE_COMPOUNDS.copy()
        if current_compound:
            # Update tire life expectancy based on current age
            compound_data = compounds_data[current_compound]
            remaining_life = compound_data['max_life'] - tire_age
            if remaining_life < 0:
                remaining_life = 0
            compounds_data[current_compound]['remaining_life'] = remaining_life
        
        # Calculate optimal number of remaining stops
        if remaining_laps <= 0:
            return {
                'recommended_stops': 0,
                'pit_windows': [],
                'strategy_notes': ['Race is finished']
            }
        
        # Determine optimal number of stops based on:
        # 1. Remaining laps
        # 2. Current tire condition
        # 3. Track characteristics
        # 4. Previous pit stop pattern
        base_stops = self._calculate_optimal_stops(
            remaining_laps=remaining_laps,
            current_compound=current_compound,
            tire_age=tire_age,
            previous_stops=previous_stops
        )
        
        recommended_stops = min(base_stops, max_remaining_stops)
        
        # Calculate stop windows
        stop_windows = self._calculate_stop_windows(
            current_lap=current_lap,
            base_stops=recommended_stops,
            remaining_laps=remaining_laps,
            current_compound=current_compound,
            tire_age=tire_age
        )
        
        # Generate strategy notes
        strategy_notes = self._generate_strategy_notes(
            current_position=current_position,
            stop_windows=stop_windows,
            current_compound=current_compound,
            tire_age=tire_age
        )
        
        return {
            'recommended_stops': recommended_stops,
            'pit_windows': stop_windows,
            'strategy_notes': strategy_notes
        }
        
    def _calculate_optimal_stops(
        self,
        remaining_laps: int,
        current_compound: str,
        tire_age: int,
        previous_stops: List[int]
    ) -> int:
        """Calculate optimal number of remaining pit stops."""
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
        if self.track_characteristics.tire_degradation > 1.2:  # High degradation
            base_stops += 1
        elif self.track_characteristics.tire_degradation < 0.8:  # Low degradation
            base_stops = max(0, base_stops - 1)
        
        # Consider previous pit stop pattern
        if previous_stops:
            avg_stint_length = remaining_laps / (len(previous_stops) + 1)
            remaining_stints = remaining_laps / avg_stint_length
            base_stops = max(base_stops, int(remaining_stints))
        
        return base_stops
    
    def _calculate_stop_windows(
        self,
        current_lap: int,
        base_stops: int,
        remaining_laps: int,
        current_compound: str,
        tire_age: int
    ) -> List[Dict]:
        """Calculate optimal pit stop windows."""
        stop_windows = []
        remaining = remaining_laps
        
        for stop in range(base_stops):
            # Calculate optimal stint length for this stop
            if stop == base_stops - 1:
                # Last stint - use all remaining laps
                stint_length = remaining
            else:
                # Earlier stints - distribute remaining laps
                base_stint = remaining / (base_stops - stop)
                
                # Adjust stint length based on track characteristics
                if self.track_characteristics.track_type == 'high_speed':
                    # Shorter stints for high-speed tracks due to higher wear
                    base_stint *= 0.95
                elif self.track_characteristics.track_type == 'street':
                    # Longer stints possible on street circuits
                    base_stint *= 1.05
                
                # Further adjust for high degradation
                if self.track_characteristics.high_degradation:
                    base_stint *= 0.95
                
                stint_length = base_stint
            
            # Calculate window margins
            margin = self.track_characteristics.pit_window_margin
            if self.track_characteristics.difficult_overtaking:
                margin = max(2, margin - 1)  
            elif self.track_characteristics.track_type == 'high_speed':
                margin = min(6, margin + 1)  
            
            # Create window
            window = {
                'start_lap': max(current_lap, round(current_lap + stint_length - margin)),
                'optimal_lap': round(current_lap + stint_length),
                'end_lap': min(
                    self.track_characteristics.race_laps - 1,
                    round(current_lap + stint_length + margin)
                ),
                'compound': self._recommend_compound(stop, base_stops, remaining, current_compound)
            }
            
            stop_windows.append(window)
            current_lap = window['optimal_lap']
            remaining = self.track_characteristics.race_laps - current_lap
        
        return stop_windows
    
    def _recommend_compound(self, stop_number: int, num_stops: int, remaining_laps: int, current_compound: str = None) -> str:
        """Recommend tire compound based on strategy and remaining laps."""
        available_compounds = self.track_characteristics.tire_compounds
        compounds_data = self.TIRE_COMPOUNDS
        
        # Filter out current compound if it's worn
        if current_compound and current_compound in available_compounds:
            compound_data = compounds_data[current_compound]
            if compound_data.get('remaining_life', compound_data['max_life']) < 10:
                available_compounds = [c for c in available_compounds if c != current_compound]
        
        # No valid compounds available
        if not available_compounds:
            return 'medium'  # Default to medium if no other options
        
        # Single compound available
        if len(available_compounds) == 1:
            return available_compounds[0]
        
        # Calculate race phase (start: 0-0.3, middle: 0.3-0.7, end: 0.7-1.0)
        race_progress = 1 - (remaining_laps / self.track_characteristics.race_laps)
        
        if race_progress < 0.3:
            phase = 'start'
        elif race_progress < 0.7:
            phase = 'middle'
        else:
            phase = 'end'
        
        # Filter compounds by phase preference
        phase_compounds = [
            c for c in available_compounds
            if compounds_data[c]['preferred_phase'] in [phase, 'any']
        ]
        
        if phase_compounds:
            available_compounds = phase_compounds
        
        # Consider track-specific high degradation
        high_deg_compounds = [
            c for c in available_compounds
            if self.track_name in compounds_data[c]['high_deg_tracks']
        ]
        
        if high_deg_compounds and race_progress > 0.7:
            # Avoid high degradation compounds near the end
            available_compounds = [
                c for c in available_compounds
                if c not in high_deg_compounds
            ]
        
        # Sort by optimal window
        viable_compounds = []
        for compound in available_compounds:
            window_start, window_end = compounds_data[compound]['optimal_window']
            if window_start <= remaining_laps <= window_end:
                viable_compounds.append(compound)
        
        if viable_compounds:
            # Pick middle compound by pace advantage
            return sorted(
                viable_compounds,
                key=lambda c: compounds_data[c]['pace_advantage']
            )[len(viable_compounds)//2]
        
        # Default to hardest available compound
        return sorted(
            available_compounds,
            key=lambda c: compounds_data[c]['max_life']
        )[-1]
    
    def _generate_strategy_notes(
        self,
        current_position: int,
        stop_windows: List[Dict],
        current_compound: str,
        tire_age: int
    ) -> List[str]:
        """Generate strategy notes."""
        strategy_notes = []
        
        # Add position-based strategy
        if current_position <= 3:
            strategy_notes.append("Priority: Maintain track position")
            if self.track_characteristics.track_type == 'high_speed':
                strategy_notes.append("Use DRS to defend and manage gap")
        elif current_position <= 10:
            strategy_notes.append("Priority: Look for undercut opportunities")
            if not self.track_characteristics.difficult_overtaking:
                strategy_notes.append("Be aggressive in overtaking zones")
        else:
            strategy_notes.append("Priority: Consider alternate strategy")
            if self.track_characteristics.high_safety_car:
                strategy_notes.append("Stay out longer - potential safety car advantage")
        
        # Add key track-specific notes
        if self.track_name == 'japanese_gp':
            strategy_notes.extend([
                "Key overtaking: Spoon Curve and 130R",
                "Critical sectors: Sectors 1 and 2",
                "Monitor weather - high variability"
            ])
        elif self.track_name == 'bahrain_gp':
            strategy_notes.extend([
                "Key overtaking: Turn 1 and Turn 4",
                "Watch brake temperatures",
                "Use slipstream on straights"
            ])
        elif self.track_name == 'monaco_gp':
            strategy_notes.extend([
                "Track position is everything",
                "Undercut is very powerful",
                "Key sector: Casino to Tunnel"
            ])
        
        # Add tire management note if needed
        if self.track_characteristics.high_degradation:
            strategy_notes.append("Manage tires in high-load corners")
        
        # Add safety car note if high probability
        if self.track_characteristics.high_safety_car:
            strategy_notes.append("Keep gaps under 20s for safety car")
        
        # Add current tire status note
        if current_compound and tire_age:
            compound_data = self.TIRE_COMPOUNDS[current_compound]
            remaining_life = compound_data['max_life'] - tire_age
            if remaining_life < 10:
                strategy_notes.append(f"Current tires ({current_compound}) are worn out ({tire_age} laps old)")
        
        return strategy_notes

if __name__ == "__main__":
    TrackStrategyOptimizer.get_track_details()
