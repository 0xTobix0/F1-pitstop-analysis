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
import math

@dataclass
class TrackCharacteristics:
    """Characteristics of the Australian GP track."""
    length: float = 5.278
    race_laps: int = 58
    pit_loss_time: float = 21.5
    track_type: str = 'standard'
    max_stops: int = 3
    tire_degradation: float = 1.0
    track_evolution: float = 0.08
    safety_car_probability: float = 0.35
    traffic_impact: float = 0.6
    pit_window_margin: int = 3
    difficult_overtaking: bool = False

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
    """F1 Strategy Optimizer for Australian GP."""
    
    # Tire compound characteristics
    TIRE_COMPOUNDS = {
        'soft': {
            'max_life': 20,
            'grip_level': 1.0,
            'degradation_rate': 0.15
        },
        'medium': {
            'max_life': 30,
            'grip_level': 0.9,
            'degradation_rate': 0.12
        },
        'hard': {
            'max_life': 40,
            'grip_level': 0.8,
            'degradation_rate': 0.10
        }
    }
    
    def __init__(self):
        """Initialize strategy optimizer for Australian GP."""
        self.track_characteristics = TrackCharacteristics()
        self._current_position = None
    
    def generate_strategy(
        self,
        current_lap: int,
        current_position: int,
        current_compound: str,
        tire_age: int,
        previous_stops: List[int]
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
            current_compound (str): Current tire compound (soft/medium/hard)
            tire_age (int): Age of current tires
            previous_stops (List[int]): Lap numbers of previous pit stops
        
        Returns:
            Dict: Strategy recommendation containing:
                - recommended_stops: Number of remaining pit stops
                - pit_windows: List of StopWindow objects
                - strategy_notes: List of important strategy considerations
                - current_compound: Current tire compound
                - tire_age: Current tire age
                - pit_stops_made: Number of pit stops made
                - previous_stops: List of previous pit stop laps
        """
        remaining_laps = self.track_characteristics.race_laps - current_lap
        stops_made = len(previous_stops)
        
        # Store current position for strategy calculations
        self._current_position = current_position
        
        # Early return if race is finished
        if remaining_laps <= 0:
            return {
                'recommended_stops': 0,
                'pit_windows': [],
                'strategy_notes': ['Race is finished'],
                'current_compound': current_compound,
                'tire_age': tire_age,
                'pit_stops_made': stops_made,
                'previous_stops': previous_stops
            }
        
        # Calculate optimal strategy considering track type and characteristics
        base_stops = self._calculate_optimal_stops(
            remaining_laps=remaining_laps,
            current_compound=current_compound,
            tire_age=tire_age,
            stops_made=stops_made
        )
        
        # Generate pit windows for each remaining stop
        pit_windows = self._generate_pit_windows(
            remaining_laps=remaining_laps,
            num_stops=base_stops,
            current_compound=current_compound,
            tire_age=tire_age,
            current_lap=current_lap
        )
        
        # Add strategy notes based on specific conditions
        strategy_notes = self._generate_strategy_notes(
            current_compound=current_compound,
            tire_age=tire_age,
            current_position=current_position,
            recommended_stops=base_stops,
            pit_windows=pit_windows,
            current_lap=current_lap
        )
        
        return {
            'recommended_stops': base_stops,
            'pit_windows': pit_windows,
            'strategy_notes': strategy_notes,
            'current_compound': current_compound,
            'tire_age': tire_age,
            'pit_stops_made': stops_made,
            'previous_stops': previous_stops,
            'current_lap': current_lap
        }
    
    def _calculate_optimal_stops(
        self,
        remaining_laps: int,
        current_compound: str,
        tire_age: int,
        stops_made: int
    ) -> int:
        """Calculate optimal number of remaining pit stops."""
        # Base calculation from tire life
        compound_data = self.TIRE_COMPOUNDS[current_compound]
        remaining_tire_life = compound_data['max_life'] - tire_age
        
        # Calculate theoretical stops needed
        theoretical_stops = math.ceil(
            (remaining_laps - remaining_tire_life) /
            self.TIRE_COMPOUNDS['medium']['max_life']  # Use medium as baseline
        )
        
        # Adjust for Australian GP characteristics
        base_stops = theoretical_stops
        
        # Prefer 2-stop strategy for Australian GP
        total_stops = stops_made + base_stops
        if total_stops > 2:
            # Only recommend 3 stops if really necessary
            if (
                remaining_laps > 40 and  # Lots of laps remaining
                (tire_age > compound_data['max_life'] * 0.8 or  # Current tires almost done
                 self.track_characteristics.tire_degradation > 1.2)  # Very high degradation
            ):
                return min(base_stops, 3 - stops_made)
            else:
                # Force 2-stop strategy
                return min(base_stops, 2 - stops_made)
        
        return base_stops
    
    def _generate_pit_windows(
        self,
        remaining_laps: int,
        num_stops: int,
        current_compound: str,
        tire_age: int,
        current_lap: int
    ) -> List[Dict]:
        """
        Generate optimal pit windows for the remaining stops.
        
        This method calculates pit windows considering:
        1. Tire compound life
        2. Track evolution
        3. Safety car probability
        4. Traffic conditions
        5. Current tire condition and compound
        
        Args:
            remaining_laps (int): Number of laps left in race
            num_stops (int): Number of remaining pit stops
            current_compound (str): Current tire compound
            tire_age (int): Age of current tires
            current_lap (int): Current lap number
            
        Returns:
            List[Dict]: List of pit windows with:
                - window_start: Starting lap for pit window
                - window_end: Ending lap for pit window
                - optimal_lap: Optimal lap to pit
                - compound: Recommended compound
        """
        if num_stops == 0:
            return []
        
        # Get tire compound characteristics
        compounds = list(self.TIRE_COMPOUNDS.keys())
        windows = []
        
        # Calculate first stop based on current tire life
        if current_compound and tire_age is not None:
            compound_data = self.TIRE_COMPOUNDS[current_compound]
            remaining_tire_life = compound_data['max_life'] - tire_age
            
            # Adjust for track type tire degradation factor
            if self.track_characteristics.track_type == 'street':
                remaining_tire_life = int(remaining_tire_life / 0.8)  # Lower degradation
            
            # First window starts near end of current tire life
            first_window_start = current_lap + (remaining_tire_life * 0.7)  # Start window at 70% of tire life
            first_window_end = current_lap + remaining_tire_life
            first_optimal_lap = first_window_start + (first_window_end - first_window_start) * 0.5
        else:
            # Default window if no tire data
            first_window_start = current_lap + 20
            first_window_end = first_window_start + self.track_characteristics.pit_window_margin * 2
            first_optimal_lap = first_window_start + self.track_characteristics.pit_window_margin
        
        # Calculate remaining stint lengths
        remaining_after_first = remaining_laps - (first_optimal_lap - current_lap)
        avg_stint = remaining_after_first // num_stops if num_stops > 1 else remaining_after_first
        
        # Add first window
        laps_after_stop = remaining_laps - (first_optimal_lap - current_lap)
        first_compound = self._select_compound(laps_after_stop, current_compound)
        windows.append({
            'window_start': int(first_window_start),
            'window_end': int(first_window_end),
            'optimal_lap': int(first_optimal_lap),
            'compound': first_compound
        })
        
        # Add remaining windows if any
        last_optimal = first_optimal_lap
        for i in range(1, num_stops):
            window_start = last_optimal + (avg_stint * 0.7)  # Start at 70% of stint
            window_end = window_start + self.track_characteristics.pit_window_margin * 2
            optimal_lap = window_start + self.track_characteristics.pit_window_margin
            
            laps_after_stop = remaining_laps - (optimal_lap - current_lap)
            compound = self._select_compound(laps_after_stop, current_compound)
            
            windows.append({
                'window_start': int(window_start),
                'window_end': int(window_end),
                'optimal_lap': int(optimal_lap),
                'compound': compound
            })
            last_optimal = optimal_lap
        
        return windows
    
    def _select_compound(self, remaining_laps: int, current_compound: str = None) -> str:
        """Select optimal tire compound based on remaining laps and track type."""
        if remaining_laps > 35:  # Long stint
            return 'hard'
        elif remaining_laps > 25:  # Medium stint
            return 'medium'
        else:  # Short stint or end of race
            # For street circuits, prefer harder compounds unless very short stint
            if self.track_characteristics.track_type == 'street' and remaining_laps > 15:
                return 'medium'
            return 'soft'
    
    def _generate_strategy_notes(
        self,
        current_compound: str,
        tire_age: int,
        current_position: int,
        recommended_stops: int,
        pit_windows: List[Dict],
        current_lap: int
    ) -> List[str]:
        """Generate strategy notes based on Australian GP characteristics."""
        notes = []
        
        # Track-specific notes
        notes.append("High safety car chance at Albert Park - stay ready to adapt strategy")
        
        # Position-based strategy
        if current_position <= 5:
            notes.append("Focus on maintaining track position, only pit when necessary")
        elif current_position <= 10:
            notes.append("Look for undercut opportunities on cars ahead")
        else:
            notes.append("Consider aggressive strategy to gain positions")
            
        # Tire compound specific notes
        if current_compound == 'soft':
            if tire_age > 10:
                notes.append("Warning: Soft tires approaching critical age (>15 laps)")
            if tire_age > 15:
                notes.append("Critical: Plan pit stop immediately - soft tires severely degraded")
        elif current_compound == 'medium':
            if tire_age > 20:
                notes.append("Warning: Medium tires approaching critical age (>25 laps)")
            if tire_age > 25:
                notes.append("Critical: Plan pit stop - medium tires severely degraded")
        elif current_compound == 'hard':
            if tire_age > 30:
                notes.append("Warning: Hard tires approaching critical age (>35 laps)")
            if tire_age > 35:
                notes.append("Critical: Plan pit stop - hard tires severely degraded")
                
        # Next compound recommendation
        remaining_laps = 58 - current_lap  # Australian GP is 58 laps
        if remaining_laps <= 20:
            notes.append("Final stint on softs - manage tire life for late race overtaking")
        elif remaining_laps <= 30:
            notes.append("Medium tires provide good balance for final stint")
        else:
            notes.append("Consider hard tires for longer stint flexibility")
            
        # Track evolution notes
        if current_lap < 20:
            notes.append("Track grip improving - expect faster lap times")
        elif current_lap > 40:
            notes.append("Track fully rubbered in - optimal grip conditions")
            
        return notes
    
    def _validate_pit_stop(
        self,
        stop_lap: int,
        current_lap: int,
        previous_stops: List[int],
        compound: str,
        tire_age: int,
        is_historical: bool = False
    ) -> Tuple[bool, str]:
        """Validate pit stop timing based on Australian GP characteristics."""
        # Check basic lap validity
        if stop_lap < 1 or stop_lap > 58:  # Australian GP is 58 laps
            return False, "Invalid lap number"
            
        # Different validation for historical vs future stops
        if is_historical:
            if stop_lap >= current_lap:
                return False, f"Historical pit stop must be before current lap {current_lap}"
        else:
            if stop_lap <= current_lap:
                return False, "Future pit stop must be after current lap"
            
        # Check compound-specific tire life
        compound_data = self.TIRE_COMPOUNDS[compound]
        if not is_historical and tire_age > compound_data['max_life']:
            return False, f"Current {compound} tires will not last until lap {stop_lap}"
            
        # Validate minimum stint length (based on Australian GP characteristics)
        min_stint = 5  # Minimum viable stint length
        if previous_stops:
            last_stop = max(previous_stops)
            if stop_lap - last_stop < min_stint:
                return False, f"Stint must be at least {min_stint} laps long"
            
        # Only validate critical tire ages for future stops
        if not is_historical:
            if compound == 'soft' and tire_age > 15:
                return False, "Soft tires are beyond critical age (>15 laps)"
            elif compound == 'medium' and tire_age > 25:
                return False, "Medium tires are beyond critical age (>25 laps)"
            elif compound == 'hard' and tire_age > 35:
                return False, "Hard tires are beyond critical age (>35 laps)"
            
        return True, ""

    def display_strategy(self, strategy: Dict) -> None:
        """
        Display the recommended pit stop strategy.
        
        Args:
            strategy (Dict): Strategy recommendation
        """
        print("\nRecommended Strategy:")
        print("-" * 40)
        
        # Current situation
        compound_age_str = f"{strategy['current_compound'].title()} ({strategy['tire_age']} laps old)"
        print(f"Current Tires: {compound_age_str}")
        
        # Pit stop history
        stops_made = len(strategy['previous_stops'])
        print(f"Pit Stops Made: {stops_made}")
        if strategy['previous_stops']:
            stops_str = ', '.join(f"Lap {lap}" for lap in strategy['previous_stops'])
            print(f"Previous Stops: {stops_str}")
        
        # Tire life analysis
        compound_data = self.TIRE_COMPOUNDS[strategy['current_compound']]
        remaining_life = compound_data['max_life'] - strategy['tire_age']
        print(f"\nCurrent Tire Analysis:")
        print(f"- Maximum Life: {compound_data['max_life']} laps")
        print(f"- Remaining Life: ~{remaining_life} laps")
        print(f"- Grip Level: {compound_data['grip_level']:.1f}")
        print(f"- Degradation Rate: {compound_data['degradation_rate']*100:.1f}%")
        
        # Future strategy
        print(f"\nOptimal Number of Remaining Stops: {strategy['recommended_stops']}")
        print("\nPit Windows:")
        for i, window in enumerate(strategy['pit_windows'], 1):
            print(f"\nStop {i}:")
            print(f"- Window: Laps {window['window_start']}-{window['window_end']}")
            print(f"- Optimal Lap: {window['optimal_lap']}")
            print(f"- Compound: {window['compound'].title()}")
            
            # Add compound-specific info
            compound_info = self.TIRE_COMPOUNDS[window['compound']]
            print(f"  > Max Life: {compound_info['max_life']} laps")
            print(f"  > Grip Level: {compound_info['grip_level']:.1f}")
            print(f"  > Degradation: {compound_info['degradation_rate']*100:.1f}%")
        
        if strategy['strategy_notes']:
            print("\nStrategy Notes:")
            for note in strategy['strategy_notes']:
                print(f"- {note}")
            print()
    
    @classmethod
    def run_interactive(cls):
        """Run the strategy optimizer in interactive mode."""
        while True:
            print("\nF1 Strategy Optimizer - Australian GP")
            print("=" * 40)
            print("\nOptions:")
            print("1. Generate new strategy")
            print("2. Exit")
            
            try:
                choice = input("\nEnter your choice (1-2): ").strip()
                if choice == "2":
                    print("\nGoodbye!")
                    break
                elif choice != "1":
                    print("\nInvalid choice. Please enter 1 or 2.")
                    continue
                
                print("\nAustralian GP Configuration:")
                print("-" * 40)
                track = TrackCharacteristics()
                print(f"Track Length: {track.length} km")
                print(f"Track Type: {track.track_type.title()}")
                print(f"Maximum Pit Stops: {track.max_stops}")
                print(f"Pit Loss Time: {track.pit_loss_time} seconds\n")
                
                print("Track Characteristics:")
                print(f"- Tire Degradation: {track.tire_degradation}x")
                print(f"- Track Evolution: {track.track_evolution:.3f}")
                print(f"- Safety Car Probability: {track.safety_car_probability*100:.1f}%")
                print(f"- Traffic Impact: {track.traffic_impact:.2f}")
                print(f"- Overtaking Difficulty: {'High' if track.difficult_overtaking else 'Normal'}")
                print(f"- Pit Window Margin: {track.pit_window_margin} laps\n")
                
                # Get current race situation
                print("Enter Race Details:")
                print("-" * 20)
                current_position = int(input("Current position (1-20): "))
                if not 1 <= current_position <= 20:
                    raise ValueError("Position must be between 1 and 20")
                
                current_lap = int(input("Current lap (1-58): "))
                if not 1 <= current_lap <= 58:
                    raise ValueError("Lap number must be between 1 and 58")
                
                print("\nCurrent Tire Details:")
                print("-" * 20)
                print("Available compounds: soft, medium, hard")
                print("Compound characteristics:")
                print("- Soft:   Max life 20 laps, High grip, 15% degradation")
                print("- Medium: Max life 30 laps, Good grip, 12% degradation")
                print("- Hard:   Max life 40 laps, Lower grip, 10% degradation")
                current_compound = input("Current tire compound: ").lower()
                if current_compound not in ['soft', 'medium', 'hard']:
                    raise ValueError("Invalid tire compound")
                
                tire_age = int(input("Current tire age (laps): "))
                if tire_age < 0:
                    raise ValueError("Tire age cannot be negative")
                if tire_age > current_lap:
                    raise ValueError("Tire age cannot be greater than current lap")
                
                # Validate tire age against compound limits
                compound_data = cls.TIRE_COMPOUNDS[current_compound]
                if tire_age > compound_data['max_life']:
                    raise ValueError(f"{current_compound.title()} tires cannot last {tire_age} laps (max {compound_data['max_life']} laps)")
                
                print("\nPrevious Pit Stops:")
                print("-" * 20)
                stops_made = int(input("Number of pit stops made (0-3): "))
                if not 0 <= stops_made <= 3:
                    raise ValueError("Number of stops must be between 0 and 3")
                
                previous_stops = []
                optimizer = cls()
                for i in range(stops_made):
                    while True:
                        try:
                            lap_input = input(f"Lap number of pit stop {i+1} (or 'back' to restart): ")
                            if lap_input.lower() == 'back':
                                previous_stops = []
                                break
                                
                            lap = int(lap_input)
                            valid, error = optimizer._validate_pit_stop(
                                stop_lap=lap,
                                current_lap=current_lap,
                                previous_stops=previous_stops,
                                compound=current_compound,
                                tire_age=tire_age,
                                is_historical=True
                            )
                            if not valid:
                                print(f"Invalid pit stop: {error}")
                                continue
                            previous_stops.append(lap)
                            break
                        except ValueError:
                            print("Please enter a valid lap number or 'back'")
                            continue
                            
                    if lap_input.lower() == 'back':
                        continue
                
                # Sort pit stops chronologically
                previous_stops.sort()
                
                # Generate and display strategy
                strategy = optimizer.generate_strategy(
                    current_lap=current_lap,
                    current_position=current_position,
                    current_compound=current_compound,
                    tire_age=tire_age,
                    previous_stops=previous_stops
                )
                
                optimizer.display_strategy(strategy)
                
                input("\nPress Enter to continue...")
                
            except ValueError as e:
                print(f"\nError: {str(e)}")
                input("\nPress Enter to continue...")
            except Exception as e:
                print(f"\nUnexpected error: {str(e)}")
                input("\nPress Enter to continue...")

if __name__ == "__main__":
    TrackStrategyOptimizer.run_interactive()
