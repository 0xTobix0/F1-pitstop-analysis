"""
Points-optimized strategy analyzer for specific F1 circuits.
Uses historical race data to determine optimal pit strategies that maximize points potential.
"""

import fastf1
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class StrategyResult:
    """Represents a historical strategy result."""
    stops: int
    compounds: List[str]
    stop_laps: List[int]
    final_position: int
    points_scored: int
    race_conditions: Dict
    strategy_success: float  # 0-1 score based on position gain/loss

class PointsOptimizer:
    """Analyzes historical race data to recommend points-optimizing strategies."""
    
    POINTS_SYSTEM = {
        1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
        6: 8, 7: 6, 8: 4, 9: 2, 10: 1
    }
    
    def __init__(self):
        """Initialize the points optimizer."""
        self.historical_data = []
        fastf1.Cache.enable_cache('cache')
    
    def load_historical_data(self, circuit: str, start_year: int = 2018) -> None:
        """Load historical race data for a specific circuit."""
        current_year = datetime.now().year
        
        for year in range(start_year, current_year):
            try:
                session = fastf1.get_session(year, circuit, 'R')
                session.load()
                
                # Analyze each driver's race
                for driver in session.drivers:
                    driver_laps = session.laps.pick_driver(driver)
                    if driver_laps.empty:
                        continue
                    
                    # Get strategy details
                    stints = driver_laps.groupby('Stint')
                    stops = len(stints) - 1
                    compounds = list(driver_laps['Compound'].unique())
                    stop_laps = [int(stint['LapNumber'].iloc[0]) for _, stint in stints][1:]
                    
                    # Get race result
                    final_pos = driver_laps['Position'].iloc[-1]
                    points = self.POINTS_SYSTEM.get(final_pos, 0)
                    
                    # Calculate strategy success
                    start_pos = driver_laps['Position'].iloc[0]
                    pos_change = start_pos - final_pos
                    max_possible_change = start_pos - 1 if start_pos > 1 else 0
                    strategy_success = (pos_change / max_possible_change) if max_possible_change > 0 else 0
                    
                    # Get race conditions
                    weather = session.weather_data
                    conditions = {
                        'wet_race': bool(weather['Rainfall'].any()),
                        'avg_track_temp': float(weather['TrackTemp'].mean()),
                        'safety_car': any('SC' in str(msg) for msg in session.race_control_messages),
                        'red_flag': any('RED' in str(msg) for msg in session.race_control_messages)
                    }
                    
                    result = StrategyResult(
                        stops=stops,
                        compounds=compounds,
                        stop_laps=stop_laps,
                        final_position=final_pos,
                        points_scored=points,
                        race_conditions=conditions,
                        strategy_success=float(strategy_success)
                    )
                    
                    self.historical_data.append(result)
                    
            except Exception as e:
                print(f"Warning: Could not load data for {circuit} {year}: {str(e)}")
    
    def analyze_successful_strategies(self, conditions: Dict = None) -> Dict:
        """Analyze historical strategies that scored points."""
        if not self.historical_data:
            raise ValueError("No historical data loaded. Call load_historical_data first.")
        
        # Filter for points-scoring strategies
        points_strategies = [s for s in self.historical_data if s.points_scored > 0]
        
        # Filter for similar conditions if specified
        if conditions:
            points_strategies = [
                s for s in points_strategies
                if (conditions.get('wet_race') == s.race_conditions['wet_race']
                    if 'wet_race' in conditions else True)
                and (abs(conditions.get('track_temp', 0) - s.race_conditions['avg_track_temp']) < 10
                    if 'track_temp' in conditions else True)
            ]
        
        if not points_strategies:
            return {
                'error': 'No matching points-scoring strategies found in historical data'
            }
        
        # Analyze strategy patterns
        strategy_analysis = {
            'total_analyzed': len(points_strategies),
            'avg_points': np.mean([s.points_scored for s in points_strategies]),
            'stop_counts': {},
            'compound_combinations': {},
            'stop_windows': {},
            'success_rate': np.mean([s.strategy_success for s in points_strategies])
        }
        
        # Analyze stop counts
        for strat in points_strategies:
            strategy_analysis['stop_counts'][strat.stops] = strategy_analysis['stop_counts'].get(strat.stops, 0) + 1
            
            # Track compound combinations
            compound_key = '->'.join(strat.compounds)
            if compound_key not in strategy_analysis['compound_combinations']:
                strategy_analysis['compound_combinations'][compound_key] = {
                    'count': 0,
                    'avg_points': 0,
                    'success_rate': 0
                }
            combo = strategy_analysis['compound_combinations'][compound_key]
            combo['count'] += 1
            combo['avg_points'] = ((combo['avg_points'] * (combo['count'] - 1)) + strat.points_scored) / combo['count']
            combo['success_rate'] = ((combo['success_rate'] * (combo['count'] - 1)) + strat.strategy_success) / combo['count']
            
            # Track stop windows
            for stop_num, lap in enumerate(strat.stop_laps, 1):
                if stop_num not in strategy_analysis['stop_windows']:
                    strategy_analysis['stop_windows'][stop_num] = []
                strategy_analysis['stop_windows'][stop_num].append(lap)
        
        # Process stop windows into ranges
        for stop_num, laps in strategy_analysis['stop_windows'].items():
            strategy_analysis['stop_windows'][stop_num] = {
                'min': min(laps),
                'max': max(laps),
                'median': int(np.median(laps)),
                'std': np.std(laps)
            }
        
        # Sort compound combinations by success
        strategy_analysis['compound_combinations'] = dict(
            sorted(
                strategy_analysis['compound_combinations'].items(),
                key=lambda x: (x[1]['avg_points'], x[1]['success_rate']),
                reverse=True
            )
        )
        
        # Generate recommendations
        best_stops = max(strategy_analysis['stop_counts'].items(), key=lambda x: x[1])[0]
        best_compounds = list(strategy_analysis['compound_combinations'].keys())[0].split('->')
        
        strategy_analysis['recommendations'] = {
            'optimal_stops': best_stops,
            'optimal_compounds': best_compounds,
            'stop_windows': [
                strategy_analysis['stop_windows'][i]
                for i in range(1, best_stops + 1)
            ],
            'confidence_score': strategy_analysis['success_rate']
        }
        
        return strategy_analysis

    def get_points_optimized_strategy(self, circuit: str, conditions: Dict = None) -> Dict:
        """Get a points-optimized strategy for a specific circuit."""
        try:
            # Load historical data if not already loaded
            if not self.historical_data:
                self.load_historical_data(circuit)
            
            # Analyze successful strategies
            analysis = self.analyze_successful_strategies(conditions)
            
            if 'error' in analysis:
                return analysis
            
            # Format the response
            strategy = {
                'circuit': circuit,
                'strategy_summary': {
                    'recommended_stops': analysis['recommendations']['optimal_stops'],
                    'tire_compounds': analysis['recommendations']['optimal_compounds'],
                    'confidence': f"{analysis['recommendations']['confidence_score']:.2%}"
                },
                'stop_windows': [
                    {
                        'stop_number': i + 1,
                        'window': window,
                        'optimal_lap': window['median']
                    }
                    for i, window in enumerate(analysis['recommendations']['stop_windows'])
                ],
                'historical_insights': {
                    'analyzed_races': analysis['total_analyzed'],
                    'average_points': f"{analysis['avg_points']:.1f}",
                    'most_successful_combinations': [
                        {
                            'compounds': combo.split('->'),
                            'success_rate': f"{data['success_rate']:.2%}",
                            'avg_points': f"{data['avg_points']:.1f}"
                        }
                        for combo, data in list(analysis['compound_combinations'].items())[:3]
                    ]
                }
            }
            
            return strategy
            
        except Exception as e:
            return {'error': f"Failed to generate strategy: {str(e)}"}
