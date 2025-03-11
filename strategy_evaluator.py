"""F1 Strategy Evaluator.

This script evaluates circuit-specific strategy recommendations.
"""

import fastf1
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from src.models.fastf1_loader import FastF1DataLoader
from src.models.track_strategy import TrackStrategyOptimizer
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class StrategyEvaluator:
    """Evaluates strategy recommendations for specific circuits."""
    
    # Track name mapping for FastF1
    TRACK_MAPPING = {
        'monaco': 'Monaco',
        'spa': 'Belgium',
        'monza': 'Italy',
        'silverstone': 'Great Britain',
        'melbourne': 'Australia'
    }
    
    # Track-specific characteristics
    TRACK_CHARACTERISTICS = {
        'monaco': {
            'description': 'Tight street circuit with limited overtaking opportunities',
            'key_factors': [
                'Track position critical',
                'High traffic impact',
                'Overcut potential',
                'Safety car likely'
            ],
            'strategy_priority': 'Track Position'
        },
        'spa': {
            'description': 'High-speed circuit with multiple overtaking zones',
            'key_factors': [
                'High tire degradation',
                'Multiple overtaking spots',
                'Weather variability',
                'Long lap length'
            ],
            'strategy_priority': 'Optimal Stop Timing'
        },
        'monza': {
            'description': 'High-speed circuit with long straights',
            'key_factors': [
                'Low downforce setup',
                'High tire stress',
                'DRS effectiveness',
                'Draft importance'
            ],
            'strategy_priority': 'Straight Line Speed'
        },
        'silverstone': {
            'description': 'Fast, flowing circuit with high-speed corners',
            'key_factors': [
                'High tire loads',
                'Multiple racing lines',
                'Weather impact',
                'Medium tire wear'
            ],
            'strategy_priority': 'Tire Management'
        },
        'australia': {
            'description': 'Technical circuit with mix of corners',
            'key_factors': [
                'Medium tire wear',
                'Track evolution',
                'Mixed corner types',
                'DRS zones'
            ],
            'strategy_priority': 'Balanced Approach'
        }
    }
    
    def __init__(self):
        """Initialize strategy evaluator."""
        self.loader = FastF1DataLoader()
        fastf1.Cache.enable_cache('cache')
    
    def get_circuit_statistics(self, year: int, gp_name: str) -> Dict:
        """Get circuit-specific statistics from FastF1."""
        fastf1_name = self.TRACK_MAPPING.get(gp_name.lower(), gp_name)
        session = fastf1.get_session(year, fastf1_name, 'R')
        session.load()
        
        # Analyze pit stop patterns
        all_stops = []
        compounds_used = set()
        
        for driver in session.drivers:
            driver_laps = session.laps.pick_driver(driver)
            stints = driver_laps['Stint'].unique()
            compounds_used.update(driver_laps['Compound'].unique())
            all_stops.append(len(stints) - 1)
        
        return {
            'median_stops': np.median(all_stops),
            'compounds_used': list(compounds_used),
            'safety_car': any('SC' in str(msg) for msg in session.race_control_messages)
        }
    
    def evaluate_circuit_strategy(self, year: int, gp_name: str) -> Dict:
        """Evaluate strategy for a specific circuit."""
        optimizer = TrackStrategyOptimizer(gp_name.lower(), year)
        session_data = self.loader.load_session(year, gp_name, 'R')
        
        optimizer.update_characteristics(
            tire_data=session_data['tire_performance'],
            weather_data=session_data['weather']
        )
        
        recommendation = optimizer.recommend_strategy(
            race_distance=session_data['race_distance'],
            weather_condition='wet' if session_data['weather'].get('rainfall', False) else 'dry'
        )
        
        circuit_stats = self.get_circuit_statistics(year, gp_name)
        
        return {
            'recommendation': recommendation,
            'circuit_stats': circuit_stats,
            'weather': session_data['weather'],
            'track_info': self.TRACK_CHARACTERISTICS.get(gp_name.lower(), {
                'description': 'Standard F1 circuit',
                'key_factors': ['Balanced characteristics'],
                'strategy_priority': 'Flexible Strategy'
            })
        }
    
    def plot_strategy_visualization(self, evaluation_results: Dict, circuit_name: str, save_path: str = None):
        """Create circuit strategy visualization."""
        plt.figure(figsize=(15, 8))
        
        # Plot 1: Stop Windows (Top Left)
        plt.subplot(2, 2, 1)
        windows = evaluation_results['recommendation']['stop_windows']
        for i, window in enumerate(windows, 1):
            plt.axvspan(window['start'], window['end'], alpha=0.2, label=f'Stop {i} Window')
            plt.axvline(window['optimal'], color='red', linestyle='--', label=f'Optimal Stop {i}')
        plt.title('Pit Stop Windows')
        plt.xlabel('Race Distance (km)')
        plt.ylabel('Stop Number')
        plt.legend()
        
        # Plot 2: Track Characteristics (Top Right)
        plt.subplot(2, 2, 2)
        track_chars = evaluation_results['recommendation']['track_characteristics']
        metrics = [
            ('Track Evolution', track_chars.track_evolution * 100),
            ('Safety Car Prob', track_chars.safety_car_probability * 100),
            ('Traffic Impact', track_chars.traffic_impact * 100),
            ('Overtaking Diff', track_chars.overtaking_difficulty * 100)
        ]
        x = [m[0] for m in metrics]
        y = [m[1] for m in metrics]
        bars = plt.bar(x, y)
        plt.title('Track Characteristics')
        plt.ylabel('Percentage')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom')
        
        # Plot 3: Circuit Info (Bottom Left)
        plt.subplot(2, 2, 3)
        track_info = evaluation_results['track_info']
        plt.axis('off')
        info_text = [
            f"Circuit: {circuit_name}",
            f"Type: {track_chars.track_type.replace('_', ' ').title()}",
            f"Length: {track_chars.track_length:.3f} km",
            f"Description: {track_info['description']}",
            "\nKey Factors:",
        ] + [f"• {factor}" for factor in track_info['key_factors']] + [
            f"\nStrategy Priority: {track_info['strategy_priority']}"
        ]
        plt.text(0, 1, '\n'.join(info_text), 
                va='top', ha='left', wrap=True,
                fontsize=9)
        
        # Plot 4: Strategy Summary (Bottom Right)
        plt.subplot(2, 2, 4)
        plt.axis('off')
        weather = evaluation_results['weather']
        circuit_stats = evaluation_results['circuit_stats']
        strategy_text = [
            "Strategy Summary:",
            f"• Recommended Stops: {evaluation_results['recommendation']['recommended_stops']}",
            f"• Weather: {'Wet' if weather.get('rainfall', False) else 'Dry'}",
            f"• Track Temp: {weather.get('track_temp', 'N/A'):.1f}°C",
            f"• Available Compounds: {', '.join(sorted(circuit_stats['compounds_used']))}",
            f"• Pit Loss Time: {track_chars.pit_loss_time:.1f}s",
            "\nStrategy Notes:"
        ] + [f"• {note}" for note in evaluation_results['recommendation']['strategy_notes'][:5]]
        plt.text(0, 1, '\n'.join(strategy_text), 
                va='top', ha='left', wrap=True,
                fontsize=9)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def evaluate_circuit_strategy(circuit_name: str, year: int = 2023):
    """Evaluate strategy for a specific circuit."""
    evaluator = StrategyEvaluator()
    
    print(f"\nAnalyzing {circuit_name} {year} Strategy...")
    print("=" * 50)
    
    try:
        results = evaluator.evaluate_circuit_strategy(year, circuit_name)
        recommendation = results['recommendation']
        circuit_stats = results['circuit_stats']
        weather = results['weather']
        track_info = results['track_info']
        
        # Print Circuit Information
        print("\nCircuit Information:")
        print(f"Description: {track_info['description']}")
        print("\nKey Factors:")
        for factor in track_info['key_factors']:
            print(f"- {factor}")
        print(f"\nStrategy Priority: {track_info['strategy_priority']}")
        
        # Print Circuit Characteristics
        print("\nCircuit Characteristics:")
        track_chars = recommendation['track_characteristics']
        print(f"Track Type: {track_chars.track_type.replace('_', ' ').title()}")
        print(f"Track Length: {track_chars.track_length:.3f} km")
        print(f"Pit Loss Time: {track_chars.pit_loss_time:.1f} seconds")
        print(f"Overtaking Difficulty: {track_chars.overtaking_difficulty:.1%}")
        print(f"Track Evolution Rate: {track_chars.track_evolution:.1%}")
        print(f"Base Safety Car Probability: {track_chars.safety_car_probability:.1%}")
        
        # Print Strategy Recommendation
        print("\nStrategy Recommendation:")
        print(f"Recommended Stops: {recommendation['recommended_stops']}")
        print("\nStop Windows:")
        for window in recommendation['stop_windows']:
            print(f"  {window['start']:.1f}km - {window['optimal']:.1f}km - {window['end']:.1f}km")
        
        print("\nStrategy Notes:")
        for note in recommendation['strategy_notes']:
            print(f"- {note}")
        
        # Print Circuit Statistics
        print("\nCircuit Statistics:")
        print(f"Weather Condition: {'Wet' if weather.get('rainfall', False) else 'Dry'}")
        print(f"Track Temperature: {weather.get('track_temp', 'N/A')}°C")
        print(f"Historical Median Stops: {circuit_stats['median_stops']:.1f}")
        print(f"Available Compounds: {', '.join(sorted(circuit_stats['compounds_used']))}")
        print(f"Safety Car Deployed: {circuit_stats['safety_car']}")
        
        # Plot visualization
        plot_path = f"circuit_strategy_{circuit_name.lower()}_{year}.png"
        evaluator.plot_strategy_visualization(results, circuit_name, plot_path)
        print(f"\nStrategy visualization saved to: {plot_path}")
        
    except Exception as e:
        print(f"Error analyzing {circuit_name}: {str(e)}")

if __name__ == "__main__":
    # Example usage
    circuits = [
        "Monaco",  # Monaco GP
        "Belgium",  # Belgian GP (Spa)
        "Italy",   # Italian GP (Monza)
        "Great Britain",  # British GP (Silverstone)
        "Australia"      # Australian GP (Melbourne)
    ]
    
    for circuit in circuits:
        evaluate_circuit_strategy(circuit)
