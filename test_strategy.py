"""
Test script for track strategy analysis.
"""

from src.models.fastf1_loader import FastF1DataLoader
from src.models.track_strategy import TrackStrategyOptimizer
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict

def analyze_monaco_strategy():
    """Analyze pit stop strategy for Monaco GP."""
    print("Analyzing Monaco 2023 Strategy...")
    
    # Initialize components
    loader = FastF1DataLoader()
    session_data = loader.load_session(2023, 'monaco', 'R')
    optimizer = TrackStrategyOptimizer('monaco', 2023)
    
    # Update track characteristics with real data
    optimizer.update_characteristics(
        tire_data=session_data['tire_performance'],
        weather_data=session_data['weather']
    )
    
    # Get strategy recommendations
    strategy = optimizer.recommend_strategy(
        race_distance=session_data['race_distance'],
        weather_condition='dry'
    )
    
    print("\nTrack Analysis:")
    print(f"Tire Degradation: {optimizer.track_characteristics.tire_degradation:.4f}")
    print(f"Track Evolution: {optimizer.track_characteristics.track_evolution:.4f}")
    print(f"Safety Car Probability: {optimizer.track_characteristics.safety_car_probability:.2f}")
    print(f"Traffic Impact: {optimizer.track_characteristics.traffic_impact:.2f}")
    
    print("\nTire Performance:")
    for compound, data in session_data['tire_performance']['compound_performance'].items():
        print(f"\n{compound}:")
        print(f"  Average Lap Time: {data['avg_lap_time']:.2f}s")
        print(f"  Fastest Lap: {data['fastest_lap']:.2f}s")
        print(f"  Average Tire Life: {data['avg_tire_life']:.1f} laps")
    
    print("\nStrategy Recommendation:")
    print(f"Recommended Stops: {strategy['recommended_stops']} (Maximum: 2)")
    print("\nStop Windows (±2 laps to minimize traffic):")
    for window in strategy['stop_windows']:
        print(f"  {window['start']:.1f}km - {window['optimal']:.1f}km - {window['end']:.1f}km")
    
    print("\nStrategy Notes:")
    for note in strategy['strategy_notes']:
        print(f"- {note}")
    
    # Plot tire performance
    plt.figure(figsize=(10, 6))
    compounds = []
    avg_times = []
    fastest_times = []
    
    for compound, data in session_data['tire_performance']['compound_performance'].items():
        compounds.append(compound)
        avg_times.append(data['avg_lap_time'])
        fastest_times.append(data['fastest_lap'])
    
    x = range(len(compounds))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], avg_times, width, label='Average Lap Time', color='skyblue')
    plt.bar([i + width/2 for i in x], fastest_times, width, label='Fastest Lap', color='lightgreen')
    
    plt.xlabel('Tire Compound')
    plt.ylabel('Lap Time (seconds)')
    plt.title('Tire Performance Analysis - Monaco 2023')
    plt.xticks(x, compounds)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('tire_analysis.png')
    plt.close()

def analyze_track_strategy(year: int, track_name: str, weather_override: Dict = None):
    """Analyze strategy for a specific track with optional weather override.
    
    Args:
        year: Season year
        track_name: Name of the track
        weather_override: Optional weather data to override
    """
    print(f"\nAnalyzing {track_name} {year} Strategy...")
    
    # Initialize components
    loader = FastF1DataLoader()
    
    # Map track names to FastF1 format
    track_mapping = {
        'spa': 'Belgian Grand Prix',
        'australia': 'Australian Grand Prix',
        'monaco': 'Monaco Grand Prix'
    }
    fastf1_track = track_mapping.get(track_name.lower(), track_name)
    
    session_data = loader.load_session(year, fastf1_track, 'R')
    
    # Override weather if specified
    if weather_override:
        session_data['weather'].update(weather_override)
        print(f"\nWeather Conditions: {weather_override}")
    
    # Initialize strategy optimizer
    optimizer = TrackStrategyOptimizer(track_name.lower(), year)
    
    # Update track characteristics
    optimizer.update_characteristics(
        tire_data=session_data['tire_performance'],
        weather_data=session_data['weather']
    )
    
    # Print track analysis
    print("\nTrack Analysis:")
    print(f"Tire Degradation: {optimizer.track_characteristics.tire_degradation:.4f}")
    print(f"Track Evolution: {optimizer.track_characteristics.track_evolution:.4f}")
    print(f"Safety Car Probability: {optimizer.track_characteristics.safety_car_probability:.2f}")
    print(f"Traffic Impact: {optimizer.track_characteristics.traffic_impact:.2f}")
    
    # Print tire performance
    print("\nTire Performance:\n")
    for compound, data in session_data['tire_performance']['compound_performance'].items():
        print(f"{compound}:")
        print(f"  Average Lap Time: {data['avg_lap_time']:.2f}s")
        print(f"  Fastest Lap: {data['fastest_lap']:.2f}s")
        print(f"  Average Tire Life: {data['avg_tire_life']:.1f} laps\n")
    
    # Get strategy recommendation
    strategy = optimizer.recommend_strategy(
        race_distance=session_data['race_distance'],
        weather_condition='wet' if session_data['weather'].get('rainfall', False) else 'dry'
    )
    
    # Print strategy
    print("Strategy Recommendation:")
    print(f"Recommended Stops: {strategy['recommended_stops']}\n")
    print("Stop Windows:")
    for window in strategy['stop_windows']:
        print(f"  {window['start']:.1f}km - {window['optimal']:.1f}km - {window['end']:.1f}km")
    
    print("\nStrategy Notes:")
    for note in strategy['strategy_notes']:
        print(f"- {note}")

def test_data_availability():
    """Test data availability across different years."""
    evaluator = StrategyEvaluator()
    current_year = 2025
    
    print("\nTesting FastF1 data availability:")
    print("=" * 40)
    
    for year in range(2015, current_year + 1):
        try:
            # Try to load Monaco GP data for each year
            results = evaluator.evaluate_circuit_strategy(year, "Monaco")
            print(f"{year}: Data available")
        except Exception as e:
            print(f"{year}: {str(e)}")
            
if __name__ == "__main__":
    analyze_monaco_strategy()
    
    # Analyze Spa with different weather conditions
    
    # Scenario 1: Dry conditions
    print("\n=== Spa: Dry Conditions ===")
    analyze_track_strategy(2023, 'spa', {
        'condition': 'dry',
        'track_temp': 25,
        'humidity': 50,
        'rainfall': False
    })
    
    # Scenario 2: Hot conditions
    print("\n=== Spa: Hot Conditions ===")
    analyze_track_strategy(2023, 'spa', {
        'condition': 'dry',
        'track_temp': 45,
        'humidity': 70,
        'rainfall': False
    })
    
    # Scenario 3: Wet conditions
    print("\n=== Spa: Wet Conditions ===")
    analyze_track_strategy(2023, 'spa', {
        'condition': 'wet',
        'track_temp': 18,
        'humidity': 90,
        'rainfall': True
    })
    
    # Analyze Australian GP with different conditions
    
    # Scenario 1: Typical Melbourne afternoon
    print("\n=== Australian GP: Typical Conditions ===")
    analyze_track_strategy(2023, 'australia', {
        'condition': 'dry',
        'track_temp': 28,
        'humidity': 65,
        'rainfall': False
    })
    
    # Scenario 2: Hot day with high track temperatures
    print("\n=== Australian GP: Hot Conditions ===")
    analyze_track_strategy(2023, 'australia', {
        'condition': 'dry',
        'track_temp': 42,
        'humidity': 55,
        'rainfall': False
    })
    
    # Scenario 3: Melbourne's unpredictable rain
    print("\n=== Australian GP: Wet Conditions ===")
    analyze_track_strategy(2023, 'australia', {
        'condition': 'wet',
        'track_temp': 22,
        'humidity': 85,
        'rainfall': True
    })
    
    test_data_availability()
