"""Test the points-optimized strategy analyzer."""

from src.models.points_optimizer import PointsOptimizer

def analyze_monaco_strategy():
    """Analyze Monaco-specific points-optimized strategy."""
    optimizer = PointsOptimizer()
    
    # Current conditions for Monaco
    conditions = {
        'wet_race': False,
        'track_temp': 35.0  # Average track temperature
    }
    
    print("\nAnalyzing Monaco Points-Optimized Strategy")
    print("=" * 50)
    
    # Get strategy recommendation
    strategy = optimizer.get_points_optimized_strategy('Monaco', conditions)
    
    if 'error' in strategy:
        print(f"Error: {strategy['error']}")
        return
    
    # Print strategy details
    print("\nStrategy Summary:")
    print(f"Recommended Stops: {strategy['strategy_summary']['recommended_stops']}")
    print(f"Tire Compounds: {' → '.join(strategy['strategy_summary']['tire_compounds'])}")
    print(f"Strategy Confidence: {strategy['strategy_summary']['confidence']}")
    
    print("\nStop Windows:")
    for stop in strategy['stop_windows']:
        window = stop['window']
        print(f"Stop {stop['stop_number']}:")
        print(f"  Optimal Lap: {window['median']}")
        print(f"  Window: {window['min']}-{window['max']} (±{window['std']:.1f} laps)")
    
    print("\nHistorical Insights:")
    print(f"Analyzed Races: {strategy['historical_insights']['analyzed_races']}")
    print(f"Average Points Scored: {strategy['historical_insights']['average_points']}")
    
    print("\nMost Successful Compound Combinations:")
    for combo in strategy['historical_insights']['most_successful_combinations']:
        print(f"- {' → '.join(combo['compounds'])}")
        print(f"  Success Rate: {combo['success_rate']}")
        print(f"  Average Points: {combo['avg_points']}")

if __name__ == "__main__":
    analyze_monaco_strategy()
