# F1 Pitstop Strategy Optimizer

An advanced Formula 1 race strategy optimization tool that leverages real-time FastF1 data to provide intelligent pit stop and tire management recommendations. The system is particularly optimized for challenging circuits like Monaco, where track position and timing are crucial.

## Features

- **Real-Time Data Integration**
  - Live weather data and track conditions
  - Tire compound performance analysis
  - Enhanced degradation calculations using actual stint data
  - Track evolution metrics
  - Traffic impact analysis
  - Dynamic safety car probability based on conditions

- **Track-Specific Optimizations**
  - Customized strategy for different circuits (Monaco, Monza, Spa, etc.)
  - Track-specific tire degradation modifiers
  - Adaptive traffic impact calculations
  - Circuit-based safety car probabilities
  - Specialized pit window calculations

- **Monaco GP Specific Features**
  - Maximum 2 planned stops due to track position importance
  - Reduced tire degradation impact (20% lower) due to slower speeds
  - Higher traffic impact (0.8) and safety car probability (0.4 base)
  - Tighter pit windows (±2 laps) to minimize traffic interference
  - Special strategy considerations for overcut opportunities

## Project Structure

```
f1_pitstop_optimizer/
├── src/
│   └── models/
│       ├── fastf1_loader.py     # FastF1 data integration
│       ├── track_strategy.py    # Strategy optimization logic
│       └── __init__.py
├── notebooks/
│   └── 04_strategy_test.py     # Main test script
├── cache/                       # FastF1 data cache
├── requirements.txt            # Project dependencies
└── README.md                  # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/0xTobix0/F1-pitstop-analysis.git
cd F1-pitstop-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the strategy test script:
```bash
python test_points_strategy.py
```

The script will:
1. Load historical F1 data (2018-2025) using FastF1
2. Analyze track conditions and tire performance
3. Generate points-optimized pit stop strategies
4. Provide detailed strategy recommendations considering:
   - Weather conditions
   - Track evolution
   - Traffic impact
   - Safety car probability
   - Tire degradation

## Example Output

```
Monaco GP Strategy Analysis
==========================

Track Characteristics:
Tire Degradation: 0.0004 (20% reduced due to slower speeds)
Track Evolution: 0.0030
Traffic Impact: 0.80 (High: 0.8)
Safety Car Probability: 0.60 (Base: 0.4)
Overtaking Difficulty: 0.90

Strategy Recommendation:
Recommended Stops: 2 (Maximum: 2)

Pit Windows (±2 laps to minimize traffic):
  80.1km - 86.8km - 93.4km
  166.9km - 173.5km - 180.2km

Strategy Notes:
- Track position is critical - prioritize clean air
- Consider overcut opportunities due to high track evolution
- Safety car probability is 60.0% - prepare offset strategy
- Two-stop strategy allows for more aggressive tire usage
- Overtaking difficult - track position priority over tire management
```

## Dependencies

- FastF1 (>=3.0.0): Real-time F1 telemetry data
- Pandas (>=1.2.4): Data manipulation and analysis
- NumPy (>=1.19.2): Numerical computations
- Matplotlib (>=3.3.4): Data visualization
- Seaborn (>=0.11.1): Statistical visualizations

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.
