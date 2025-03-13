# F1 Pit Stop Strategy Optimizer

A Python tool for optimizing Formula 1 pit stop strategies using track characteristics, tire compounds, and FastF1 data integration.

## Features

### Track Analysis
- Comprehensive track data for all F1 circuits:
  - Track length and type (Street, Technical, High Speed, Standard)
  - Race distance and lap count
  - Pit loss time (20.5s - 23.5s)
  - Overtaking difficulty factors

### Strategy Calculation
- Pit stop optimization considering:
  - Tire degradation (0.8x - 1.28x base rate)
  - Track evolution (0.003 - 0.017 per lap)
  - Safety car probabilities (20% - 65%)
  - Traffic impact (0.4 - 0.8 factor)

### Tire Management
- Compound-specific strategies:
  - Performance vs. longevity trade-offs
  - Track-specific recommendations
  - Position-based adjustments
  - Real-time wear analysis

### FastF1 Integration
- Live data integration for:
  - Current season (2024)
  - Key reference races (2023)
  - Weather conditions
  - Track status

## Code Structure

```
f1_pitstop_optimizer/
├── src/
│   └── models/
│       ├── __init__.py
│       └── track_strategy.py    # Core strategy logic
│           ├── TrackCharacteristics  # Track parameters
│           ├── StopWindow           # Pit window data
│           └── TrackStrategyOptimizer # Main optimizer
├── cache/                      # FastF1 data cache
│   ├── 2024/                  # Current season
│   └── 2023/                  # Reference races
└── requirements.txt           # Dependencies
```

### Core Components

#### TrackCharacteristics
- Manages track-specific parameters:
  - Base characteristics (length, type)
  - Dynamic factors (degradation, evolution)
  - Race conditions (safety car, traffic)

#### StopWindow
- Defines pit stop opportunities:
  - Start/end lap ranges
  - Optimal pit lap
  - Compound recommendations

#### TrackStrategyOptimizer
- Handles strategy calculations:
  - Track configuration loading
  - Real-time data integration
  - Stop timing optimization
  - Compound selection

## Installation

1. Clone the repository:
```bash
git clone https://github.com/0xTobix0/F1-pitstop-analysis.git
cd f1_pitstop_optimizer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the optimizer:
```bash
python -m src.models.track_strategy
```

### Input Parameters

1. Track Selection:
   - Full name (e.g., "Monaco Grand Prix")
   - Common name (e.g., "monaco", "spa")
   - Track key (e.g., "monaco_gp")

2. Race Situation:
   - Current position (1-20)
   - Current lap number
   - Tire compound and age
   - Previous pit stop history

### Output Information

1. Track Configuration:
   - Basic parameters (length, type)
   - Dynamic characteristics
   - Current conditions

2. Strategy Recommendations:
   - Optimal pit stop count
   - Detailed pit windows
   - Compound choices
   - Key strategy points

## Example Output

```
Dutch Grand Prix Configuration:
----------------------------------------
Track Length: 4.259 km
Track Type: Technical
Maximum Pit Stops: 2
Pit Loss Time: 22.0 seconds

Track Characteristics:
- Tire Degradation: 1.25x
- Track Evolution: 0.013
- Safety Car Probability: 40.00%
- Traffic Impact: 0.70
- Overtaking Difficulty: 0.80
- Pit Window Margin: 4 laps

Recommended Strategy:
----------------------------------------
Current Tires: Medium (5 laps old)
Pit Stops Made: 1
Previous Stops: Lap 20

Optimal Number of Remaining Stops: 1

Stop 1 of 1:
- Window: Lap 43 - 53
- Optimal Lap: 48
- Compound: medium

Key Strategy Points:
- Manage tires in high-load corners
- Keep gaps under 20s for safety car
```

## Dependencies

- Python 3.8+
- FastF1 >= 3.0.0
- NumPy >= 1.24.0
- Pandas >= 2.0.0

## Contributing

Feel free to open issues or submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
