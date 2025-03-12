# F1 Pit Stop Strategy Optimizer

A Python tool for optimizing Formula 1 pit stop strategies based on track characteristics, tire compounds, and race conditions.

## Features

- Track-specific strategy generation considering:
  - Tire degradation rates (ranging from 0.8x to 1.28x base rate)
  - Track evolution (0.003 to 0.017 per lap)
  - Safety car probabilities (20% to 65%)
  - Overtaking difficulty (based on track type)
  - Traffic impact (0.4 to 0.8 factor)
  - Weather conditions

- Tire compound optimization based on:
  - Compound lifespan
  - Performance advantage
  - Track characteristics
  - Race position
  - Real-time degradation data

- Real-time data integration using FastF1:
  - Live tire wear analysis
  - Weather impact assessment
  - Track evolution tracking
  - Safety car probability adjustment

- Track Types and Characteristics:
  - Standard (e.g., Australian GP, Austrian GP)
  - Technical (e.g., Japanese GP, Spanish GP)
  - High Speed (e.g., Belgian GP, Italian GP)
  - Street (e.g., Monaco GP, Singapore GP)

## Project Structure

```
f1_pitstop_optimizer/
├── src/
│   └── models/
│       ├── __init__.py
│       ├── track_strategy.py    # Main strategy optimization logic
│       └── fastf1_loader.py     # Real-time race data integration
├── cache/                       # FastF1 data cache
│   └── ...                     # Race data cache files
└── requirements.txt            # Project dependencies
```

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

Run the strategy optimizer:
```bash
python -m src.models.track_strategy
```

The tool will:
1. Display available F1 tracks with:
   - Track name
   - Number of laps
   - Track length
   - Track type

2. Accept track input by:
   - Full name (e.g., "Monaco Grand Prix")
   - Common name (e.g., "monaco", "spa", "monza")
   - Track key (e.g., "monaco_gp")

3. Show track characteristics:
   - Basic info (length, type, pit loss time)
   - Tire degradation factor
   - Track evolution rate
   - Safety car probability
   - Traffic impact
   - Overtaking difficulty
   - Pit window margins

4. Take race inputs:
   - Current position
   - Current lap
   - Tire compound
   - Tire age
   - Previous pit stops

5. Generate optimized strategy:
   - Recommended number of stops
   - Pit windows with optimal laps
   - Compound recommendations
   - Key strategy points

## Dependencies

- Python 3.8+
- NumPy >= 1.24.0
- FastF1 >= 3.0.0
- Pandas >= 2.0.0

## Example Output

```
Available F1 Tracks:
-----------------------------------------------------------------
Track                               Laps   Length   Type
-----------------------------------------------------------------
Australian Grand Prix               58     5.278    Standard
Chinese Grand Prix                  56     5.451    Technical
Japanese Grand Prix                 53     5.807    Technical
Bahrain Grand Prix                 57     5.412    Standard
...

Monaco Grand Prix Configuration:
----------------------------------------
Track Length: 3.337 km
Track Type: Street
Maximum Pit Stops: 1
Pit Loss Time: 23.5 seconds

Track Characteristics:
- Tire Degradation: 0.80x
- Track Evolution: 0.003
- Safety Car Probability: 60.00%
- Traffic Impact: 0.80
- Overtaking Difficulty: 0.80
- Pit Window Margin: 2 laps

Enter Race Details:
--------------------
Current position: 1
Current lap: 5

Current Tire Details:
--------------------
Current tire compound (soft/medium/hard/inter/wet): medium
Current tire age (laps): 5

Previous Pit Stops:
--------------------
Number of pit stops made (0-3): 0

Recommended Strategy:
----------------------------------------
Current Tires: Medium (5 laps old)
Pit Stops Made: 0

Optimal Number of Remaining Stops: 1

Stop 1 of 1:
- Window: Lap 76 - 77
- Optimal Lap: 78
- Compound: hard

Key Strategy Points:
- Priority: Maintain track position
- Track position is everything
- Undercut is very powerful
- Key sector: Casino to Tunnel
- Keep gaps under 20s for safety car
```

## Contributing

Feel free to open issues or submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
