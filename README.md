# RL FrozenLake IFT7201

A modular reinforcement learning sandbox built around the FrozenLake environment.

## Installation
```bash
uv install
source .venv/bin/activate
```

## Usage
```bash
python main.py [--env ENV] [--strategy STRATEGY] [--episodes N] [--render] [--plot] [--window N]
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--env` | `random` | Environment key (`random`, `baseline`) |
| `--strategy` | `random` | Strategy key (`random`, `sarsa`) |
| `--episodes` | `10` | Number of training episodes |
| `--render` | `False` | Display the environment visually |
| `--plot` | `False` | Display metrics plots after training |
| `--window` | `100` | Smoothing window size for learning curves |

### Examples
```bash
# Run 1000 episodes with SARSA on the baseline map
python main.py --env baseline --strategy sarsa --episodes 1000

# Run with rendering enabled
python main.py --env baseline --episodes 5 --render

# Run and display metrics (smoothed over 200 episodes)
python main.py --env baseline --strategy sarsa --episodes 5000 --plot --window 200
```

## Code Formatting
```bash
uv run black .
```