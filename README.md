# RL FrozenLake IFT7201

A modular reinforcement learning sandbox built around the FrozenLake environment.

## Installation
```bash
uv install
source .venv/bin/activate
```

## Usage
```bash
python main.py [--env ENV] [--strategy STRATEGY] [--episodes N] [--iterations N] [--render] [--plot] [--window N] [--save-dir DIR]
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--env` | `random` | Environment key (`random`, `baseline`, `slippery`, `corridor`) |
| `--strategy` | `random` | Strategy key (`random`, `sarsa`, `qlearning`) |
| `--episodes` | `10` | Number of training episodes |
| `--iterations` | `1` | Number of independent runs (each with a different seed) |
| `--render` | `False` | Display the environment visually |
| `--plot` | `False` | Save training curves and policy figures |
| `--window` | `200` | Smoothing window size for learning curves |
| `--save-dir` | `figures` | Directory to save figures |

### Examples
```bash
# Run 5000 episodes with SARSA on baseline, 20 iterations, save figures
python main.py --env baseline --strategy sarsa --episodes 5000 --iterations 20 --plot

# Run with Q-Learning on corridor
python main.py --env corridor --strategy qlearning --episodes 5000 --iterations 10 --plot --window 300

# Run with rendering enabled
python main.py --env baseline --episodes 5 --render
```

## Code Formatting
```bash
uv run black .
```
