# RL FrozenLake IFT7201

A modular reinforcement learning sandbox built around the FrozenLake environment.

## Installation
```bash
uv install
source .venv/bin/activate
```

## Usage
```bash
python main.py [--env ENV] [--strategy STRATEGY] [--episodes N] [--render]
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--env` | `random` | Environment key |
| `--strategy` | `random` | Strategy key |
| `--episodes` | `10` | Number of episodes to run |
| `--render` | `False` | Display the environment visually |

### Examples
```bash
# Run 100 episodes with default random strategy
python main.py --episodes 100

# Run with rendering enabled
python main.py --env custom8x8 --strategy random --episodes 5 --render

# Run a specific strategy silently
python main.py --env default --strategy random --episodes 500
```

## Code Formatting
```bash
black *.py
```