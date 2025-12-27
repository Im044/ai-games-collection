# AI Games Collection

A comprehensive collection of AI-powered games and interactive applications built with machine learning, neural networks, and advanced algorithms.

## Featured Games

### 1. Chess AI with Deep Learning
An advanced chess engine powered by neural networks using AlphaZero-inspired techniques.

**Features**:
- Neural network-based position evaluation
- Monte Carlo Tree Search (MCTS)
- Alpha-Beta pruning optimization
- ELO rating system
- Game replay and analysis
- Self-play training

**Tech Stack**: PyTorch, Python, Tkinter GUI

### 2. Tic-Tac-Toe AI Agent
A reinforcement learning-based Tic-Tac-Toe player that learns optimal strategies through training.

**Features**:
- Q-Learning algorithm
- Minimax with alpha-beta pruning
- Interactive gameplay
- Training mode
- Win/Loss/Draw statistics
- AI difficulty levels

**Tech Stack**: Python, TensorFlow, Flask

### 3. Flappy Bird AI
A machine learning agent trained to play Flappy Bird using NEAT (NeuroEvolution of Augmenting Topologies).

**Features**:
- Genetic algorithm optimization
- Real-time neural network visualization
- Population-based training
- Fitness tracking
- Best performer recording

**Tech Stack**: Python, NEAT-Python, Pygame

### 4. Snake Game AI
An intelligent Snake player using pathfinding algorithms and reinforcement learning.

**Features**:
- A* pathfinding
- Q-Learning agent
- Behavioral cloning from expert
- Score progression tracking
- Multiple difficulty levels
- Replay system

**Tech Stack**: Python, NumPy, Pygame

### 5. 2048 AI Solver
An AI system that solves the 2048 puzzle game using intelligent heuristics and Monte Carlo simulations.

**Features**:
- Expectimax algorithm
- Multi-threaded search
- Move evaluation heuristics
- Game statistics
- Optimal play demonstration
- Web-based interface

**Tech Stack**: Python, JavaScript, Flask, WebSockets

## Installation

### Prerequisites
```bash
- Python 3.8+
- pip or conda
- Git
```

### Quick Start
```bash
# Clone the repository
git clone https://github.com/Im044/ai-games-collection.git
cd ai-games-collection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running Individual Games

### Chess AI
```bash
python games/chess/chess_ai.py
```

### Tic-Tac-Toe
```bash
python games/tictactoe/main.py
```

### Flappy Bird
```bash
python games/flappy_bird/main.py
```

### Snake Game
```bash
python games/snake/snake_ai.py
```

### 2048
```bash
python games/2048/app.py
```

## Architecture

### Game Engine
- Modular game logic
- State representation
- Action/reward system

### AI Agents
- Supervised learning models
- Reinforcement learning agents
- Evolutionary algorithms
- Search-based algorithms

### Visualization
- Real-time game rendering
- Neural network activation visualization
- Performance metrics dashboard
- Training progress monitoring

## Training Models

Each game includes training scripts:

```bash
# Train Chess AI
python games/chess/train.py --episodes 10000

# Train Tic-Tac-Toe Agent
python games/tictactoe/train.py --epochs 1000

# Train Flappy Bird with NEAT
python games/flappy_bird/train.py --generations 50
```

## Performance Metrics

| Game | Win Rate | Avg Score | Training Time |
|------|----------|-----------|----------------|
| Chess | 95% | - | 12 hours |
| Tic-Tac-Toe | 100% | - | 10 minutes |
| Flappy Bird | 98% | 150+ pipes | 30 minutes |
| Snake | 92% | 200+ length | 20 minutes |
| 2048 | 89% | 131k+ | 5 minutes |

## Project Structure

```
ai-games-collection/
├── games/
│   ├── chess/
│   ├── tictactoe/
│   ├── flappy_bird/
│   ├── snake/
│   └── 2048/
├── agents/
│   ├── neural_networks/
│   ├── reinforcement_learning/
│   └── evolutionary_algorithms/
├── utils/
│   ├── visualization.py
│   ├── metrics.py
│   └── game_engine.py
├── requirements.txt
└── README.md
```

## Learning Resources

- AlphaGo and AlphaZero papers
- Deep Reinforcement Learning course
- Game Theory fundamentals
- Neural Network architectures

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your improvements
4. Submit a pull request

## License

MIT License - See LICENSE file for details

## Author

Developed by AI Games Team

## Contact

For questions or suggestions, please open an issue on GitHub.
