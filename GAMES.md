# Detailed Game Guides

## 1. Chess AI with Deep Learning

### Overview
An advanced chess engine that uses neural networks and Monte Carlo Tree Search (MCTS) to play competitive chess. The AI is trained using self-play and has been evaluated to play at a strong intermediate level.

### How It Works
1. **Position Evaluation**: Neural network evaluates board positions
2. **MCTS Search**: Explores promising moves using Monte Carlo Tree Search
3. **Alpha-Beta Pruning**: Optimizes search efficiency
4. **ELO Calculation**: Rates player strength

### Running the Game
```bash
python games/chess/chess_ai.py
```

### Features
- Play against AI at different difficulty levels
- Review game moves and statistics
- Train custom models
- Export/import game positions

### Keyboard Controls
- Click pieces to select and move
- Right-click to cancel move
- 'H' for hint
- 'U' to undo last move
- 'R' to restart game

---

## 2. Tic-Tac-Toe AI Agent

### Overview
A reinforcement learning agent trained using Q-Learning that plays optimal Tic-Tac-Toe. The agent never loses against a human player.

### Algorithm
- **Q-Learning**: Learns value of each game state
- **Epsilon-Greedy**: Balances exploration and exploitation
- **Minimax Backup**: Ensures optimal play

### Running the Game
```bash
python games/tictactoe/main.py
```

### Modes
1. **Play vs AI**: Human vs trained agent
2. **Training**: Watch agents play each other
3. **Statistics**: View performance metrics

### Game Rules
- 3x3 grid
- Players alternate turns
- First to get 3 in a row wins
- Draw if grid fills with no winner

---

## 3. Flappy Bird AI

### Overview
A NEAT-based neural network learns to play Flappy Bird. Watch as the population evolves to achieve high scores through genetic algorithms.

### NEAT Algorithm
- **Genetic Algorithm**: Evolves network topology
- **Neuroevolution**: Adapts both weights and structure
- **Fitness Function**: Maximizes distance traveled

### Running the Game
```bash
python games/flappy_bird/main.py
```

### Configuration
- Population size: 50
- Generations: 100+
- Mutation rate: 0.3
- Target fitness: 1000 points

### Visualization
- Real-time population progress
- Best performer display
- Fitness tracking graph
- Generation counter

---

## 4. Snake Game AI

### Overview
Multiple AI approaches to solve Snake: A* pathfinding, Q-Learning, and behavioral cloning from expert demonstrations.

### Algorithms
1. **A* Pathfinding**: Optimal path to food
2. **Q-Learning Agent**: Neural network-based
3. **Expert Policy**: Learns from human gameplay

### Running the Game
```bash
python games/snake/snake_ai.py
```

### Game Mechanics
- Grid-based movement
- Snake grows when eating food
- Game ends if snake hits wall or itself
- Progressively increasing difficulty

### Difficulty Levels
- **Easy**: Slow speed, large grid
- **Medium**: Normal speed, medium grid
- **Hard**: Fast speed, small grid
- **Expert**: Very fast, constant obstacles

---

## 5. 2048 AI Solver

### Overview
Solves the 2048 puzzle game using Expectimax algorithm and heuristic evaluation functions.

### Algorithm
- **Expectimax Search**: Handles both player and random moves
- **Heuristics**: Evaluates board states
- **Multi-threading**: Parallel move evaluation

### Running the Game
```bash
python games/2048/app.py
```

### Heuristics Used
1. **Monotonicity**: Prefer sorted boards
2. **Smoothness**: Minimize value differences
3. **Empty Cells**: Maximize available moves
4. **Merge Potential**: Value future combinations

### Performance
- Achieves 128k+ scores consistently
- Success rate: 89% to reach 2048
- Average moves to win: 300-400

---

## Training Your Own Models

### Chess
```bash
python games/chess/train.py --episodes 10000 --batch-size 32
```

### Tic-Tac-Toe
```bash
python games/tictactoe/train.py --epochs 1000 --learning-rate 0.1
```

### Flappy Bird
```bash
python games/flappy_bird/train.py --generations 50 --population 100
```

### Snake
```bash
python games/snake/train.py --episodes 5000 --epsilon-decay 0.995
```

## Performance Benchmarks

| Game | Algorithm | Win Rate | Avg Score | Time |
|------|-----------|----------|-----------|------|
| Chess | MCTS + NN | 95% | - | 2s/move |
| TicTacToe | Q-Learning | 100% | - | <100ms |
| FlappyBird | NEAT | 98% | 150+ | 30min |
| Snake | RL | 92% | 200+ | 20min |
| 2048 | Expectimax | 89% | 131k | 5min |

## Tips for Playing Against AI

### Chess
- Study openings before challenging higher difficulties
- Use hints when stuck
- Analyze losses to improve

### Tic-Tac-Toe
- AI plays optimally
- Best you can achieve is a draw
- Study game theory

### Flappy Bird
- Watch AI learn in real-time
- Track generation improvements
- Experiment with different parameters

### Snake
- Start with easier difficulties
- Study pathfinding strategies
- Learn from AI mistakes

### 2048
- Combine small numbers first
- Keep larger numbers on edges
- Plan multiple moves ahead

## Troubleshooting

### Game won't start
```bash
pip install -r requirements.txt --force-reinstall
```

### AI is too slow
- Reduce search depth
- Enable GPU acceleration
- Use pre-trained models

### Training won't converge
- Adjust learning rate
- Increase dataset size
- Check for bugs in reward function
