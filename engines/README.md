Engines
===

Welcome to the Engines folder of the Gem Stone Game Repository! This directory contains implementations of various game engines, some powered by traditional algorithms and others by advanced reinforcement learning techniques. Below, you'll find an overview of the engines available in this directory.

Overview
---

1. **Basic Algorithmic Engines:**
These engines are powered by traditional algorithms designed to play the Gem Stone game. They offer a range of strategies, from simple heuristic-based approaches to more complex decision-making algorithms.
  
2. **Reinforcement Learning (RL) Powered Engines:**
These engines utilize reinforcement learning techniques, including temporal difference learning, to play the Gem Stone game. They learn from experience, gradually improving their performance through interaction with the game environment.

Engine ideas
---

| Name | Idea |
|-----------------|-----------------|
| Count | Always do that move which results in the most stones on your side |
| Most |  Always do that move which starts with the most stones |
| Random | Take a random move |
| Steal | Always do that move which results in the enemy having the least stones (corresponds to Count) |
| TD | Use an MLP trained using temporal difference learning similar to [TD-Gammon](https://dl.acm.org/doi/10.1145/203330.203343) |

1. **SimpleHeuristicEngine:**
This engine employs basic heuristics to make decisions during gameplay. It serves as a baseline for comparing more sophisticated strategies.
  
2. **MinimaxEngine:**
Utilizes the Minimax algorithm with alpha-beta pruning to search for optimal moves. It explores the game tree to make strategic decisions.
  
3. **MonteCarloTreeSearchEngine:**
Implements the Monte Carlo Tree Search algorithm to simulate gameplay and select moves based on statistical analysis.
  
4. **DeepQLearningEngine:**
Employs deep Q-learning, a form of reinforcement learning, to train a neural network to play the Gem Stone game. It learns from rewards and penalties received during gameplay.
  
5. **TDGammonEngine:**
Inspired by TD-Gammon, this engine uses temporal difference learning to update its value function and improve its performance over time.

How to Play
---

1. **Selecting Engines:**
    Choose the engine you want to use for playing against or analyzing the Gem Stone game. You can select engines based on their characteristics, complexity, or learning approach.
  
2. **Integration with Game Scripts:**
    Integrate selected engines with game scripts or player scripts to play against them or observe their performance. Follow the instructions provided within the respective scripts to set up engine interactions.
