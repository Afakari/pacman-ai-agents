# Pacman AI Agents

This repository contains a series of intelligent agents developed for the classic Pacman game as part of an Artificial Intelligence course at university. The project focuses on implementing various search algorithms and strategies to solve different challenges within the Pacman environment.

## Project Overview

The goal of this project is to develop AI agents that can navigate the Pacman game world efficiently, using different search algorithms and techniques. The agents are designed to solve specific problems, such as finding the shortest path, collecting all food, or escaping ghosts.

The project is based on the [Berkeley AI Pacman Project](http://ai.berkeley.edu/search.html), which provides a framework for developing and testing AI agents in the Pacman environment.
> This is the final grading i have achieved.
```
Provisional grades
==================
Question q1: 4/4
Question q2: 4/4
Question q3: 4/4
Question q4: 3/3
Question q5: 3/3
Question q6: 5/4
Question q7: 3/3
------------------
Total: 26/25
```
## Getting Started

### Prerequisites

- Python 3.x >= 3.11
- The Pacman framework provided by the Berkeley AI course.

### Installation

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/afakari/pacman-ai-agents
   ```

2. Navigate to the project directory:
   ```bash
   cd pacman-ai-agents
   ```

3. Ensure you have Python 3.x installed. You can check your Python version by running:
   ```bash
   python --version
   ```

### Running the Project

To run the project, use the following command:

```bash
python pacman.py
```

This will start the Pacman game with the default layout. You can specify different layouts and agents by using command-line arguments. For example:

```bash
python pacman.py -l mediumMaze -p SearchAgent -a fn=bfs
```

This command runs the Pacman game on the `mediumMaze` layout using a `SearchAgent` with the Breadth-First Search (BFS) algorithm.

### Available Agents and Algorithms

The following agents and algorithms are implemented in this project:

- **Depth-First Search (DFS)**
- **Breadth-First Search (BFS)**
- **Uniform Cost Search (UCS)**
- **A* Search**

Each agent can be invoked using the appropriate command-line arguments. For example, to run the A* Search agent, use:

```bash
python pacman.py -l bigMaze -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic
```

### Project Structure

- `search.py`: Contains the implementation of various search algorithms.
- `searchAgents.py`: Contains the implementation of the Pacman agents that use the search algorithms.
- `pacman.py`: The main game file that runs the Pacman game with the specified agent and layout.

## Grading

The project is graded based on the performance of the agents in solving specific problems, the default grading mechanism can be done using:
```angular2html
python autograder.py
```

Each problem is associated with a specific question or task that needs to be solved by the agent. The grading criteria are based on the correctness and efficiency of the implemented algorithms.
> ill add more in depth explain of the questions after i have solved them.

## Contributing

If you would like to contribute to this project, feel free to fork the repository and submit a pull request. Please ensure that your code follows the project's coding standards and includes appropriate documentation.
