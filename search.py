# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def expand(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (child,
        action, stepCost), where 'child' is a child to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that child.
        """
        util.raiseNotDefined()

    def getActions(self, state):
        """
          state: Search state

        For a given state, this should return a list of possible actions.
        """
        util.raiseNotDefined()

    def getActionCost(self, state, action, next_state):
        """
          state: Search state
          action: action taken at state.
          next_state: next Search state after taking action.

        For a given state, this should return the cost of the (s, a, s') transition.
        """
        util.raiseNotDefined()

    def getNextState(self, state, action):
        """
          state: Search state
          action: action taken at state

        For a given state, this should return the next state after taking action from state.
        """
        util.raiseNotDefined()

    def getCostOfActionSequence(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions

    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]

# to run autograder just run
# -> python3 autograder.py

# q1: 4/4 grade
# python3 ./pacman.py --layout bigMaze -p searchAgent -z 0.5 --frameTime 0.01
def depthFirstSearch(problem):
    starting_node = problem.getStartState()
    if problem.isGoalState(starting_node):
        return []

    queue = util.Stack()
    visited_nodes = set()

    queue.push((starting_node, []))

    while not queue.isEmpty():
        current_node, actions = queue.pop()
        if current_node in visited_nodes:
            continue
        visited_nodes.add(current_node)
        if problem.isGoalState(current_node):
            return actions

        for next_node, action, _ in problem.expand(current_node):
            if next_node not in visited_nodes:
                new_action = actions + [action]
                queue.push((next_node, new_action))
    return []

# q2: 4/4
# python3 ./pacman.py --layout bigMaze -p searchAgent -a fn=bfs -z 0.5 --frameTime 0.01
def breadthFirstSearch(problem):
    starting_node = problem.getStartState()
    if problem.isGoalState(starting_node):
        return []

    queue = util.Queue()
    visited_nodes = set()
    queue.push((starting_node, []))

    while not queue.isEmpty():
        current_node, actions = queue.pop()
        if current_node in visited_nodes:
            continue

        if problem.isGoalState(current_node):
            return actions

        visited_nodes.add(current_node)
        for next_node, next_action, _ in problem.expand(current_node):
            if next_node not in visited_nodes:
                new_action = actions + [next_action]
                queue.push((next_node, new_action))
    return []


# python3 ./pacman.py --layout bigMaze -p searchAgent -a fn=ucs -z 0.5 --frameTime 0.01
def uniformCostSearch(problem):
    starting_node = problem.getStartState()
    if problem.isGoalState(starting_node):
        return []
    priority_queue = util.PriorityQueue()
    priority_queue.push((starting_node, [], 0), 0)
    cost_so_far = {starting_node: 0}

    while not priority_queue.isEmpty():
        current_node, actions, cost = priority_queue.pop()
        if cost > cost_so_far.get(current_node, float("inf")):
            continue
        if problem.isGoalState(current_node):
            return actions

        for next_node, next_action, next_cost in problem.expand(current_node):
            new_cost = next_cost + cost
            if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                cost_so_far[next_node] = new_cost
                new_actions = actions + [next_action]
                priority_queue.push((next_node, new_actions, new_cost), new_cost)
    return []


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    starting_node = problem.getStartState()
    if problem.isGoalState(starting_node):
        return []
    priority_queue = util.PriorityQueue()
    priority_queue.push((starting_node, [], 0), 0)
    cost_so_far = {starting_node: 0}

    while not priority_queue.isEmpty():
        current_node, actions, cost = priority_queue.pop()
        if cost > cost_so_far.get(current_node, float("inf")):
            continue
        if problem.isGoalState(current_node):
            return actions

        for next_node, next_action, next_cost in problem.expand(current_node):
            new_cost = next_cost + cost
            new_heuristic = new_cost + heuristic(next_node, problem)
            if next_node not in cost_so_far or new_heuristic < cost_so_far[next_node]:
                new_actions = actions + [next_action]
                cost_so_far[next_node] = new_cost
                priority_queue.push((next_node, new_actions, new_cost), new_heuristic)
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
