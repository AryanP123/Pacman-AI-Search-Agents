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

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
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
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    from util import Stack

    fringe = Stack()
    closed = []
    #push first state onto stack and begin
    fringe.push((problem.getStartState(), []))
    while(not fringe.isEmpty()):
        #get top value on stack
        cur = fringe.pop()
        curState = cur[0]
        curPath = cur[1]
        #if current state is goal state, return path up til this point
        if problem.isGoalState(curState):
            return curPath
        #if state has not been expanded before, append it to closed list
        if(curState not in closed):
            closed.append(curState)
            #add all successors that haven't been discovered yet
            successors = problem.getSuccessors(curState)
            i = len(successors) - 1
            while i >= 0:
                if successors[i][0] not in closed:
                    nextPath = curPath.copy()
                    nextPath.append(successors[i][1])
                    fringe.push((successors[i][0], nextPath))
                i -= 1

    util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from util import Queue

    fringe = Queue()
    closed = []
    #push first state into queue and begin
    fringe.push((problem.getStartState(), []))
    while(not fringe.isEmpty()):
        #get first value on queue
        cur = fringe.pop()
        curState = cur[0]
        curPath = cur[1]
        #if current state is goal state, return path up til this point
        if problem.isGoalState(curState):
            return curPath
        #if state has not been expanded before, append it to closed list
        if(curState not in closed):
            closed.append(curState)
            #add all successors that haven't been discovered yet
            successors = problem.getSuccessors(curState)
            i = len(successors) - 1
            while i >= 0:
                if successors[i][0] not in closed:
                    nextPath = curPath.copy()
                    nextPath.append(successors[i][1])
                    fringe.push((successors[i][0], nextPath))
                i -= 1

    util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue

    fringe = PriorityQueue()
    closed = []
    #push first state into priority queue and begin
    fringe.push((problem.getStartState(), [], 0), 0)
    while(not fringe.isEmpty()):
        #get first value on queue
        cur = fringe.pop()
        curState = cur[0]
        curPath = cur[1]
        curCost = cur[2]
        #if current state is goal state, return path up til this point
        if problem.isGoalState(curState):
            return curPath
        #if state has not been expanded before, append it to closed list
        if(curState not in closed):
            closed.append(curState)
            #add all successors that haven't been discovered yet or have lower costs
            successors = problem.getSuccessors(curState)
            i = len(successors) - 1
            while i >= 0:
                nextPath = curPath.copy()
                nextPath.append(successors[i][1])
                nextCost = curCost + successors[i][2]
                fringe.update((successors[i][0], nextPath, nextCost), nextCost)
                i -= 1
    
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue

    fringe = PriorityQueue()
    closed = []
    #push first state into priority queue and begin
    fringe.push((problem.getStartState(), [], 0), 0)
    while(not fringe.isEmpty()):
        #get first value on queue
        cur = fringe.pop()
        curState = cur[0]
        curPath = cur[1]
        curCost = cur[2]
        #if current state is goal state, return path up til this point
        if problem.isGoalState(curState):
            return curPath
        #if state has not been expanded before, append it to closed list
        if(curState not in closed):
            closed.append(curState)
            #add all successors that haven't been discovered yet
            successors = problem.getSuccessors(curState)
            i = len(successors) - 1
            while i >= 0:
                nextPath = curPath.copy()
                nextPath.append(successors[i][1])
                pathCost = curCost + successors[i][2]
                queueVal = pathCost + heuristic(successors[i][0], problem)
                fringe.update((successors[i][0], nextPath, pathCost), queueVal)
                i -= 1
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
