# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        score = successorGameState.getScore()

        foodList = newFood.asList()
        if len(foodList) > 0:
            i = 0
            closestFood = 1000000
            while i < len(foodList):
                food = foodList[i]
                distance = manhattanDistance(newPos, food)
                if distance < closestFood:
                    closestFood = distance
                i += 1
            score += 1.0 / closestFood

        scared = [ghostState for ghostState in newGhostStates if ghostState.scaredTimer > 0]
        if len(scared) > 0:
            i = 0
            closestGhost = 1000000
            while i < len(scared):
                ghost = scared[i]
                distance = manhattanDistance(newPos, ghost.getPosition())
                if distance < closestGhost:
                    closestGhost = distance
                i += 1
            score += 1.0 / closestGhost

        notScared = [ghostState for ghostState in newGhostStates if ghostState.scaredTimer == 0]
        if len(notScared) > 0:
            i = 0
            closestGhost = 1000000
            while i < len(notScared):
                ghost = notScared[i]
                distance = manhattanDistance(newPos, ghost.getPosition())
                if distance < closestGhost:
                    closestGhost = distance
                i += 1
            if closestGhost == 0:
                score -= 1000
            else:
                score -= 1.0 / closestGhost

        return score



def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        def minValue(gameState, ghost, depth):
            value = 1000000
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            
            # check through all possible actions for ghost[#]--> loop through each action -->

            # if it's the last ghost calculate the minimum value comparing value and pacman's maximum (find min of all max) --> 
            # to get pacman max it gets the gameState after the last ghost takes its action and increases depth by 1 --> ...
            # if it's the last ghost calculate the minimum value comparing value and pacman's maximum (find min of all max) --> 
            # to get pacman max it gets the gameState after the last ghost takes its action and increases depth by 1 --> ...

            # if it's not the last ghost, it generates the successor state after the current ghost takes its action and 
            # calls minValue recursively for the next ghost at the same depth level.
            # This process continues until all ghosts have taken their turns for the current depth level in the game tree.
            # Then it returns the minimum score among all possible sequences of actions for the ghosts at this depth level.
            
            # Find set of moves that will provide the smallest value

            actions = gameState.getLegalActions(ghost)
            action_index = 0
            value = float("inf")

            while action_index < len(actions):
                action = actions[action_index]
                successorGameState = gameState.generateSuccessor(ghost, action)
                if ghost == gameState.getNumAgents() - 1:
                    pacMax = maxValue(successorGameState, depth + 1)
                    #print(pacMax)
                    value = min(value, pacMax)
                else:
                    nextGhost = minValue(successorGameState, ghost + 1, depth)
                    value = min(value, nextGhost)
                    #print(value)
                action_index += 1

            return value


        def maxValue(gameState, depth):
            value = -1000000
            #print(depth)
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            actions = gameState.getLegalActions(0)
            action_index = 0

            # check through all possible actions for pacman--> loop through each action -->

            # find maximum value move from all ghosts possible minimum moves

            while action_index < len(actions):
                action = actions[action_index]
                #print(action)
                successorGameState = gameState.generateSuccessor(0, action)
                ghostMin = minValue(successorGameState, 1, depth)
                value = max(value, ghostMin)
                #print(value)
                action_index += 1

            return value


        maxScore = -1000000

        # check through all possible actions for pacman --> loop through each action --> get the game state of taking that action -->
        # find the min value from the ghosts actions --> ...

        depth, ghost = 0, 1
        actions = gameState.getLegalActions(0)
        index = 0

        while index < len(actions):
            action = actions[index]
            score = minValue(gameState.generateSuccessor(0, action), ghost, depth)
            if score > maxScore:
                maxScore, finalAction = score, action
            index += 1

        if(finalAction): return finalAction
        else: return Directions.STOP
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def minValue(gameState, depth, ghost, a, B):
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return [self.evaluationFunction(gameState), Directions.STOP]

            min_value = 1000000
            move = Directions.STOP
            actions = gameState.getLegalActions(ghost)
            #print(actions)
            action_index = 0

            # loop over all actions for a ghost agent.
            # for each action, create a new game state.
            # if the ghost is the last agent, calculate the maximum value from the new game state.
            # if not, calculate the minimum value for the next ghost.
            # keep track of the action leading to the minimum value.
            # if a lower value is found than the best so far (alpha), stop and return this value and action.
            # update Beta to be the minimum of Beta and the current minimum value.
            # after all actions, return the minimum value and its action.

            while action_index < len(actions):
                action = actions[action_index]
                #print(action)
                successorGameState = gameState.generateSuccessor(ghost, action)
                #print(successorGameState)
                if ghost == gameState.getNumAgents() - 1:
                    value = maxValue(successorGameState, depth + 1, a, B)[0]
                    #print(value)
                    if value < min_value:
                        move = action
                        min_value = value
                        #print(move)
                        #print(min_value)
                    if min_value < a:
                        return [min_value, move]
                    B = min(B, min_value)
                    #print(B)
                else:
                    value = minValue(successorGameState, depth, ghost + 1, a, B)[0]
                    #print(value)
                    if value < min_value:
                        move = action
                        min_value = value
                        #print(move)
                        #print(min_value)
                    if min_value < a:
                        return [min_value, move]
                    B = min(B, min_value)
                    #print(B)
                action_index += 1

            return [min_value, move]


        def maxValue(gameState, depth, a, B):
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return [self.evaluationFunction(gameState), Directions.STOP]

            max_value = -1000000
            move = Directions.STOP
            actions = gameState.getLegalActions(0)
            #print(actions)
            action_index = 0

            # loop over all actions for a player.
            # for each action, create a new game state.
            # calculate the minimum value from the new game state.
            # keep track of the action leading to the maximum value.
            # if a higher value is found than the best so far (Beta), stop and return this value and action.
            # update Alpha to be the maximum of Alpha and the current maximum value.
            # after all actions, return the maximum value and its action.

            while action_index < len(actions):
                action = actions[action_index]
                #print(action)
                successorGameState = gameState.generateSuccessor(0, action)
                value = minValue(successorGameState, depth, 1, a, B)[0]
                #print(value)
                if value > max_value:
                    move = action
                    max_value = value
                    #print(move)
                    #print(max_value)
                    #!!! broken --> DONT FORGET to change this to how it is in minvalue
                if max_value > B:
                    return [max_value, move]
                a = max(a, max_value)
                #print(a)
                action_index += 1

            return [max_value, move]

        return maxValue(gameState, 0, -1000000, 1000000)[1]

        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        def expectedValue(gameState, ghost, depth): #edited minValue function
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            actions = gameState.getLegalActions(ghost)
            action_index = 0
            value = 0
            # Identify all possible actions the ghost can take.
            # Loop Through Actions: For each action:
            # Generate the game state that would result if the ghost took that action.
            # If this is the last ghost, calculate the maximum score the Pacman could get from this state and add it to a running total.
            # If there are more ghosts, recursively call expectedValue for the next ghost and add the result to the running total.
            # After going through all actions, divide the total score by the number of actions to get the average expected score. 
            # This assumes each action is equally likely.
            while action_index < len(actions):
                action = actions[action_index]
                #print(action)
                successorGameState = gameState.generateSuccessor(ghost, action)
                #print(successorGameState)
                if ghost == gameState.getNumAgents() - 1:
                    pacMax = maxValue(successorGameState, depth + 1)
                    #print(pacMax)
                    value += pacMax
                else:
                    nextGhost = expectedValue(successorGameState, ghost + 1, depth)
                    value += nextGhost
                    #print(value)
                action_index += 1

            return value / len(actions) #expected outcome if we assume that each action is equally likely


        def maxValue(gameState, depth):
            value = -1000000
            #print(depth)
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            actions = gameState.getLegalActions(0)
            action_index = 0

            # check through all possible actions for pacman--> loop through each action -->

            # find maximum value move from all ghosts possible minimum moves

            while action_index < len(actions):
                action = actions[action_index]
                #print(action)
                successorGameState = gameState.generateSuccessor(0, action)
                ghostExpected = expectedValue(successorGameState, 1, depth)
                value = max(value, ghostExpected)
                #print(value)
                action_index += 1

            return value


        maxScore = -1000000
        depth, ghost = 0, 1
        actions = gameState.getLegalActions(0)
        index = 0

        while index < len(actions):
            action = actions[index]
            score = expectedValue(gameState.generateSuccessor(0, action), ghost, depth)
            if score > maxScore:
                maxScore, finalAction = score, action
            index += 1

        if(finalAction): return finalAction
        else: return Directions.STOP
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: The main change I made was to remove the line where we generate a successor 
    game state based on an action. In the original function, we were generating a successor 
    game state for a given action using currentGameState.generatePacmanSuccessor(action). 
    But this question we’re evaluating states, not actions so we don’t need to generate a successor
    state we can just evaluate the current game state directly. Instead of using successorGameState, 
    I replaced it with currentGameState.
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    score = currentGameState.getScore()

    foodList = newFood.asList()
    if len(foodList) > 0:
        i = 0
        closestFood = 1000000
        while i < len(foodList):
            food = foodList[i]
            distance = manhattanDistance(newPos, food)
            if distance < closestFood:
                closestFood = distance
            i += 1
        score += 1.0 / closestFood

    scared = [ghostState for ghostState in newGhostStates if ghostState.scaredTimer > 0]
    if len(scared) > 0:
        i = 0
        closestGhost = 1000000
        while i < len(scared):
            ghost = scared[i]
            distance = manhattanDistance(newPos, ghost.getPosition())
            if distance < closestGhost:
                closestGhost = distance
            i += 1
        score += 1.0 / closestGhost

    notScared = [ghostState for ghostState in newGhostStates if ghostState.scaredTimer == 0]
    if len(notScared) > 0:
        i = 0
        closestGhost = 1000000
        while i < len(notScared):
            ghost = notScared[i]
            distance = manhattanDistance(newPos, ghost.getPosition())
            if distance < closestGhost:
                closestGhost = distance
            i += 1
        if closestGhost == 0:
            score -= 1000
        else:
            score -= 1.0 / closestGhost

    return score
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
