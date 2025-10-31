package org.example

import kotlin.random.Random

/**
 * Represents a 2D warehouse environment for Q-learning.
 * States are grid coordinates (row, col).
 * Actions are discrete directions.
 *
 * @property width the width (# cols) of the warehouse
 * @property height the height (# rows) of the warehouse
 * @property start the initial state of the agent
 * @property goal the rewarding terminal state of the agent
 * @property obstacles a list of obstacle locations
 * @property hazards a list of hazard locations
 * @property slippageRate the probability of incorrect action execution. If zero, the model is deterministic.
 */
class WarehouseEnv(
    val width: Int,
    val height: Int,
    val start: CellIndex,
    val goal: CellIndex,
    val obstacles: List<CellIndex>,
    val hazards: List<CellIndex>,
    val slippageRate: Double,) {
    // critical values: what is the reward field of the warehouse, and where is the agent
    private val grid = Array(height) { Array(width) { CellCost.EMPTY } }
    var agentState = start
        private set

    // initialize with agent positional requirements and obstacle/hazard placement
    init {
        // requirements
        require(start !in obstacles && start !in hazards && start != goal) {
            "Start cell cannot overlap with obstacles, hazards, or goal."
        }
        require(goal !in obstacles && goal !in hazards) {
            "Goal cell cannot overlap with obstacles or hazards"
        }

        // build warehouse
        for (cell in obstacles) { set(cell, CellCost.OBSTACLE) }
        for (cell in hazards) { set(cell, CellCost.HAZARD) }
        set(goal, CellCost.GOAL)
    }

    /**
     * Convert a state's linear index in the Q-Table to a CellIndex in the warehouse
     * @param index the linear index of a state
     * @return the cell index of that state
     */
    fun linearToGridIndex(index: Int): CellIndex {
        return CellIndex(
            row=index / width,
            col=index % width,
        )
    }

    /**
     * Convert a state's CellIndex in the warehouse to alinear index in the Q-Table
     * @param cellindex the cell index of that state
     * @return the linear index of a state
     */
    fun gridToLinearIndex(cellindex: CellIndex): Int {
        return width*cellindex.row + cellindex.col
    }

    /**
     * Given a state, return a list of valid actions
     * @param state the given state
     * @return a list of valid actions
     */
    fun getValidActions(state: CellIndex): List<Action> {
        return Action.entries.filter {
            val next = CellIndex(state.row + it.dy, state.col + it.dx)
            isValidState(next)
        }
    }

    /**
     * Take an action and transition to the next state
     * If [slippageRate] is nonzero, the model is stochastic and the agent may take a different action.
     * @param action an integer representing the chosen move
     * @return pair of next state and result of state-action pair
     */
    fun agentStep(action: Action): Pair<CellIndex, Result> {
        val r = Random.nextDouble()
        var actualAction = action
        // some probability that the agent slips and moves incorrectly
        if (r < slippageRate) {
            actualAction = Action.entries. filter {it != action}.random()
        }
        val nextState = CellIndex(agentState.row + actualAction.dy, agentState.col + actualAction.dx)
        if (isValidState(nextState)) {
            agentState = nextState
            return Pair(nextState, getReward(nextState))
        }
        return Pair(agentState, Result(CellCost.ILLEGALMOVE, false))
    }

    /**
     * Return the agent to its starting state
     */
    fun reset() {
        agentState = start
    }

    // private methods

    /**
     * Get a value from the grid
     * PUBLIC FOR TESTING
     * @param cell the index of the cell to access
     * @return the value in the cell
     */
    fun get(cell: CellIndex): CellCost {
        return grid[cell.row][cell.col]
    }

    /**
     * Set a value in the grid
     * @param cell the index of the cell to edit
     * @param value the celltype to apply
     */
    private fun set(cell: CellIndex, value: CellCost) {
        grid[cell.row][cell.col] = value
    }

    /**
     * Check if a cell is a valid state; i.e. not out-of-bounds or an obstacle
     * @param cell the index of the cell to evaluate
     * @return whether the cell is a valid state
     */
    private fun isValidState(cell: CellIndex): Boolean {
        val withinBounds = cell.row in 0 until height && cell.col in 0 until width
        if (!withinBounds) return false
        return get(cell) != CellCost.OBSTACLE
    }

    /**
     * Get the reward and done flag for a given state.
     */
    private fun getReward(state: CellIndex): Result {
        return Result(get(state), state == goal)
    }
}