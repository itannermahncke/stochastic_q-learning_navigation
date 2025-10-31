package org.example

import kotlin.math.max
import kotlin.random.Random

class Agent(
    states: Int,
    actions: Int,
    val alpha: Double,
    val gamma: Double,
    var epsilon: Double,
) {
    private var qTable = Array(states) { Array(actions) { 0.0 } }

    /**
     * Get the entire q-table
     */
    fun getQTable(): Array<Array<Double>> {
        return qTable
    }

    /**
     * Reset the agent's q-table.
     */
    fun reset() {
        for (row in 0 until qTable.size) {
            for (col in 0 until qTable[row].size) {
                qTable[row][col] = 0.0
            }
        }
    }

    /**
     * Decay the epsilon value. Should occur once per episode.
     * Higher values encourage exploration; lower values encourage exploitation.
     * @param epsilonDecayRate the rate by which epsilon decays
     * @param minEpsilon the minimum acceptable epsilon. used to prevent killing exploration.
     */
    fun decayEpsilon(epsilonDecayRate: Double, minEpsilon: Double = 0.05) {
        epsilon = max(minEpsilon, epsilon * epsilonDecayRate)
    }

    /**
     * Select an action from the available choices with an epsilon-greedy algorithm.
     * @param state the state the agent is in; used for exploitation only
     * @param validActions the available action choices
     * @return the selected action
     */
    fun selectAction(state: Int, validActions: List<Action>): Action {
        val r = Random.nextDouble()
        // exploration
        return if (r < epsilon) {
            validActions.random()
        // exploitation
        } else {
            getBestAction(state, validActions)
        }
    }

    /**
     * Using the q-learning update rule, update the q-value of a given state and action using immediate and future rewards. Store the result directly in the q-table rather than returning it.
     *
     * @param priorState the state the agent was in when it took an action
     * @param actionTaken the action the agent took to get the reward
     * @param reward the resultant cost of the state-action pair
     * @param newState the resultant state of the state-action pair
     * @param newStateValidActions possible actions that can be taken from [newState]
     * @param done whether the agent has reached the terminal state
     */
    fun updateQValue(
        priorState: Int,
        actionTaken: Action,
        reward: Double,
        newState: Int,
        newStateValidActions: List<Action>,
        done: Boolean,
    ) {
        // current q-value
        val currentQ = getQValue(priorState, actionTaken)

        // new q-value depends on whether there is a future state to consider
        val newQ: Double
        if (!done) {
            val maxFutureQValue = getQValue(newState, getBestAction(newState, newStateValidActions))
            newQ = reward + gamma*maxFutureQValue
        } else {
            newQ = reward
        }

        // weight by alpha and store in table
        val newQValue = (1.0 - alpha) * currentQ + alpha * newQ
        setQValue(priorState, actionTaken, newQValue)
    }

    // private methods

    /**
     * Get the q-value of a state-action pair from the q-table
     * PUBLIC FOR TESTING ONLY
     * @param state
     * @param action
     * @return q-value
     */
    fun getQValue(state: Int, action: Action): Double {
        return qTable[state][action.ordinal]
    }

    /**
     * Directly update a value in the q-table.
     * PUBLIC FOR TESTING ONLY
     * @param state
     * @param action
     * @param qValue
     */
    fun setQValue(state: Int, action: Action, qValue: Double) {
        qTable[state][action.ordinal] = qValue
    }

    /**
     * Given a state and list of valid actions, return the action with the best q-value.
     * @param state any state
     * @param actions valid actions to take from that state
     * @return the action in [actions] with the highest q-value for [state]
     */
    private fun getBestAction(state: Int, actions: List<Action>): Action {
        return actions.maxBy { getQValue(state, it) }
    }
}