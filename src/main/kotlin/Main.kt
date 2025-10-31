package org.example

import kotlinx.coroutines.*
import org.knowm.xchart.SwingWrapper
import java.time.Duration
import kotlin.math.abs

/**
 * Wrapper function to simplify making a warehouse.
 * @param simple indicate if 3x3 model or full 9x6 warehouse is desired
 * @param stochasticity the rate at which an agent may fail to execute the correct action
 * @return an explorable warehouse environment
 */
fun generateWarehouseEnv(simple: Boolean, stochasticity: Double): WarehouseEnv {
    if (!simple) {
        // obstacles
        val obstacles = mutableListOf<CellIndex>()
        for (shelf in 1 until 8 step 2) {
            for (row in 1 until 5) {
                obstacles.add(CellIndex(row, shelf))
            }
        }

        // hazards
        val hazards = mutableListOf<CellIndex>()
        for (aisle in 2 until 5 step 2) {
            for (row in 1 until 5) {
                hazards.add(CellIndex(row, aisle))
            }
        }
        hazards.add(CellIndex(0, 1))
        hazards.add(CellIndex(2, 6))

        return WarehouseEnv(
            9,
            6,
            CellIndex(5, 4),
            CellIndex(0, 4),
            obstacles,
            hazards,
            stochasticity,
        )
    } else {
        // obstacles
        val obstacles = mutableListOf<CellIndex>()
        obstacles.add(CellIndex(1, 1))

        // hazards
        val hazards = mutableListOf<CellIndex>()
        hazards.add(CellIndex(1, 0))

        return WarehouseEnv(
            3,
            3,
            CellIndex(0, 0),
            CellIndex(2, 2),
            obstacles,
            hazards,
            0.0,
        )
    }
}

/**
 * Perform a single episode of training.
 * @param episodeID unique identifier for this episode
 * @param agent the agent learning to navigate [env]
 * @param env a prefilled warehouse environment
 * @return an Episode containing ID, epsilon of agent, and steps taken
 */
fun playEpisode(episodeID: Int, agent: Agent, env: WarehouseEnv): Episode {
    // agent situation
    var agentState: CellIndex
    var validActions: List<Action>
    var nextState: CellIndex

    // agent actions
    var nextAction: Action
    var actionReward: CellCost
    var terminate = false

    // explore the warehouse
    val steps = mutableListOf<Pair<CellIndex, Action>>()
    var totalCost = 0.0
    while (!terminate) {
//        println("-- Starting turn ${steps.size + 1} --")
        // update agent values
        agentState = env.agentState
        validActions = env.getValidActions(agentState)

//        println("-> Agent is at ${env.agentState} --")
        // select an action
        nextAction = agent.selectAction(
            env.gridToLinearIndex(agentState),
            validActions,
            )
//        println("-> Agent chose to move ${nextAction.name}")

        // take the action and experience consequences
        val out = env.agentStep(nextAction)
        nextState = out.first
        actionReward = out.second.reward
        terminate = out.second.done
//        println("-> Agent experienced cost ${actionReward.cost} --")
//        println("-> Agent is now at ${env.agentState} --")
//        println("-> Terminal state is $terminate --")

        // learn
        agent.updateQValue(
            env.gridToLinearIndex(agentState),
            nextAction,
            actionReward.cost,
            env.gridToLinearIndex(nextState),
            env.getValidActions(nextState),
            terminate,
        )

        // update episode history
        steps.add(Pair(agentState, nextAction))
        if (!terminate) {
            totalCost += actionReward.cost
        }
    }

    // add terminal step and return episode details
    steps.add(Pair(env.agentState, Action.UP)) // irresponsible...
    return Episode(episodeID, agent.epsilon, steps, totalCost)
}

/**
 * Train an agent in a warehouse environment.
 * @param simpleEnv indicate if 3x3 model or full 9x6 warehouse is desired
 * @param learningRate the weight applied to new q-values opposed to old q-values
 * @param discountFactor the weight applied to actions with future rewards opposed to current rewards
 * @param epsilonDecayRate the rate at which the agent should begin exploiting over exploring
 * @param stochasticity the rate at which an agent may fail to execute the correct action
 * @param episodes the number of desired episodes
 * @return A triple containing the trained agent, the warehouse environment, and the training history
 */
fun train(simpleEnv: Boolean,
          learningRate: Double,
          discountFactor: Double,
          epsilonDecayRate: Double,
          stochasticity: Double,
          episodes: Int,
          ): Triple<Agent, WarehouseEnv, MutableList<Episode>> {
    // assemble the warehouse
    val warehouse = generateWarehouseEnv(simpleEnv, stochasticity)

    // establish the agent
    val agent = Agent(
        (warehouse.width * warehouse.height),
        Action.entries.size,
        learningRate,
        discountFactor,
        1.0
    )
    val history = mutableListOf<Episode>()
    for (i in 0 until episodes) {
        warehouse.reset()
        history.add(playEpisode(i, agent, warehouse))
        agent.decayEpsilon(
            epsilonDecayRate,
            0.1,
        )
    }
    return Triple(agent, warehouse, history)
}

/**
 * Find the episode at which the agent converged to the optimal solution.
 * @param history the agent's history of training
 * @param optimalSteps # of steps of optimal solution //TODO: derive from warehouse with BFS
 * @param tolerance max difference between step size average and [optimalSteps] to consider converged
 * @param window # of consecutive episodes within tolerance required consider converged
 * @return episode ID at moment of convergence, or null if no convergence occurred
 */
fun findConvergenceEpisode(
    history: List<Episode>,
    optimalSteps: Int,
    tolerance: Int,
    window: Int,
): Int? {
    for (i in window - 1 until history.size) {
        val windowEpisodes = history.subList(i - window + 1, i + 1)
        if (windowEpisodes.all { abs(it.steps.size - optimalSteps) <= tolerance }) {
            return history[i].id
        }
    }
    return null
}


fun main() {
    // to make visuals
    val viz = Visualizer()
    println()

    // --- Q-LEARNING IN ACTION ---

    // -> Test 1: Deterministic World
//    val out = train(
//        false,
//        1.0,
//        0.99,
//        0.995,
//        0.0,
//        1000,
//    )
//    val warehouse = out.second
//    val history = out.third
//    val convergenceEpisode = findConvergenceEpisode(history, 11, 4, 20)
//    val historyCostChart = viz.plotHistoryStats(history, 10, true, warehouse.slippageRate > 0, convergenceEpisode, { -it.totalCost }, "Incurred Cost")
//    SwingWrapper(historyCostChart).displayChart()
//    val historyEpsilonChart = viz.plotHistoryStats(history, 10, false, warehouse.slippageRate > 0, null, { it.epsilon }, "Rate of Exploration")
//    SwingWrapper(historyEpsilonChart).displayChart()
//
    // -> Test 2: Stochastic World
    val out = train(
        false,
        0.1,
        0.99,
        0.995,
        0.3,
        500,
    )
    val warehouse = out.second
    val history = out.third
    val convergenceEpisode = findConvergenceEpisode(history, 11, 4, 10)
    val historyCostChart = viz.plotHistoryStats(history, 10, true, warehouse.slippageRate > 0, convergenceEpisode, { -it.totalCost }, "Incurred Cost")
    SwingWrapper(historyCostChart).displayChart()
    val historyEpsilonChart = viz.plotHistoryStats(history, 10, false, warehouse.slippageRate > 0, null, { it.epsilon }, "Rate of Exploration")
    SwingWrapper(historyEpsilonChart).displayChart()

    // -> Test 3: Learning Rate Sweep in Stochastic World
//    val alphaConvergence = mutableMapOf<Double, Double>()
//    for (i in 1..10) { // alpha values
//        val alpha = i.toDouble() / 10.0
//        val convergences = mutableListOf<Int>()
//        for (j in 0 until 10) { // averaging convergences
//            val out = train(
//                false,
//                alpha,
//                0.99,
//                0.995,
//                0.3,
//                1000,
//            )
//            val history = out.third
//            val convergenceEpisode = findConvergenceEpisode(history, 11, 4, 10)
//            if (convergenceEpisode != null) {
//                convergences.add(convergenceEpisode)
//            }
//            println("Learning rate $alpha converged at $convergenceEpisode")
//        }
//        alphaConvergence[alpha] = convergences.average()
//    }
//    val alphaConvChart = viz.plotAlphaConvergence(alphaConvergence)
//    SwingWrapper(alphaConvChart).displayChart()

//    viz.renderWarehouseWithAgent(warehouse, warehouse.start)

    // visualize
//    // watch selected episodes
//    runBlocking {
//        while (true) {
//            print("Select an episode to play back (0 - ${history.size - 1}): ")
//            val response = readln().toIntOrNull()
//            if (response != null) {
//                viz.playbackEpisode(warehouse, history[response], Duration.ofMillis(300))
//            } else {
//                break
//            }
//        }
//    }

}