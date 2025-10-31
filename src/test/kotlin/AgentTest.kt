package org.example

import kotlin.test.Test
import org.junit.jupiter.api.BeforeEach
import kotlin.enums.enumEntries
import kotlin.math.exp
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.test.assertFalse

class AgentTest {
    // test constants
    private val actionList = enumEntries<Action>()
    private val standardState = 4
    private val testQValue = -1.0

    // set up agent
    private lateinit var agent: Agent
    private lateinit var populatedAgent: Agent
    private val states = 9
    private val actions = actionList.size

    @BeforeEach
    fun setup() {
        // make an agent
        agent = Agent(
            states, // 3 by 3 world
            actions,
            1.0, // deterministic world needs no learning rate
            0.995,
            1.0,
        )

        // make another agent
        populatedAgent = Agent(
            states, // 3 by 3 world
            actions,
            1.0, // deterministic world needs no learning rate
            0.995,
            1.0,
        )

        // populate the q-table with some values
        populatedAgent.setQValue(4, Action.UP, testQValue)
        populatedAgent.setQValue(4, Action.DOWN, testQValue*2)
        populatedAgent.setQValue(4, Action.LEFT, testQValue*3)
        populatedAgent.setQValue(4, Action.RIGHT, testQValue*-1)
    }

    // --- Unit tests for agent initialization ---

    @Test
    fun testQTableDimensions() {
        assertEquals(states, agent.getQTable().size)
        assertEquals(actions, agent.getQTable().first().size)
    }

    @Test
    fun testEmptyQTableAtInitialization() {
        for (state in 0 until states) {
            for (a in actionList) {
                assertEquals(0.0, agent.getQValue(state, a))
            }
        }
    }

    // --- Unit tests for retrieving and modifying q-table values and params directly ---

    @Test
    fun testGetSetQValue() {
        assertEquals(0.0, agent.getQValue(0, Action.RIGHT))
        agent.setQValue(0, Action.RIGHT, testQValue)
        assertEquals(testQValue, agent.getQValue(0, Action.RIGHT))
    }

    @Test
    fun testReset() {
        populatedAgent.reset()
        assertEquals(0.0, populatedAgent.getQValue(standardState, Action.RIGHT))
    }

    @Test
    fun testSingleDecay() {
        val testDecayRate = 0.5
        val originalEpsilon = agent.epsilon
        agent.decayEpsilon(testDecayRate, 0.1)
        assertEquals(originalEpsilon*testDecayRate, agent.epsilon)
    }

    @Test
    fun testDecayPastMin() {
        val testDecayRate = 0.5
        val minEpsilon = 0.75
        agent.decayEpsilon(testDecayRate, minEpsilon)
        assertEquals(minEpsilon, agent.epsilon)
    }

    // --- Unit tests for action selection ---

    @Test
    fun testFullExploration() {
        val selected = mutableListOf<Action>()
        for (i in 0 until 100) {
            selected.add(populatedAgent.selectAction(standardState, actionList))
        }
        assertTrue(selected.containsAll(actionList))
    }

    @Test
    fun testFullExploitation() {
        populatedAgent.decayEpsilon(exp(-12.0), 0.0)
        val selected = mutableListOf<Action>()
        for (i in 0 until 100) {
            selected.add(populatedAgent.selectAction(standardState, actionList))
        }
        assertTrue(selected.contains(Action.RIGHT))
        assertFalse(selected.contains(Action.LEFT))
        assertFalse(selected.contains(Action.UP))
        assertFalse(selected.contains(Action.DOWN))
    }

    // --- Unit tests for q-learning update rule ---

    @Test
    fun testMinAlpha() {
        // make an agent with a = 0; refuses to learn new info
        agent = Agent(
            states, // 3 by 3 world
            actions,
            0.0,
            0.0,
            1.0,
        )
        val currentQValue = agent.getQValue(0, Action.RIGHT)

        // take an action
        agent.updateQValue(
            0,
            Action.RIGHT,
            testQValue,
            1,
            listOf(Action.RIGHT, Action.DOWN, Action.LEFT),
            false,
        )

        // q-value should not change
        assertEquals(currentQValue, agent.getQValue(0, Action.RIGHT))
    }

    @Test
    fun testMaxAlpha() {
        // make an agent with a = 1; immediately discards old info
        agent = Agent(
            states, // 3 by 3 world
            actions,
            1.0,
            0.0,
            1.0,
        )

        // take an action
        agent.updateQValue(
            0,
            Action.RIGHT,
            testQValue,
            1,
            listOf(Action.RIGHT, Action.DOWN, Action.LEFT),
            false,
        )

        // q-value should completely update to the new reward
        assertEquals(testQValue, agent.getQValue(0, Action.RIGHT))
    }

}