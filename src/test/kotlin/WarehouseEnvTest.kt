package org.example

import kotlin.test.Test
import org.junit.jupiter.api.BeforeEach
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

class WarehouseEnvTest {
    private lateinit var env: WarehouseEnv
    private val h = 4
    private val w = 3
    private val start = CellIndex(0, 0)
    private val goal = CellIndex(2, 0)
    private val obstacles = listOf(CellIndex(0, 1))
    private val hazards = listOf(CellIndex(1, 0))
    private val slippage = 0.0

    @BeforeEach
    fun setup() {
        env = WarehouseEnv(
            width = w,
            height = h,
            start = start,
            goal = goal,
            obstacles = obstacles,
            hazards = hazards,
            slippage,
        )
    }

    // --- Unit tests for warehouse initialization ---

    @Test
    fun testInvalidInitialization() {
        assertFailsWith<IllegalArgumentException> {
            WarehouseEnv(w, h, start, goal, listOf(start), hazards, slippage)
        }
        assertFailsWith<IllegalArgumentException> {
            WarehouseEnv(w, h, start, goal, obstacles, listOf(start), slippage)
        }
        assertFailsWith<IllegalArgumentException> {
            WarehouseEnv(w, h, start, start, obstacles, hazards, slippage)
        }
    }

    @Test
    fun testSizeInitialization() {
        assertEquals(Pair(h, w), Pair(env.height, env.width))
    }

    @Test
    fun testHazardInitialization() {
        for (i in hazards.indices) {
            assertEquals(CellCost.HAZARD, env.get(hazards[i]))
        }
    }

    @Test
    fun testObstacleInitialization() {
        for (i in obstacles.indices) {
            assertEquals(CellCost.OBSTACLE, env.get(obstacles[i]))
        }
    }

    // --- Unit tests for grid-linear conversions ---

    @Test
    fun testFirstIndex() {
        val first = 0
        val zeroZero = CellIndex(0, 0)
        assertEquals(first, env.gridToLinearIndex(zeroZero))
        assertEquals(zeroZero, env.linearToGridIndex(first))
    }

    @Test
    fun testSpecificIndex() {
        val idx = 4
        val rowCol = CellIndex(1, 1)
        assertEquals(idx, env.gridToLinearIndex(rowCol))
        assertEquals(rowCol, env.linearToGridIndex(idx))
    }

    @Test
    fun testLastIndex() {
        val last = env.height * env.width - 1
        val hW = CellIndex(env.height - 1, env.width - 1)
        assertEquals(last, env.gridToLinearIndex(hW))
        assertEquals(hW, env.linearToGridIndex(last))
    }

    // --- Unit tests for retrieving grid values ---

    @Test
    fun testGet() {
        assertEquals(CellCost.EMPTY, env.get(start))
    }

    @Test
    fun testGetActionsAll() {
        val actions = env.getValidActions(CellIndex(2, 1))
        assertEquals(4, actions.size)
    }

    @Test
    fun testGetActionsEdges() {
        val actions = env.getValidActions(CellIndex(3, 0))
        assertEquals(2, actions.size)
        assertFalse(actions.contains(Action.LEFT))
        assertFalse(actions.contains(Action.DOWN))
    }

    @Test
    fun testGetActionsObstacle() {
        val actions = env.getValidActions(CellIndex(1, 1))
        assertEquals(3, actions.size)
        assertFalse(actions.contains(Action.UP))
    }

    @Test
    fun testGetActionsSurrounded() {
        val actions = env.getValidActions(start)
        assertEquals(1, actions.size)
        assertFalse(actions.contains(Action.RIGHT))
        assertFalse(actions.contains(Action.LEFT))
        assertFalse(actions.contains(Action.UP))
    }

    // --- Unit tests for moving the agent ---

    @Test
    fun testOutOfBoundsMove() {
        val stateReward = env.agentStep(Action.LEFT)
        assertEquals(start, stateReward.first)
        assertEquals(CellCost.ILLEGALMOVE, stateReward.second.reward)
        assertEquals(false, stateReward.second.done)
    }

    @Test
    fun testObstacleMove() {
        val stateReward = env.agentStep(Action.RIGHT)
        assertEquals(start, stateReward.first)
        assertEquals(CellCost.ILLEGALMOVE, stateReward.second.reward)
        assertFalse { stateReward.second.done }
    }

    @Test
    fun testValidMove() {
        val stateReward = env.agentStep(Action.DOWN)
        assertEquals(CellIndex(start.row + 1, start.col), stateReward.first)
        assertEquals(CellCost.HAZARD, stateReward.second.reward)
        assertFalse { stateReward.second.done }
    }

    @Test
    fun testWinningMove() {
        env.agentStep(Action.DOWN)
        val stateReward = env.agentStep(Action.DOWN)
        assertEquals(CellIndex(start.row + 2, start.col), stateReward.first)
        assertEquals(CellCost.GOAL, stateReward.second.reward)
        assertTrue { stateReward.second.done }
    }

    @Test
    fun testSingleMoveReset() {
        env.agentStep(Action.DOWN)
        env.reset()
        assertEquals(start, env.agentState)
    }

    @Test
    fun testMultiMoveReset() {
        env.agentStep(Action.DOWN)
        env.agentStep(Action.DOWN)
        env.reset()
        assertEquals(start, env.agentState)
    }

    // --- Unit tests for stochasticity ---
    @Test
    fun testStochasticMove() {
        env = WarehouseEnv(
            width = w,
            height = h,
            start = start,
            goal = goal,
            obstacles = obstacles,
            hazards = hazards,
            1.0,
        )
        val stateReward = env.agentStep(Action.DOWN)
        assertNotEquals(CellIndex(start.row + 1, start.col), stateReward.first)
    }

}