package org.example

data class CellIndex(val row: Int, val col: Int)

data class Result(val reward: CellCost, val done: Boolean)

data class Episode(val id: Int,
                   val epsilon: Double,
                   val steps: List<Pair<CellIndex, Action>>,
                   val totalCost: Double)

enum class CellCost(val cost: Double, val color: Triple<Int, Int, Int>) {
    EMPTY(-1.0, Triple(255, 255, 255)),
    HAZARD(-5.0, Triple(255, 0, 0)),
    ILLEGALMOVE(-5.0, Triple(0, 0, 0)),
    OBSTACLE(Double.NaN, Triple(0, 0, 0)),
    GOAL(+100.0, Triple(0, 255, 0)),
}

enum class Action(val dy: Int, val dx: Int) {
    DOWN(1, 0),
    UP(-1, 0),
    LEFT(0, -1),
    RIGHT(0, 1),
}