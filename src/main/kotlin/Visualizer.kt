package org.example

import kotlinx.coroutines.time.delay
import org.knowm.xchart.CategoryChart
import org.knowm.xchart.CategoryChartBuilder
import java.awt.Color
import org.knowm.xchart.style.Styler
import org.knowm.xchart.style.markers.SeriesMarkers
import java.awt.Font
import java.time.Duration

class Visualizer {

    /**
     * Print a visual representation of the warehouse. Agent location provided.
     * @param env the warehouse environment to visualize
     * @param agentState the location at which to render the agent
     */
    fun renderWarehouseWithAgent(env: WarehouseEnv, agentState: CellIndex) {
        for (r in 0 until env.height) {
            print("     ") // 5 spaces
            for (c in 0 until env.width) {
                val cell = CellIndex(r, c)
                val symbol = when {
                    cell == agentState -> "\uD83E\uDD16"
                    cell == env.goal -> "\uD83D\uDFE9"
                    env.obstacles.contains(cell) -> "\uD83D\uDDC4\uFE0F"
                    env.hazards.contains(cell) -> "\uD83D\uDFE7"
                    else -> "\uD83C\uDFFE"
                }
                print("$symbol ")
            }
            println()
        }
        println()
    }

    /**
     * Animate the progression of an agent through a warehouse over a single episode.
     * @param env the warehouse environment
     * @param ep the episode to play back, containing agent's state-action pairs
     */
    suspend fun playbackEpisode(env: WarehouseEnv, ep: Episode, delay: Duration) {
        for ((i, state) in ep.steps.withIndex()) {
            // render current warehouse state
            println("     Step $i") // 5 spaces
            renderWarehouseWithAgent(env, state.first)
//            delay(delay)

            // clear the terminal
            print("\u001b[${env.height}A")  // move cursor up N lines
            delay(delay)
        }
    }

    /**
     * Plot a parameter sweep of agent learning rate's affect on average convergence episode.
     * @param alphaValues a map from an alpha value to its average convergence episode
     * @return the displayable chart
     */
    fun plotAlphaConvergence(alphaValues: Map<Double, Double>): CategoryChart {
        val alphas = alphaValues.keys.toList()
        val convEps = alphaValues.values.toList()
        val chart = CategoryChartBuilder()
            .width(800)
            .height(600)
            .title("Success Of Learning Rate Values in Stochastic Worlds")
            .xAxisTitle("Learning rate (α)")
            .yAxisTitle("Episode of Convergence, Negative For Failure")
            .build()
        chart.addSeries("Episode of Convergence", alphas, convEps)
        chart.styler.legendPosition = Styler.LegendPosition.InsideNE
        val customTitleFont = Font("Arial", Font.BOLD, 30)
        val customLegendFont = Font("Arial", Font.PLAIN, 25)
        chart.styler.legendFont = customLegendFont
        chart.styler.chartTitleFont = customTitleFont
        return chart
    }

    /**
     * Plot a bar graph of the average step size and incurred cost in episodes of a history.
     * @param history episode list of a trained agent
     * @param bucketSize number of episodes to average into a single bucket in the graph
     * @param isStochastic whether the training occurred in a stochastic world
     * @param convergenceEpisode episode at which convergence occurred, if any
     * @param metric a function that maps an Episode to a certain Double attribute
     * @param metricName the name of the metric, for plot title
     * @return the displayable chart
     */
    fun plotHistoryStats(history: List<Episode>,
                         bucketSize: Int,
                         minMaxPlot: Boolean,
                         isStochastic: Boolean,
                         convergenceEpisode: Int?,
                         metric: (Episode) -> Double,
                         metricName: String): CategoryChart {
        // set up space for the data we are interested in
        val avgCostData = mutableListOf<Double>()
        val minCostData = mutableListOf<Double>()
        val maxCostData = mutableListOf<Double>()

        // bucketize the episodes
        val numBuckets = history.size / bucketSize
        val labels = mutableListOf<String>()
        for (bucket in 0 until numBuckets) {
            // make the bucket
            val start = bucket * bucketSize
            val end = bucket * bucketSize + bucketSize
            val episodes = history.subList(start, end)

            // add the data
            labels.add("$end")
            val values = episodes.map(metric)
            avgCostData.add(values.average())
            minCostData.add(values.min())
            maxCostData.add(values.max())
        }

        // build the chart
        val stochastic = if (isStochastic) "With" else "Without"
        val chartTitle = "Agent $metricName Over Episodes $stochastic Stochasticity"
        val chart = CategoryChartBuilder()
            .width(800)
            .height(600)
            .title(chartTitle)
            .xAxisTitle("Episodes (buckets of $bucketSize)")
            .yAxisTitle(metricName)
            .build()
        val customTitleFont = Font("Arial", Font.BOLD, 30)
        val customLegendFont = Font("Arial", Font.PLAIN, 25)
        chart.styler.legendPosition = Styler.LegendPosition.InsideNE
        chart.styler.legendFont = customLegendFont
        chart.styler.chartTitleFont = customTitleFont

        // add convergence marker
        var convergenceBucket: Int? = null
        convergenceEpisode?.let {
            convergenceBucket = (convergenceEpisode - 1) / bucketSize
        }
        convergenceBucket?.let { bucket ->
            if (bucket in labels.indices) {
                val highlightValues = labels.mapIndexed { i, _ -> if (i == bucket) maxCostData.max() else Double.NaN }
                val highlight = chart.addSeries("Agent Convergence On Optimal Path", labels, highlightValues)
                highlight.fillColor = Color.RED
            }
        }

        // add actual data
        if (minMaxPlot) {
            val maxBar = chart.addSeries("Highest $metricName", labels, maxCostData)
            maxBar.fillColor = Color.GRAY
        }
        val avgBar = chart.addSeries("Average $metricName", labels, avgCostData)
        avgBar.fillColor = Color.blue
        if (minMaxPlot) {
            val minBar = chart.addSeries("Lowest $metricName", labels, minCostData)
            minBar.fillColor = Color.green
        }

        // return
        return chart
    }
}