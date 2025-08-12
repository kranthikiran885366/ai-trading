const express = require("express")
const router = express.Router()

// Get trade history with filtering and pagination
router.get("/", async (req, res) => {
  try {
    const {
      page = 1,
      limit = 50,
      symbol,
      action,
      startDate,
      endDate,
      strategy,
      minProfit,
      maxProfit,
      sortBy = "timestamp",
      sortOrder = "desc",
    } = req.query

    // Build filter criteria
    const filters = {}

    if (symbol) filters.symbol = symbol
    if (action) filters.action = action
    if (strategy) filters.strategy = strategy

    if (startDate || endDate) {
      filters.timestamp = {}
      if (startDate) filters.timestamp.$gte = new Date(startDate)
      if (endDate) filters.timestamp.$lte = new Date(endDate)
    }

    if (minProfit || maxProfit) {
      filters.profit = {}
      if (minProfit) filters.profit.$gte = Number.parseFloat(minProfit)
      if (maxProfit) filters.profit.$lte = Number.parseFloat(maxProfit)
    }

    // Get trade history from trading bot
    const tradeHistory = await req.app.locals.tradingBot.getTradeHistory({
      filters,
      page: Number.parseInt(page),
      limit: Number.parseInt(limit),
      sortBy,
      sortOrder,
    })

    // Calculate summary statistics
    const summary = calculateTradeSummary(tradeHistory.trades)

    res.json({
      success: true,
      trades: tradeHistory.trades,
      pagination: {
        page: Number.parseInt(page),
        limit: Number.parseInt(limit),
        total: tradeHistory.total,
        pages: Math.ceil(tradeHistory.total / limit),
      },
      summary,
      timestamp: new Date(),
    })
  } catch (error) {
    console.error("❌ Error fetching trade history:", error)
    res.status(500).json({
      success: false,
      error: error.message,
    })
  }
})

// Get trade details by ID
router.get("/:tradeId", async (req, res) => {
  try {
    const { tradeId } = req.params

    const trade = await req.app.locals.tradingBot.getTradeById(tradeId)

    if (!trade) {
      return res.status(404).json({
        success: false,
        error: "Trade not found",
      })
    }

    // Get related trades (if part of a strategy)
    const relatedTrades = await req.app.locals.tradingBot.getRelatedTrades(tradeId)

    res.json({
      success: true,
      trade,
      relatedTrades,
      timestamp: new Date(),
    })
  } catch (error) {
    console.error("❌ Error fetching trade details:", error)
    res.status(500).json({
      success: false,
      error: error.message,
    })
  }
})

// Get trade statistics
router.get("/stats/summary", async (req, res) => {
  try {
    const { timeframe = "30d", symbol, strategy } = req.query

    const stats = await req.app.locals.tradingBot.getTradeStatistics({
      timeframe,
      symbol,
      strategy,
    })

    res.json({
      success: true,
      statistics: stats,
      timeframe,
      timestamp: new Date(),
    })
  } catch (error) {
    console.error("❌ Error fetching trade statistics:", error)
    res.status(500).json({
      success: false,
      error: error.message,
    })
  }
})

// Get performance analytics
router.get("/analytics/performance", async (req, res) => {
  try {
    const { timeframe = "30d", groupBy = "day" } = req.query

    const analytics = await req.app.locals.tradingBot.getPerformanceAnalytics({
      timeframe,
      groupBy,
    })

    res.json({
      success: true,
      analytics,
      timeframe,
      groupBy,
      timestamp: new Date(),
    })
  } catch (error) {
    console.error("❌ Error fetching performance analytics:", error)
    res.status(500).json({
      success: false,
      error: error.message,
    })
  }
})

// Get strategy performance comparison
router.get("/analytics/strategies", async (req, res) => {
  try {
    const { timeframe = "30d" } = req.query

    const strategyPerformance = await req.app.locals.tradingBot.getStrategyPerformance(timeframe)

    res.json({
      success: true,
      strategies: strategyPerformance,
      timeframe,
      timestamp: new Date(),
    })
  } catch (error) {
    console.error("❌ Error fetching strategy performance:", error)
    res.status(500).json({
      success: false,
      error: error.message,
    })
  }
})

// Export trade history
router.get("/export/:format", async (req, res) => {
  try {
    const { format } = req.params
    const { startDate, endDate, symbol, strategy } = req.query

    if (!["csv", "json", "xlsx"].includes(format)) {
      return res.status(400).json({
        success: false,
        error: "Invalid export format. Supported: csv, json, xlsx",
      })
    }

    const filters = {}
    if (symbol) filters.symbol = symbol
    if (strategy) filters.strategy = strategy
    if (startDate || endDate) {
      filters.timestamp = {}
      if (startDate) filters.timestamp.$gte = new Date(startDate)
      if (endDate) filters.timestamp.$lte = new Date(endDate)
    }

    const exportData = await req.app.locals.tradingBot.exportTradeHistory(format, filters)

    // Set appropriate headers
    const filename = `trade_history_${new Date().toISOString().split("T")[0]}.${format}`

    if (format === "csv") {
      res.setHeader("Content-Type", "text/csv")
    } else if (format === "json") {
      res.setHeader("Content-Type", "application/json")
    } else if (format === "xlsx") {
      res.setHeader("Content-Type", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    }

    res.setHeader("Content-Disposition", `attachment; filename="${filename}"`)
    res.send(exportData)
  } catch (error) {
    console.error("❌ Error exporting trade history:", error)
    res.status(500).json({
      success: false,
      error: error.message,
    })
  }
})

// Get drawdown analysis
router.get("/analytics/drawdown", async (req, res) => {
  try {
    const { timeframe = "30d" } = req.query

    const drawdownAnalysis = await req.app.locals.tradingBot.getDrawdownAnalysis(timeframe)

    res.json({
      success: true,
      drawdown: drawdownAnalysis,
      timeframe,
      timestamp: new Date(),
    })
  } catch (error) {
    console.error("❌ Error fetching drawdown analysis:", error)
    res.status(500).json({
      success: false,
      error: error.message,
    })
  }
})

// Get risk metrics
router.get("/analytics/risk", async (req, res) => {
  try {
    const { timeframe = "30d" } = req.query

    const riskMetrics = await req.app.locals.tradingBot.getRiskMetrics(timeframe)

    res.json({
      success: true,
      risk: riskMetrics,
      timeframe,
      timestamp: new Date(),
    })
  } catch (error) {
    console.error("❌ Error fetching risk metrics:", error)
    res.status(500).json({
      success: false,
      error: error.message,
    })
  }
})

// Helper function to calculate trade summary
function calculateTradeSummary(trades) {
  if (!trades || trades.length === 0) {
    return {
      totalTrades: 0,
      winningTrades: 0,
      losingTrades: 0,
      winRate: 0,
      totalProfit: 0,
      totalLoss: 0,
      netProfit: 0,
      averageProfit: 0,
      averageLoss: 0,
      profitFactor: 0,
      largestWin: 0,
      largestLoss: 0,
    }
  }

  const totalTrades = trades.length
  const winningTrades = trades.filter((t) => t.profit > 0)
  const losingTrades = trades.filter((t) => t.profit < 0)

  const totalProfit = winningTrades.reduce((sum, t) => sum + t.profit, 0)
  const totalLoss = Math.abs(losingTrades.reduce((sum, t) => sum + t.profit, 0))
  const netProfit = totalProfit - totalLoss

  const averageProfit = winningTrades.length > 0 ? totalProfit / winningTrades.length : 0
  const averageLoss = losingTrades.length > 0 ? totalLoss / losingTrades.length : 0

  const profitFactor = totalLoss > 0 ? totalProfit / totalLoss : totalProfit > 0 ? Number.POSITIVE_INFINITY : 0

  const largestWin = winningTrades.length > 0 ? Math.max(...winningTrades.map((t) => t.profit)) : 0
  const largestLoss = losingTrades.length > 0 ? Math.min(...losingTrades.map((t) => t.profit)) : 0

  return {
    totalTrades,
    winningTrades: winningTrades.length,
    losingTrades: losingTrades.length,
    winRate: (winningTrades.length / totalTrades) * 100,
    totalProfit,
    totalLoss,
    netProfit,
    averageProfit,
    averageLoss,
    profitFactor,
    largestWin,
    largestLoss,
  }
}

module.exports = router
