const express = require("express")
const router = express.Router()
const TradingBotService = require("../services/trading_bot_service")
const AIService = require("../services/ai_service")
const DataStreamService = require("../services/data_stream_service")
const RiskManager = require("../services/risk_manager")
const AlertService = require("../services/alert_service")

// Start trading bot
router.post("/start", async (req, res) => {
  try {
    const { config } = req.body

    // Validate configuration
    if (!config || !config.strategies || !config.riskConfig) {
      return res.status(400).json({
        success: false,
        error: "Invalid configuration provided",
      })
    }

    // Start the trading bot
    const result = await req.app.locals.tradingBot.start(config)

    // Log the start event
    console.log(`🚀 Trading bot started by user at ${new Date().toISOString()}`)

    // Send real-time update
    req.app.locals.io.emit("botStatusUpdate", {
      status: "STARTED",
      timestamp: new Date(),
      config: config,
    })

    res.json(result)
  } catch (error) {
    console.error("❌ Error starting bot:", error)
    res.status(500).json({
      success: false,
      error: error.message,
    })
  }
})

// Stop trading bot
router.post("/stop", async (req, res) => {
  try {
    const result = await req.app.locals.tradingBot.stop()

    console.log(`🛑 Trading bot stopped by user at ${new Date().toISOString()}`)

    req.app.locals.io.emit("botStatusUpdate", {
      status: "STOPPED",
      timestamp: new Date(),
    })

    res.json(result)
  } catch (error) {
    console.error("❌ Error stopping bot:", error)
    res.status(500).json({
      success: false,
      error: error.message,
    })
  }
})

// Get bot status
router.get("/status", async (req, res) => {
  try {
    const status = req.app.locals.tradingBot.getStatus()

    // Add additional system metrics
    const systemMetrics = {
      uptime: process.uptime(),
      memoryUsage: process.memoryUsage(),
      cpuUsage: process.cpuUsage(),
      nodeVersion: process.version,
    }

    res.json({
      success: true,
      status: status,
      systemMetrics: systemMetrics,
      timestamp: new Date(),
    })
  } catch (error) {
    console.error("❌ Error getting bot status:", error)
    res.status(500).json({
      success: false,
      error: error.message,
    })
  }
})

// Pause trading but keep monitoring
router.post("/pause", async (req, res) => {
  try {
    req.app.locals.tradingBot.isActive = false

    req.app.locals.io.emit("botStatusUpdate", {
      status: "PAUSED",
      timestamp: new Date(),
    })

    res.json({
      success: true,
      message: "Bot paused successfully",
    })
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message,
    })
  }
})

// Resume trading
router.post("/resume", async (req, res) => {
  try {
    req.app.locals.tradingBot.isActive = true

    req.app.locals.io.emit("botStatusUpdate", {
      status: "RESUMED",
      timestamp: new Date(),
    })

    res.json({
      success: true,
      message: "Bot resumed successfully",
    })
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message,
    })
  }
})

// Emergency stop
router.post("/emergency-stop", async (req, res) => {
  try {
    // Emergency stop - close all positions immediately
    await req.app.locals.tradingBot.closeAllPositions()
    await req.app.locals.tradingBot.cancelAllOrders()
    req.app.locals.tradingBot.isActive = false

    // Send critical alert
    const alertService = new AlertService()
    await alertService.sendAlert("CRITICAL", "Emergency stop executed - all positions closed")

    req.app.locals.io.emit("emergencyStop", {
      timestamp: new Date(),
      message: "Emergency stop executed",
    })

    res.json({
      success: true,
      message: "Emergency stop executed successfully",
    })
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message,
    })
  }
})

// Get performance metrics
router.get("/performance", async (req, res) => {
  try {
    const { timeframe = "24h" } = req.query

    // Get performance metrics
    const performance = await req.app.locals.tradingBot.getPerformanceMetrics(timeframe)

    res.json({
      success: true,
      performance: performance,
      timeframe: timeframe,
      timestamp: new Date(),
    })
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message,
    })
  }
})

// Update bot configuration
router.post("/update-config", async (req, res) => {
  try {
    const { config } = req.body

    // Validate new configuration
    if (!config) {
      return res.status(400).json({
        success: false,
        error: "Configuration is required",
      })
    }

    // Update bot configuration
    await req.app.locals.tradingBot.updateConfiguration(config)

    req.app.locals.io.emit("configUpdate", {
      config: config,
      timestamp: new Date(),
    })

    res.json({
      success: true,
      message: "Configuration updated successfully",
    })
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message,
    })
  }
})

// Get bot logs
router.get("/logs", async (req, res) => {
  try {
    const { limit = 100, level = "all" } = req.query

    // Get bot logs
    const logs = await req.app.locals.tradingBot.getLogs(limit, level)

    res.json({
      success: true,
      logs: logs,
      count: logs.length,
      timestamp: new Date(),
    })
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message,
    })
  }
})

// Execute manual trade
router.post("/manual-trade", async (req, res) => {
  try {
    const { symbol, action, quantity, orderType, price } = req.body

    // Validate manual trade request
    if (!symbol || !action || !quantity) {
      return res.status(400).json({
        success: false,
        error: "Symbol, action, and quantity are required",
      })
    }

    // Execute manual trade
    const tradeResult = await req.app.locals.tradingBot.executeManualTrade({
      symbol,
      action,
      quantity,
      orderType: orderType || "MARKET",
      price,
    })

    req.app.locals.io.emit("manualTrade", {
      trade: tradeResult,
      timestamp: new Date(),
    })

    res.json({
      success: true,
      trade: tradeResult,
    })
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message,
    })
  }
})

// Get current positions
router.get("/positions", async (req, res) => {
  try {
    const positions = await req.app.locals.tradingBot.getCurrentPositions()

    res.json({
      success: true,
      positions: positions,
      count: positions.length,
      timestamp: new Date(),
    })
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message,
    })
  }
})

// Close specific position
router.post("/close-position", async (req, res) => {
  try {
    const { symbol, percentage = 100 } = req.body

    if (!symbol) {
      return res.status(400).json({
        success: false,
        error: "Symbol is required",
      })
    }

    const result = await req.app.locals.tradingBot.closePosition(symbol, percentage)

    req.app.locals.io.emit("positionClosed", {
      symbol,
      percentage,
      result,
      timestamp: new Date(),
    })

    res.json({
      success: true,
      result: result,
    })
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message,
    })
  }
})

// Get available strategies
router.get("/strategies", async (req, res) => {
  try {
    const strategies = await req.app.locals.tradingBot.getAvailableStrategies()

    res.json({
      success: true,
      strategies: strategies,
      count: strategies.length,
    })
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message,
    })
  }
})

// Enable specific strategy
router.post("/strategy/enable", async (req, res) => {
  try {
    const { strategyName } = req.body

    if (!strategyName) {
      return res.status(400).json({
        success: false,
        error: "Strategy name is required",
      })
    }

    await req.app.locals.tradingBot.enableStrategy(strategyName)

    res.json({
      success: true,
      message: `Strategy ${strategyName} enabled`,
    })
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message,
    })
  }
})

// Disable specific strategy
router.post("/strategy/disable", async (req, res) => {
  try {
    const { strategyName } = req.body

    if (!strategyName) {
      return res.status(400).json({
        success: false,
        error: "Strategy name is required",
      })
    }

    await req.app.locals.tradingBot.disableStrategy(strategyName)

    res.json({
      success: true,
      message: `Strategy ${strategyName} disabled`,
    })
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message,
    })
  }
})

module.exports = router
