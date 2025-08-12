const express = require("express")
const router = express.Router()

// Get current risk settings
router.get("/", async (req, res) => {
  try {
    const riskSettings = await req.app.locals.tradingBot.getRiskSettings()

    res.json({
      success: true,
      riskSettings,
      timestamp: new Date(),
    })
  } catch (error) {
    console.error("❌ Error fetching risk settings:", error)
    res.status(500).json({
      success: false,
      error: error.message,
    })
  }
})

// Update risk settings
router.put("/", async (req, res) => {
  try {
    const { riskSettings } = req.body

    // Validate risk settings
    const validation = validateRiskSettings(riskSettings)
    if (!validation.isValid) {
      return res.status(400).json({
        success: false,
        error: "Invalid risk settings",
        details: validation.errors,
      })
    }

    // Update risk settings
    await req.app.locals.tradingBot.updateRiskSettings(riskSettings)

    // Broadcast update to connected clients
    req.app.locals.io.emit("riskSettingsUpdate", {
      riskSettings,
      timestamp: new Date(),
    })

    res.json({
      success: true,
      message: "Risk settings updated successfully",
      riskSettings,
    })
  } catch (error) {
    console.error("❌ Error updating risk settings:", error)
    res.status(500).json({
      success: false,
      error: error.message,
    })
  }
})

// Get risk limits
router.get("/limits", async (req, res) => {
  try {
    const riskLimits = await req.app.locals.tradingBot.getRiskLimits()

    res.json({
      success: true,
      riskLimits,
      timestamp: new Date(),
    })
  } catch (error) {
    console.error("❌ Error fetching risk limits:", error)
    res.status(500).json({
      success: false,
      error: error.message,
    })
  }
})

// Update risk limits
router.put("/limits", async (req, res) => {
  try {
    const { limits } = req.body

    // Validate limits
    if (!limits || typeof limits !== "object") {
      return res.status(400).json({
        success: false,
        error: "Invalid limits object",
      })
    }

    await req.app.locals.tradingBot.updateRiskLimits(limits)

    res.json({
      success: true,
      message: "Risk limits updated successfully",
      limits,
    })
  } catch (error) {
    console.error("❌ Error updating risk limits:", error)
    res.status(500).json({
      success: false,
      error: error.message,
    })
  }
})

// Get position sizing settings
router.get("/position-sizing", async (req, res) => {
  try {
    const positionSizing = await req.app.locals.tradingBot.getPositionSizingSettings()

    res.json({
      success: true,
      positionSizing,
      timestamp: new Date(),
    })
  } catch (error) {
    console.error("❌ Error fetching position sizing settings:", error)
    res.status(500).json({
      success: false,
      error: error.message,
    })
  }
})

// Update position sizing settings
router.put("/position-sizing", async (req, res) => {
  try {
    const { positionSizing } = req.body

    // Validate position sizing settings
    const validation = validatePositionSizing(positionSizing)
    if (!validation.isValid) {
      return res.status(400).json({
        success: false,
        error: "Invalid position sizing settings",
        details: validation.errors,
      })
    }

    await req.app.locals.tradingBot.updatePositionSizingSettings(positionSizing)

    res.json({
      success: true,
      message: "Position sizing settings updated successfully",
      positionSizing,
    })
  } catch (error) {
    console.error("❌ Error updating position sizing settings:", error)
    res.status(500).json({
      success: false,
      error: error.message,
    })
  }
})

// Get stop loss settings
router.get("/stop-loss", async (req, res) => {
  try {
    const stopLossSettings = await req.app.locals.tradingBot.getStopLossSettings()

    res.json({
      success: true,
      stopLossSettings,
      timestamp: new Date(),
    })
  } catch (error) {
    console.error("❌ Error fetching stop loss settings:", error)
    res.status(500).json({
      success: false,
      error: error.message,
    })
  }
})

// Update stop loss settings
router.put("/stop-loss", async (req, res) => {
  try {
    const { stopLossSettings } = req.body

    await req.app.locals.tradingBot.updateStopLossSettings(stopLossSettings)

    res.json({
      success: true,
      message: "Stop loss settings updated successfully",
      stopLossSettings,
    })
  } catch (error) {
    console.error("❌ Error updating stop loss settings:", error)
    res.status(500).json({
      success: false,
      error: error.message,
    })
  }
})

// Get current risk status
router.get("/status", async (req, res) => {
  try {
    const riskStatus = await req.app.locals.tradingBot.getRiskStatus()

    res.json({
      success: true,
      riskStatus,
      timestamp: new Date(),
    })
  } catch (error) {
    console.error("❌ Error fetching risk status:", error)
    res.status(500).json({
      success: false,
      error: error.message,
    })
  }
})

// Get risk alerts
router.get("/alerts", async (req, res) => {
  try {
    const { limit = 50, severity } = req.query

    const riskAlerts = await req.app.locals.tradingBot.getRiskAlerts({
      limit: Number.parseInt(limit),
      severity,
    })

    res.json({
      success: true,
      alerts: riskAlerts,
      count: riskAlerts.length,
      timestamp: new Date(),
    })
  } catch (error) {
    console.error("❌ Error fetching risk alerts:", error)
    res.status(500).json({
      success: false,
      error: error.message,
    })
  }
})

// Acknowledge risk alert
router.post("/alerts/:alertId/acknowledge", async (req, res) => {
  try {
    const { alertId } = req.params
    const { userId, note } = req.body

    await req.app.locals.tradingBot.acknowledgeRiskAlert(alertId, userId, note)

    res.json({
      success: true,
      message: "Risk alert acknowledged successfully",
    })
  } catch (error) {
    console.error("❌ Error acknowledging risk alert:", error)
    res.status(500).json({
      success: false,
      error: error.message,
    })
  }
})

// Test risk settings
router.post("/test", async (req, res) => {
  try {
    const { riskSettings, testScenarios } = req.body

    const testResults = await req.app.locals.tradingBot.testRiskSettings(riskSettings, testScenarios)

    res.json({
      success: true,
      testResults,
      timestamp: new Date(),
    })
  } catch (error) {
    console.error("❌ Error testing risk settings:", error)
    res.status(500).json({
      success: false,
      error: error.message,
    })
  }
})

// Reset risk settings to defaults
router.post("/reset", async (req, res) => {
  try {
    const defaultSettings = await req.app.locals.tradingBot.resetRiskSettingsToDefaults()

    res.json({
      success: true,
      message: "Risk settings reset to defaults",
      riskSettings: defaultSettings,
    })
  } catch (error) {
    console.error("❌ Error resetting risk settings:", error)
    res.status(500).json({
      success: false,
      error: error.message,
    })
  }
})

// Validation functions
function validateRiskSettings(settings) {
  const errors = []

  if (!settings) {
    errors.push("Risk settings object is required")
    return { isValid: false, errors }
  }

  // Validate max daily loss
  if (settings.maxDailyLoss !== undefined) {
    if (typeof settings.maxDailyLoss !== "number" || settings.maxDailyLoss <= 0) {
      errors.push("Max daily loss must be a positive number")
    }
  }

  // Validate max drawdown
  if (settings.maxDrawdown !== undefined) {
    if (typeof settings.maxDrawdown !== "number" || settings.maxDrawdown <= 0 || settings.maxDrawdown > 1) {
      errors.push("Max drawdown must be a number between 0 and 1")
    }
  }

  // Validate max position size
  if (settings.maxPositionSize !== undefined) {
    if (typeof settings.maxPositionSize !== "number" || settings.maxPositionSize <= 0 || settings.maxPositionSize > 1) {
      errors.push("Max position size must be a number between 0 and 1")
    }
  }

  // Validate risk per trade
  if (settings.riskPerTrade !== undefined) {
    if (typeof settings.riskPerTrade !== "number" || settings.riskPerTrade <= 0 || settings.riskPerTrade > 0.1) {
      errors.push("Risk per trade must be a number between 0 and 0.1 (10%)")
    }
  }

  return { isValid: errors.length === 0, errors }
}

function validatePositionSizing(positionSizing) {
  const errors = []

  if (!positionSizing) {
    errors.push("Position sizing object is required")
    return { isValid: false, errors }
  }

  // Validate method
  const validMethods = ["fixed", "percentage", "kelly", "volatility"]
  if (positionSizing.method && !validMethods.includes(positionSizing.method)) {
    errors.push(`Position sizing method must be one of: ${validMethods.join(", ")}`)
  }

  // Validate base size
  if (positionSizing.baseSize !== undefined) {
    if (typeof positionSizing.baseSize !== "number" || positionSizing.baseSize <= 0) {
      errors.push("Base size must be a positive number")
    }
  }

  // Validate max size
  if (positionSizing.maxSize !== undefined) {
    if (typeof positionSizing.maxSize !== "number" || positionSizing.maxSize <= 0) {
      errors.push("Max size must be a positive number")
    }
  }

  return { isValid: errors.length === 0, errors }
}

module.exports = router
