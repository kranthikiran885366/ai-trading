const express = require("express")
const router = express.Router()

// Get AI service status
router.get("/status", async (req, res) => {
  try {
    const aiService = req.app.locals.aiService
    const status = {
      isInitialized: aiService.isInitialized,
      modelsLoaded: Object.keys(aiService.models).length,
      marketRegime: aiService.marketRegime,
      sentimentScore: aiService.sentimentScore,
      lastUpdate: new Date(),
    }

    res.json({
      success: true,
      status,
      timestamp: new Date(),
    })
  } catch (error) {
    console.error("❌ Error fetching AI service status:", error)
    res.status(500).json({
      success: false,
      error: error.message,
    })
  }
})

// Generate trading signal
router.post("/generate-signal", async (req, res) => {
  try {
    const { symbol, marketData, options = {} } = req.body

    if (!symbol || !marketData) {
      return res.status(400).json({
        success: false,
        error: "Symbol and market data are required",
      })
    }

    const aiService = req.app.locals.aiService
    const signal = await aiService.generateTradingSignal(marketData)

    // Broadcast signal to connected clients
    req.app.locals.io.emit("aiSignal", {
      signal,
      symbol,
      timestamp: new Date(),
    })

    res.json({
      success: true,
      signal,
      symbol,
      timestamp: new Date(),
    })
  } catch (error) {
    console.error("❌ Error generating AI signal:", error)
    res.status(500).json({
      success: false,
      error: error.message,
    })
  }
})

// Get market regime classification
router.get("/market-regime", async (req, res) => {
  try {
    const { symbols } = req.query
    const symbolList = symbols ? symbols.split(",") : ["SPY", "QQQ", "IWM"]

    const aiService = req.app.locals.aiService
    const marketRegime = await aiService.classifyMarketRegime(symbolList)

    res.json({
      success: true,
      marketRegime,
      symbols: symbolList,
      timestamp: new Date(),
    })
  } catch (error) {
    console.error("❌ Error getting market regime:", error)
    res.status(500).json({
      success: false,
      error: error.message,
    })
  }
})

// Get sentiment analysis
router.post("/sentiment", async (req, res) => {
  try {
    const { symbols, sources = ["news", "social"] } = req.body

    if (!symbols || !Array.isArray(symbols)) {
      return res.status(400).json({
        success: false,
        error: "Symbols array is required",
      })
    }

    const aiService = req.app.locals.aiService
    const sentimentAnalysis = await aiService.analyzeSentiment(symbols, sources)

    res.json({
      success: true,
      sentiment: sentimentAnalysis,
      symbols,
      sources,
      timestamp: new Date(),
    })
  } catch (error) {
    console.error("❌ Error analyzing sentiment:", error)
    res.status(500).json({
      success: false,
      error: error.message,
    })
  }
})

// Train AI models
router.post("/train", async (req, res) => {
  try {
    const { modelType, trainingData, parameters = {} } = req.body

    if (!modelType) {
      return res.status(400).json({
        success: false,
        error: "Model type is required",
      })
    }

    const aiService = req.app.locals.aiService

    // Start training in background
    const trainingJob = aiService.trainModel(modelType, trainingData, parameters)

    res.json({
      success: true,
      message: "Model training started",
      modelType,
      jobId: trainingJob.id,
      timestamp: new Date(),
    })
  } catch (error) {
    console.error("❌ Error starting model training:", error)
    res.status(500).json({
      success: false,
      error: error.message,
    })
  }
})

// Get training job status
router.get("/training/:jobId", async (req, res) => {
  try {
    const { jobId } = req.params

    const aiService = req.app.locals.aiService
    const jobStatus = await aiService.getTrainingJobStatus(jobId)

    if (!jobStatus) {
      return res.status(404).json({
        success: false,
        error: "Training job not found",
      })
    }

    res.json({
      success: true,
      job: jobStatus,
      timestamp: new Date(),
    })
  } catch (error) {
    console.error("❌ Error fetching training job status:", error)
    res.status(500).json({
      success: false,
      error: error.message,
    })
  }
})

// Get model performance metrics
router.get("/models/:modelName/performance", async (req, res) => {
  try {
    const { modelName } = req.params
    const { timeframe = "30d" } = req.query

    const aiService = req.app.locals.aiService
    const performance = await aiService.getModelPerformance(modelName, timeframe)

    res.json({
      success: true,
      performance,
      modelName,
      timeframe,
      timestamp: new Date(),
    })
  } catch (error) {
    console.error("❌ Error fetching model performance:", error)
    res.status(500).json({
      success: false,
      error: error.message,
    })
  }
})

// Update model parameters
router.put("/models/:modelName/parameters", async (req, res) => {
  try {
    const { modelName } = req.params
    const { parameters } = req.body

    if (!parameters) {
      return res.status(400).json({
        success: false,
        error: "Parameters object is required",
      })
    }

    const aiService = req.app.locals.aiService
    await aiService.updateModelParameters(modelName, parameters)

    res.json({
      success: true,
      message: `Parameters updated for model ${modelName}`,
      modelName,
      parameters,
      timestamp: new Date(),
    })
  } catch (error) {
    console.error("❌ Error updating model parameters:", error)
    res.status(500).json({
      success: false,
      error: error.message,
    })
  }
})

// Get feature importance
router.get("/models/:modelName/features", async (req, res) => {
  try {
    const { modelName } = req.params

    const aiService = req.app.locals.aiService
    const featureImportance = await aiService.getFeatureImportance(modelName)

    res.json({
      success: true,
      features: featureImportance,
      modelName,
      timestamp: new Date(),
    })
  } catch (error) {
    console.error("❌ Error fetching feature importance:", error)
    res.status(500).json({
      success: false,
      error: error.message,
    })
  }
})

// Get AI predictions
router.post("/predict", async (req, res) => {
  try {
    const { modelName, inputData, options = {} } = req.body

    if (!modelName || !inputData) {
      return res.status(400).json({
        success: false,
        error: "Model name and input data are required",
      })
    }

    const aiService = req.app.locals.aiService
    const prediction = await aiService.predict(modelName, inputData, options)

    res.json({
      success: true,
      prediction,
      modelName,
      timestamp: new Date(),
    })
  } catch (error) {
    console.error("❌ Error making prediction:", error)
    res.status(500).json({
      success: false,
      error: error.message,
    })
  }
})

// Get AI insights
router.get("/insights", async (req, res) => {
  try {
    const { symbols, timeframe = "1d" } = req.query
    const symbolList = symbols ? symbols.split(",") : ["SPY"]

    const aiService = req.app.locals.aiService
    const insights = await aiService.generateInsights(symbolList, timeframe)

    res.json({
      success: true,
      insights,
      symbols: symbolList,
      timeframe,
      timestamp: new Date(),
    })
  } catch (error) {
    console.error("❌ Error generating AI insights:", error)
    res.status(500).json({
      success: false,
      error: error.message,
    })
  }
})

// Get pattern recognition results
router.post("/patterns", async (req, res) => {
  try {
    const { symbol, chartData, patternTypes = ["all"] } = req.body

    if (!symbol || !chartData) {
      return res.status(400).json({
        success: false,
        error: "Symbol and chart data are required",
      })
    }

    const aiService = req.app.locals.aiService
    const patterns = await aiService.recognizePatterns(symbol, chartData, patternTypes)

    res.json({
      success: true,
      patterns,
      symbol,
      patternTypes,
      timestamp: new Date(),
    })
  } catch (error) {
    console.error("❌ Error recognizing patterns:", error)
    res.status(500).json({
      success: false,
      error: error.message,
    })
  }
})

// Get risk assessment
router.post("/risk-assessment", async (req, res) => {
  try {
    const { portfolioData, marketData, timeHorizon = "1d" } = req.body

    if (!portfolioData) {
      return res.status(400).json({
        success: false,
        error: "Portfolio data is required",
      })
    }

    const aiService = req.app.locals.aiService
    const riskAssessment = await aiService.assessRisk(portfolioData, marketData, timeHorizon)

    res.json({
      success: true,
      riskAssessment,
      timeHorizon,
      timestamp: new Date(),
    })
  } catch (error) {
    console.error("❌ Error assessing risk:", error)
    res.status(500).json({
      success: false,
      error: error.message,
    })
  }
})

// Get optimization suggestions
router.post("/optimize", async (req, res) => {
  try {
    const { strategy, performance, constraints = {} } = req.body

    if (!strategy || !performance) {
      return res.status(400).json({
        success: false,
        error: "Strategy and performance data are required",
      })
    }

    const aiService = req.app.locals.aiService
    const optimizations = await aiService.optimizeStrategy(strategy, performance, constraints)

    res.json({
      success: true,
      optimizations,
      strategy: strategy.name,
      timestamp: new Date(),
    })
  } catch (error) {
    console.error("❌ Error optimizing strategy:", error)
    res.status(500).json({
      success: false,
      error: error.message,
    })
  }
})

// Get AI model list
router.get("/models", async (req, res) => {
  try {
    const aiService = req.app.locals.aiService
    const models = await aiService.getAvailableModels()

    res.json({
      success: true,
      models,
      count: models.length,
      timestamp: new Date(),
    })
  } catch (error) {
    console.error("❌ Error fetching AI models:", error)
    res.status(500).json({
      success: false,
      error: error.message,
    })
  }
})

// Reload AI models
router.post("/models/reload", async (req, res) => {
  try {
    const { modelNames } = req.body

    const aiService = req.app.locals.aiService
    const reloadResults = await aiService.reloadModels(modelNames)

    res.json({
      success: true,
      results: reloadResults,
      timestamp: new Date(),
    })
  } catch (error) {
    console.error("❌ Error reloading AI models:", error)
    res.status(500).json({
      success: false,
      error: error.message,
    })
  }
})

module.exports = router
