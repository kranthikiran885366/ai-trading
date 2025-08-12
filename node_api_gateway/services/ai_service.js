const tf = require("@tensorflow/tfjs-node")
const EventEmitter = require("events")

class AIService extends EventEmitter {
  constructor() {
    super()
    this.models = {
      signalModel: null,
      marketClassifier: null,
      riskPredictor: null,
      sentimentAnalyzer: null,
    }
    this.isInitialized = false
    this.marketRegime = "NORMAL"
    this.sentimentScore = 0
    this.features = []
  }

  async initialize() {
    try {
      console.log("🧠 Initializing AI Service...")

      // Load or create models
      await this.loadModels()

      // Initialize feature extractors
      this.initializeFeatureExtractors()

      this.isInitialized = true
      console.log("✅ AI Service initialized successfully")
    } catch (error) {
      console.error("❌ Failed to initialize AI Service:", error)
      throw error
    }
  }

  async loadModels() {
    try {
      // Try to load existing models, create new ones if they don't exist
      this.models.signalModel = await this.loadOrCreateSignalModel()
      this.models.marketClassifier = await this.loadOrCreateMarketClassifier()
      this.models.riskPredictor = await this.loadOrCreateRiskPredictor()
    } catch (error) {
      console.log("Creating new models...")
      await this.createNewModels()
    }
  }

  async loadOrCreateSignalModel() {
    try {
      // Try to load existing model
      return await tf.loadLayersModel("file://./ai_models/signal_model/model.json")
    } catch (error) {
      // Create new signal prediction model
      const model = tf.sequential({
        layers: [
          tf.layers.dense({ inputShape: [20], units: 64, activation: "relu" }),
          tf.layers.dropout({ rate: 0.2 }),
          tf.layers.dense({ units: 32, activation: "relu" }),
          tf.layers.dropout({ rate: 0.2 }),
          tf.layers.dense({ units: 16, activation: "relu" }),
          tf.layers.dense({ units: 3, activation: "softmax" }), // BUY, SELL, HOLD
        ],
      })

      model.compile({
        optimizer: tf.train.adam(0.001),
        loss: "categoricalCrossentropy",
        metrics: ["accuracy"],
      })

      return model
    }
  }

  async loadOrCreateMarketClassifier() {
    try {
      return await tf.loadLayersModel("file://./ai_models/market_classifier/model.json")
    } catch (error) {
      // Create market regime classifier
      const model = tf.sequential({
        layers: [
          tf.layers.dense({ inputShape: [15], units: 32, activation: "relu" }),
          tf.layers.dense({ units: 16, activation: "relu" }),
          tf.layers.dense({ units: 4, activation: "softmax" }), // BULL, BEAR, SIDEWAYS, VOLATILE
        ],
      })

      model.compile({
        optimizer: tf.train.adam(0.001),
        loss: "categoricalCrossentropy",
        metrics: ["accuracy"],
      })

      return model
    }
  }

  async loadOrCreateRiskPredictor() {
    try {
      return await tf.loadLayersModel("file://./ai_models/risk_predictor/model.json")
    } catch (error) {
      // Create risk prediction model
      const model = tf.sequential({
        layers: [
          tf.layers.dense({ inputShape: [10], units: 24, activation: "relu" }),
          tf.layers.dense({ units: 12, activation: "relu" }),
          tf.layers.dense({ units: 1, activation: "sigmoid" }), // Risk score 0-1
        ],
      })

      model.compile({
        optimizer: tf.train.adam(0.001),
        loss: "binaryCrossentropy",
        metrics: ["accuracy"],
      })

      return model
    }
  }

  initializeFeatureExtractors() {
    this.technicalIndicators = {
      sma: this.calculateSMA,
      ema: this.calculateEMA,
      rsi: this.calculateRSI,
      macd: this.calculateMACD,
      bollinger: this.calculateBollingerBands,
      stochastic: this.calculateStochastic,
      atr: this.calculateATR,
      volume: this.calculateVolumeIndicators,
    }
  }

  async generateTradingSignal(marketData) {
    if (!this.isInitialized) {
      throw new Error("AI Service not initialized")
    }

    try {
      // Extract features from market data
      const features = await this.extractFeatures(marketData)

      // Predict signal
      const signalPrediction = await this.predictSignal(features)

      // Classify market regime
      const marketRegime = await this.classifyMarketRegime(features)

      // Calculate risk score
      const riskScore = await this.predictRisk(features)

      // Generate final signal
      const signal = this.generateFinalSignal(signalPrediction, marketRegime, riskScore, marketData)

      return signal
    } catch (error) {
      console.error("❌ Error generating trading signal:", error)
      throw error
    }
  }

  async extractFeatures(marketData) {
    const { symbol, candles, volume, news } = marketData

    // Technical indicators
    const closes = candles.map((c) => c.close)
    const highs = candles.map((c) => c.high)
    const lows = candles.map((c) => c.low)
    const volumes = candles.map((c) => c.volume)

    const features = [
      // Price-based features
      ...this.calculateSMA(closes, 10).slice(-1),
      ...this.calculateSMA(closes, 20).slice(-1),
      ...this.calculateEMA(closes, 12).slice(-1),
      ...this.calculateEMA(closes, 26).slice(-1),

      // Momentum indicators
      ...this.calculateRSI(closes, 14).slice(-1),
      ...this.calculateMACD(closes).slice(-1),
      ...this.calculateStochastic(highs, lows, closes, 14).slice(-1),

      // Volatility indicators
      ...this.calculateATR(highs, lows, closes, 14).slice(-1),
      ...this.calculateBollingerBands(closes, 20).slice(-1),

      // Volume indicators
      ...this.calculateVolumeIndicators(closes, volumes).slice(-1),

      // Market structure
      this.calculateTrend(closes),
      this.calculateVolatility(closes),
      this.calculateMomentum(closes),

      // Sentiment (if news available)
      news ? await this.analyzeSentiment(news) : 0,
    ]

    return this.normalizeFeatures(features.flat())
  }

  async predictSignal(features) {
    const inputTensor = tf.tensor2d([features])
    const prediction = this.models.signalModel.predict(inputTensor)
    const probabilities = await prediction.data()

    inputTensor.dispose()
    prediction.dispose()

    return {
      buy: probabilities[0],
      sell: probabilities[1],
      hold: probabilities[2],
    }
  }

  async classifyMarketRegime(features) {
    const marketFeatures = features.slice(0, 15) // Use first 15 features for market classification
    const inputTensor = tf.tensor2d([marketFeatures])
    const prediction = this.models.marketClassifier.predict(inputTensor)
    const probabilities = await prediction.data()

    inputTensor.dispose()
    prediction.dispose()

    const regimes = ["BULL", "BEAR", "SIDEWAYS", "VOLATILE"]
    const maxIndex = probabilities.indexOf(Math.max(...probabilities))

    this.marketRegime = regimes[maxIndex]
    return {
      regime: this.marketRegime,
      confidence: probabilities[maxIndex],
    }
  }

  async predictRisk(features) {
    const riskFeatures = features.slice(-10) // Use last 10 features for risk prediction
    const inputTensor = tf.tensor2d([riskFeatures])
    const prediction = this.models.riskPredictor.predict(inputTensor)
    const riskScore = await prediction.data()

    inputTensor.dispose()
    prediction.dispose()

    return riskScore[0]
  }

  generateFinalSignal(signalPrediction, marketRegime, riskScore, marketData) {
    const { symbol, candles } = marketData
    const currentPrice = candles[candles.length - 1].close

    // Determine action based on predictions
    let action = "HOLD"
    let confidence = 0

    if (signalPrediction.buy > 0.6 && riskScore < 0.7) {
      action = "BUY"
      confidence = signalPrediction.buy * (1 - riskScore)
    } else if (signalPrediction.sell > 0.6 && riskScore < 0.7) {
      action = "SELL"
      confidence = signalPrediction.sell * (1 - riskScore)
    }

    // Adjust based on market regime
    if (marketRegime.regime === "VOLATILE" && confidence > 0) {
      confidence *= 0.7 // Reduce confidence in volatile markets
    }

    // Calculate stop loss and take profit
    const atr = this.calculateATR(
      candles.map((c) => c.high),
      candles.map((c) => c.low),
      candles.map((c) => c.close),
      14,
    ).slice(-1)[0]

    const stopLoss = action === "BUY" ? currentPrice - atr * 2 : currentPrice + atr * 2

    const takeProfit = action === "BUY" ? currentPrice + atr * 3 : currentPrice - atr * 3

    return {
      id: `signal_${Date.now()}`,
      symbol: symbol,
      action: action,
      confidence: confidence,
      price: currentPrice,
      stopLoss: stopLoss,
      takeProfit: takeProfit,
      marketRegime: marketRegime.regime,
      riskScore: riskScore,
      timestamp: new Date(),
      strategy: "AI_SIGNAL",
      reasoning: this.generateReasoning(signalPrediction, marketRegime, riskScore),
    }
  }

  generateReasoning(signalPrediction, marketRegime, riskScore) {
    const reasons = []

    if (signalPrediction.buy > 0.6) {
      reasons.push(`Strong buy signal (${(signalPrediction.buy * 100).toFixed(1)}%)`)
    } else if (signalPrediction.sell > 0.6) {
      reasons.push(`Strong sell signal (${(signalPrediction.sell * 100).toFixed(1)}%)`)
    }

    reasons.push(`Market regime: ${marketRegime.regime} (${(marketRegime.confidence * 100).toFixed(1)}%)`)
    reasons.push(`Risk score: ${(riskScore * 100).toFixed(1)}%`)

    return reasons.join(", ")
  }

  // Technical Indicators Implementation
  calculateSMA(prices, period) {
    const sma = []
    for (let i = period - 1; i < prices.length; i++) {
      const sum = prices.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0)
      sma.push(sum / period)
    }
    return sma
  }

  calculateEMA(prices, period) {
    const ema = []
    const multiplier = 2 / (period + 1)
    ema[0] = prices[0]

    for (let i = 1; i < prices.length; i++) {
      ema[i] = prices[i] * multiplier + ema[i - 1] * (1 - multiplier)
    }

    return ema
  }

  calculateRSI(prices, period = 14) {
    const gains = []
    const losses = []

    for (let i = 1; i < prices.length; i++) {
      const change = prices[i] - prices[i - 1]
      gains.push(change > 0 ? change : 0)
      losses.push(change < 0 ? Math.abs(change) : 0)
    }

    const rsi = []
    for (let i = period - 1; i < gains.length; i++) {
      const avgGain = gains.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0) / period
      const avgLoss = losses.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0) / period

      if (avgLoss === 0) {
        rsi.push(100)
      } else {
        const rs = avgGain / avgLoss
        rsi.push(100 - 100 / (1 + rs))
      }
    }

    return rsi
  }

  calculateMACD(prices, fastPeriod = 12, slowPeriod = 26, signalPeriod = 9) {
    const fastEMA = this.calculateEMA(prices, fastPeriod)
    const slowEMA = this.calculateEMA(prices, slowPeriod)

    const macdLine = []
    for (let i = 0; i < Math.min(fastEMA.length, slowEMA.length); i++) {
      macdLine.push(fastEMA[i] - slowEMA[i])
    }

    const signalLine = this.calculateEMA(macdLine, signalPeriod)
    const histogram = []

    for (let i = 0; i < Math.min(macdLine.length, signalLine.length); i++) {
      histogram.push(macdLine[i] - signalLine[i])
    }

    return histogram
  }

  calculateBollingerBands(prices, period = 20, stdDev = 2) {
    const sma = this.calculateSMA(prices, period)
    const bands = []

    for (let i = period - 1; i < prices.length; i++) {
      const slice = prices.slice(i - period + 1, i + 1)
      const mean = slice.reduce((a, b) => a + b, 0) / period
      const variance = slice.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / period
      const standardDeviation = Math.sqrt(variance)

      bands.push({
        upper: sma[i - period + 1] + standardDeviation * stdDev,
        middle: sma[i - period + 1],
        lower: sma[i - period + 1] - standardDeviation * stdDev,
      })
    }

    return bands.map((b) => [(b.upper - b.middle) / b.middle]) // Normalized band width
  }

  calculateStochastic(highs, lows, closes, period = 14) {
    const stochastic = []

    for (let i = period - 1; i < closes.length; i++) {
      const highestHigh = Math.max(...highs.slice(i - period + 1, i + 1))
      const lowestLow = Math.min(...lows.slice(i - period + 1, i + 1))
      const currentClose = closes[i]

      const k = ((currentClose - lowestLow) / (highestHigh - lowestLow)) * 100
      stochastic.push(k)
    }

    return stochastic
  }

  calculateATR(highs, lows, closes, period = 14) {
    const trueRanges = []

    for (let i = 1; i < closes.length; i++) {
      const tr1 = highs[i] - lows[i]
      const tr2 = Math.abs(highs[i] - closes[i - 1])
      const tr3 = Math.abs(lows[i] - closes[i - 1])
      trueRanges.push(Math.max(tr1, tr2, tr3))
    }

    return this.calculateSMA(trueRanges, period)
  }

  calculateVolumeIndicators(prices, volumes) {
    // On-Balance Volume (OBV)
    const obv = [volumes[0]]
    for (let i = 1; i < prices.length; i++) {
      if (prices[i] > prices[i - 1]) {
        obv.push(obv[i - 1] + volumes[i])
      } else if (prices[i] < prices[i - 1]) {
        obv.push(obv[i - 1] - volumes[i])
      } else {
        obv.push(obv[i - 1])
      }
    }

    // Volume Rate of Change
    const volumeROC = []
    for (let i = 10; i < volumes.length; i++) {
      const roc = ((volumes[i] - volumes[i - 10]) / volumes[i - 10]) * 100
      volumeROC.push(roc)
    }

    return [obv.slice(-1)[0] / 1000000, volumeROC.slice(-1)[0] || 0] // Normalized
  }

  calculateTrend(prices) {
    const returns = []
    for (let i = 1; i < prices.length; i++) {
      returns.push((prices[i] - prices[i - 1]) / prices[i - 1])
    }

    // Simple trend calculation based on recent returns
    const recentReturns = returns.slice(-10)
    const avgReturn = recentReturns.reduce((a, b) => a + b, 0) / recentReturns.length

    return avgReturn > 0 ? 1 : avgReturn < 0 ? -1 : 0
  }

  calculateVolatility(prices) {
    const returns = []
    for (let i = 1; i < prices.length; i++) {
      returns.push((prices[i] - prices[i - 1]) / prices[i - 1])
    }

    const mean = returns.reduce((a, b) => a + b, 0) / returns.length
    const variance = returns.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / returns.length

    return Math.sqrt(variance) * Math.sqrt(252) // Annualized volatility
  }

  calculateMomentum(prices) {
    const period = 10
    if (prices.length < period) return 0

    return (prices[prices.length - 1] - prices[prices.length - period]) / prices[prices.length - period]
  }

  normalizeFeatures(features) {
    // Simple min-max normalization
    const min = Math.min(...features)
    const max = Math.max(...features)

    if (max === min) return features.map(() => 0)

    return features.map((f) => (f - min) / (max - min))
  }

  async analyzeSentiment(newsText) {
    // Simple sentiment analysis using keyword matching
    const positiveWords = ["bullish", "growth", "profit", "gain", "rise", "up", "positive", "strong"]
    const negativeWords = ["bearish", "loss", "fall", "down", "negative", "weak", "decline", "drop"]

    const text = newsText.toLowerCase()
    let score = 0

    positiveWords.forEach((word) => {
      const matches = (text.match(new RegExp(word, "g")) || []).length
      score += matches
    })

    negativeWords.forEach((word) => {
      const matches = (text.match(new RegExp(word, "g")) || []).length
      score -= matches
    })

    // Normalize to -1 to 1 range
    return Math.max(-1, Math.min(1, score / 10))
  }

  async trainModel(modelName, trainingData) {
    if (!this.models[modelName]) {
      throw new Error(`Model ${modelName} not found`)
    }

    const { features, labels } = trainingData
    const xs = tf.tensor2d(features)
    const ys = tf.tensor2d(labels)

    await this.models[modelName].fit(xs, ys, {
      epochs: 100,
      batchSize: 32,
      validationSplit: 0.2,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          console.log(`Epoch ${epoch}: loss = ${logs.loss.toFixed(4)}, accuracy = ${logs.acc.toFixed(4)}`)
        },
      },
    })

    xs.dispose()
    ys.dispose()

    // Save the trained model
    await this.models[modelName].save(`file://./ai_models/${modelName}/model.json`)
  }

  isReady() {
    return this.isInitialized
  }

  getMarketRegime() {
    return this.marketRegime
  }

  getSentimentScore() {
    return this.sentimentScore
  }
}

module.exports = AIService
