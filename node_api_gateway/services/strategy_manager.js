const EventEmitter = require("events")

class StrategyManager extends EventEmitter {
  constructor() {
    super()
    this.strategies = new Map()
    this.activeStrategies = new Set()
    this.isRunning = false
    this.intervals = new Map()

    // Load built-in strategies
    this.loadBuiltInStrategies()
  }

  loadBuiltInStrategies() {
    // EMA Crossover Strategy
    this.strategies.set("EMA_CROSSOVER", {
      name: "EMA Crossover",
      description: "Buy when fast EMA crosses above slow EMA, sell when opposite",
      parameters: {
        fastPeriod: 12,
        slowPeriod: 26,
        signalPeriod: 9,
      },
      execute: this.emaStrategy.bind(this),
    })

    // RSI Strategy
    this.strategies.set("RSI_BREAKOUT", {
      name: "RSI Breakout",
      description: "Buy when RSI crosses above 30, sell when crosses below 70",
      parameters: {
        period: 14,
        oversold: 30,
        overbought: 70,
      },
      execute: this.rsiStrategy.bind(this),
    })

    // Bollinger Bands Strategy
    this.strategies.set("BOLLINGER_BANDS", {
      name: "Bollinger Bands",
      description: "Buy at lower band, sell at upper band",
      parameters: {
        period: 20,
        stdDev: 2,
      },
      execute: this.bollingerStrategy.bind(this),
    })

    // MACD Strategy
    this.strategies.set("MACD_SIGNAL", {
      name: "MACD Signal",
      description: "Buy/sell based on MACD line crossing signal line",
      parameters: {
        fastPeriod: 12,
        slowPeriod: 26,
        signalPeriod: 9,
      },
      execute: this.macdStrategy.bind(this),
    })

    // Multi-Indicator Strategy
    this.strategies.set("MULTI_INDICATOR", {
      name: "Multi Indicator",
      description: "Combines multiple indicators for signal confirmation",
      parameters: {
        rsiPeriod: 14,
        emaPeriod: 21,
        macdFast: 12,
        macdSlow: 26,
      },
      execute: this.multiIndicatorStrategy.bind(this),
    })

    console.log(`📈 Loaded ${this.strategies.size} built-in strategies`)
  }

  async loadStrategies(strategyConfigs) {
    try {
      for (const config of strategyConfigs) {
        if (this.strategies.has(config.name)) {
          this.activeStrategies.add(config.name)

          // Update parameters if provided
          if (config.parameters) {
            const strategy = this.strategies.get(config.name)
            strategy.parameters = { ...strategy.parameters, ...config.parameters }
          }

          console.log(`✅ Activated strategy: ${config.name}`)
        } else {
          console.warn(`⚠️ Strategy not found: ${config.name}`)
        }
      }

      return true
    } catch (error) {
      console.error("❌ Error loading strategies:", error)
      throw error
    }
  }

  start() {
    if (this.isRunning) return

    this.isRunning = true
    console.log("🚀 Strategy Manager started")

    // Start strategy execution intervals
    for (const strategyName of this.activeStrategies) {
      this.startStrategyExecution(strategyName)
    }
  }

  stop() {
    this.isRunning = false

    // Clear all intervals
    for (const [strategyName, interval] of this.intervals) {
      clearInterval(interval)
      console.log(`⏹️ Stopped strategy: ${strategyName}`)
    }

    this.intervals.clear()
    console.log("🛑 Strategy Manager stopped")
  }

  startStrategyExecution(strategyName) {
    const strategy = this.strategies.get(strategyName)
    if (!strategy) return

    // Execute strategy every 30 seconds
    const interval = setInterval(async () => {
      if (!this.isRunning) return

      try {
        await this.executeStrategy(strategyName)
      } catch (error) {
        console.error(`❌ Error executing strategy ${strategyName}:`, error)
      }
    }, 30000)

    this.intervals.set(strategyName, interval)
    console.log(`▶️ Started executing strategy: ${strategyName}`)
  }

  async executeStrategy(strategyName) {
    const strategy = this.strategies.get(strategyName)
    if (!strategy) return

    try {
      // Get market data for analysis
      const marketData = await this.getMarketData()

      // Execute strategy logic
      const signals = await strategy.execute(marketData, strategy.parameters)

      // Emit signals
      if (signals && signals.length > 0) {
        for (const signal of signals) {
          signal.strategy = strategyName
          signal.timestamp = new Date()
          this.emit("signal", signal)
        }
      }
    } catch (error) {
      console.error(`❌ Strategy execution error for ${strategyName}:`, error)
    }
  }

  // EMA Crossover Strategy Implementation
  async emaStrategy(marketData, params) {
    const signals = []

    for (const [symbol, data] of Object.entries(marketData)) {
      if (!data.candles || data.candles.length < params.slowPeriod) continue

      const closes = data.candles.map((c) => c.close)
      const fastEMA = this.calculateEMA(closes, params.fastPeriod)
      const slowEMA = this.calculateEMA(closes, params.slowPeriod)

      if (fastEMA.length < 2 || slowEMA.length < 2) continue

      const currentFast = fastEMA[fastEMA.length - 1]
      const currentSlow = slowEMA[slowEMA.length - 1]
      const prevFast = fastEMA[fastEMA.length - 2]
      const prevSlow = slowEMA[slowEMA.length - 2]

      // Bullish crossover
      if (prevFast <= prevSlow && currentFast > currentSlow) {
        signals.push({
          id: `ema_buy_${symbol}_${Date.now()}`,
          symbol: symbol,
          action: "BUY",
          price: data.candles[data.candles.length - 1].close,
          confidence: 0.7,
          reasoning: "Fast EMA crossed above slow EMA",
        })
      }
      // Bearish crossover
      else if (prevFast >= prevSlow && currentFast < currentSlow) {
        signals.push({
          id: `ema_sell_${symbol}_${Date.now()}`,
          symbol: symbol,
          action: "SELL",
          price: data.candles[data.candles.length - 1].close,
          confidence: 0.7,
          reasoning: "Fast EMA crossed below slow EMA",
        })
      }
    }

    return signals
  }

  // RSI Strategy Implementation
  async rsiStrategy(marketData, params) {
    const signals = []

    for (const [symbol, data] of Object.entries(marketData)) {
      if (!data.candles || data.candles.length < params.period + 1) continue

      const closes = data.candles.map((c) => c.close)
      const rsi = this.calculateRSI(closes, params.period)

      if (rsi.length < 2) continue

      const currentRSI = rsi[rsi.length - 1]
      const prevRSI = rsi[rsi.length - 2]

      // Oversold to normal (buy signal)
      if (prevRSI <= params.oversold && currentRSI > params.oversold) {
        signals.push({
          id: `rsi_buy_${symbol}_${Date.now()}`,
          symbol: symbol,
          action: "BUY",
          price: data.candles[data.candles.length - 1].close,
          confidence: 0.6,
          reasoning: `RSI crossed above ${params.oversold} (${currentRSI.toFixed(2)})`,
        })
      }
      // Overbought to normal (sell signal)
      else if (prevRSI >= params.overbought && currentRSI < params.overbought) {
        signals.push({
          id: `rsi_sell_${symbol}_${Date.now()}`,
          symbol: symbol,
          action: "SELL",
          price: data.candles[data.candles.length - 1].close,
          confidence: 0.6,
          reasoning: `RSI crossed below ${params.overbought} (${currentRSI.toFixed(2)})`,
        })
      }
    }

    return signals
  }

  // Bollinger Bands Strategy Implementation
  async bollingerStrategy(marketData, params) {
    const signals = []

    for (const [symbol, data] of Object.entries(marketData)) {
      if (!data.candles || data.candles.length < params.period) continue

      const closes = data.candles.map((c) => c.close)
      const bands = this.calculateBollingerBands(closes, params.period, params.stdDev)

      if (bands.length === 0) continue

      const currentPrice = closes[closes.length - 1]
      const currentBands = bands[bands.length - 1]

      // Price touches lower band (buy signal)
      if (currentPrice <= currentBands.lower) {
        signals.push({
          id: `bb_buy_${symbol}_${Date.now()}`,
          symbol: symbol,
          action: "BUY",
          price: currentPrice,
          confidence: 0.65,
          reasoning: `Price at lower Bollinger Band (${currentBands.lower.toFixed(2)})`,
        })
      }
      // Price touches upper band (sell signal)
      else if (currentPrice >= currentBands.upper) {
        signals.push({
          id: `bb_sell_${symbol}_${Date.now()}`,
          symbol: symbol,
          action: "SELL",
          price: currentPrice,
          confidence: 0.65,
          reasoning: `Price at upper Bollinger Band (${currentBands.upper.toFixed(2)})`,
        })
      }
    }

    return signals
  }

  // MACD Strategy Implementation
  async macdStrategy(marketData, params) {
    const signals = []

    for (const [symbol, data] of Object.entries(marketData)) {
      if (!data.candles || data.candles.length < params.slowPeriod + params.signalPeriod) continue

      const closes = data.candles.map((c) => c.close)
      const macd = this.calculateMACD(closes, params.fastPeriod, params.slowPeriod, params.signalPeriod)

      if (macd.length < 2) continue

      const current = macd[macd.length - 1]
      const previous = macd[macd.length - 2]

      // MACD line crosses above signal line (buy)
      if (previous.macd <= previous.signal && current.macd > current.signal) {
        signals.push({
          id: `macd_buy_${symbol}_${Date.now()}`,
          symbol: symbol,
          action: "BUY",
          price: data.candles[data.candles.length - 1].close,
          confidence: 0.7,
          reasoning: "MACD line crossed above signal line",
        })
      }
      // MACD line crosses below signal line (sell)
      else if (previous.macd >= previous.signal && current.macd < current.signal) {
        signals.push({
          id: `macd_sell_${symbol}_${Date.now()}`,
          symbol: symbol,
          action: "SELL",
          price: data.candles[data.candles.length - 1].close,
          confidence: 0.7,
          reasoning: "MACD line crossed below signal line",
        })
      }
    }

    return signals
  }

  // Multi-Indicator Strategy Implementation
  async multiIndicatorStrategy(marketData, params) {
    const signals = []

    for (const [symbol, data] of Object.entries(marketData)) {
      if (!data.candles || data.candles.length < 50) continue

      const closes = data.candles.map((c) => c.close)
      const currentPrice = closes[closes.length - 1]

      // Calculate indicators
      const rsi = this.calculateRSI(closes, params.rsiPeriod)
      const ema = this.calculateEMA(closes, params.emaPeriod)
      const macd = this.calculateMACD(closes, params.macdFast, 26, 9)

      if (rsi.length === 0 || ema.length === 0 || macd.length === 0) continue

      const currentRSI = rsi[rsi.length - 1]
      const currentEMA = ema[ema.length - 1]
      const currentMACD = macd[macd.length - 1]

      let bullishSignals = 0
      let bearishSignals = 0
      const reasons = []

      // RSI analysis
      if (currentRSI < 40) {
        bullishSignals++
        reasons.push(`RSI oversold (${currentRSI.toFixed(2)})`)
      } else if (currentRSI > 60) {
        bearishSignals++
        reasons.push(`RSI overbought (${currentRSI.toFixed(2)})`)
      }

      // EMA analysis
      if (currentPrice > currentEMA) {
        bullishSignals++
        reasons.push("Price above EMA")
      } else {
        bearishSignals++
        reasons.push("Price below EMA")
      }

      // MACD analysis
      if (currentMACD.macd > currentMACD.signal) {
        bullishSignals++
        reasons.push("MACD bullish")
      } else {
        bearishSignals++
        reasons.push("MACD bearish")
      }

      // Generate signal if 2 or more indicators agree
      if (bullishSignals >= 2) {
        signals.push({
          id: `multi_buy_${symbol}_${Date.now()}`,
          symbol: symbol,
          action: "BUY",
          price: currentPrice,
          confidence: Math.min(0.9, 0.5 + bullishSignals * 0.1),
          reasoning: reasons.join(", "),
        })
      } else if (bearishSignals >= 2) {
        signals.push({
          id: `multi_sell_${symbol}_${Date.now()}`,
          symbol: symbol,
          action: "SELL",
          price: currentPrice,
          confidence: Math.min(0.9, 0.5 + bearishSignals * 0.1),
          reasoning: reasons.join(", "),
        })
      }
    }

    return signals
  }

  // Technical Indicator Calculations
  calculateEMA(prices, period) {
    const ema = []
    const multiplier = 2 / (period + 1)
    ema[0] = prices[0]

    for (let i = 1; i < prices.length; i++) {
      ema[i] = prices[i] * multiplier + ema[i - 1] * (1 - multiplier)
    }

    return ema.slice(period - 1)
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

  calculateBollingerBands(prices, period = 20, stdDev = 2) {
    const bands = []

    for (let i = period - 1; i < prices.length; i++) {
      const slice = prices.slice(i - period + 1, i + 1)
      const mean = slice.reduce((a, b) => a + b, 0) / period
      const variance = slice.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / period
      const standardDeviation = Math.sqrt(variance)

      bands.push({
        upper: mean + standardDeviation * stdDev,
        middle: mean,
        lower: mean - standardDeviation * stdDev,
      })
    }

    return bands
  }

  calculateMACD(prices, fastPeriod = 12, slowPeriod = 26, signalPeriod = 9) {
    const fastEMA = this.calculateEMA(prices, fastPeriod)
    const slowEMA = this.calculateEMA(prices, slowPeriod)

    const macdLine = []
    const startIndex = slowPeriod - fastPeriod

    for (let i = 0; i < slowEMA.length; i++) {
      macdLine.push(fastEMA[i + startIndex] - slowEMA[i])
    }

    const signalLine = this.calculateEMA(macdLine, signalPeriod)
    const histogram = []
    const signalStartIndex = signalPeriod - 1

    for (let i = 0; i < signalLine.length; i++) {
      histogram.push({
        macd: macdLine[i + signalStartIndex],
        signal: signalLine[i],
        histogram: macdLine[i + signalStartIndex] - signalLine[i],
      })
    }

    return histogram
  }

  async getMarketData() {
    // This would typically fetch real market data
    // For now, return mock data structure
    return {
      BTCUSDT: {
        candles: [], // Would contain actual OHLCV data
        volume: 0,
        timestamp: new Date(),
      },
    }
  }

  updateStrategies(strategyConfigs) {
    // Update active strategies
    this.activeStrategies.clear()

    for (const config of strategyConfigs) {
      if (this.strategies.has(config.name)) {
        this.activeStrategies.add(config.name)

        if (config.parameters) {
          const strategy = this.strategies.get(config.name)
          strategy.parameters = { ...strategy.parameters, ...config.parameters }
        }
      }
    }

    // Restart strategy execution with new configuration
    if (this.isRunning) {
      this.stop()
      this.start()
    }
  }

  getActiveStrategies() {
    return Array.from(this.activeStrategies).map((name) => ({
      name,
      ...this.strategies.get(name),
    }))
  }

  getAvailableStrategies() {
    return Array.from(this.strategies.entries()).map(([name, strategy]) => ({
      name,
      ...strategy,
    }))
  }
}

module.exports = StrategyManager
