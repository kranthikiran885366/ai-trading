class RiskManager {
  constructor() {
    this.maxDailyLoss = 1000 // Default $1000 max daily loss
    this.maxDrawdown = 0.15 // 15% max drawdown
    this.maxPositionSize = 0.05 // 5% of portfolio per position
    this.maxOpenPositions = 10
    this.riskPerTrade = 0.02 // 2% risk per trade
    this.stopLossMultiplier = 2 // 2x ATR for stop loss
    this.takeProfitMultiplier = 3 // 3x ATR for take profit

    this.dailyLoss = 0
    this.currentDrawdown = 0
    this.openPositions = 0
    this.portfolioValue = 10000 // Default portfolio value

    this.riskMetrics = {
      sharpeRatio: 0,
      maxDrawdownPeriod: 0,
      winRate: 0,
      profitFactor: 0,
      averageWin: 0,
      averageLoss: 0,
    }
  }

  setParameters(config) {
    if (config.maxDailyLoss) this.maxDailyLoss = config.maxDailyLoss
    if (config.maxDrawdown) this.maxDrawdown = config.maxDrawdown
    if (config.maxPositionSize) this.maxPositionSize = config.maxPositionSize
    if (config.maxOpenPositions) this.maxOpenPositions = config.maxOpenPositions
    if (config.riskPerTrade) this.riskPerTrade = config.riskPerTrade
    if (config.portfolioValue) this.portfolioValue = config.portfolioValue

    console.log("🛡️ Risk parameters updated:", config)
  }

  async assessSignal(signal) {
    const assessment = {
      approved: true,
      reason: "",
      riskScore: 0,
      adjustedQuantity: signal.quantity || 0,
    }

    // Check daily loss limit
    if (this.dailyLoss >= this.maxDailyLoss) {
      assessment.approved = false
      assessment.reason = "Daily loss limit exceeded"
      return assessment
    }

    // Check maximum drawdown
    if (this.currentDrawdown >= this.maxDrawdown) {
      assessment.approved = false
      assessment.reason = "Maximum drawdown exceeded"
      return assessment
    }

    // Check maximum open positions
    if (this.openPositions >= this.maxOpenPositions) {
      assessment.approved = false
      assessment.reason = "Maximum open positions reached"
      return assessment
    }

    // Check signal quality
    if (signal.confidence < 0.6) {
      assessment.approved = false
      assessment.reason = "Signal confidence too low"
      return assessment
    }

    // Calculate risk score based on multiple factors
    assessment.riskScore = this.calculateRiskScore(signal)

    // Adjust position size based on risk
    assessment.adjustedQuantity = this.calculatePositionSize(signal, assessment)

    return assessment
  }

  calculateRiskScore(signal) {
    let riskScore = 0

    // Market volatility risk
    if (signal.marketRegime === "VOLATILE") {
      riskScore += 0.3
    } else if (signal.marketRegime === "BEAR") {
      riskScore += 0.2
    }

    // Signal confidence risk (inverse)
    riskScore += (1 - signal.confidence) * 0.3

    // Time-based risk (higher risk during market open/close)
    const hour = new Date().getHours()
    if ((hour >= 9 && hour <= 10) || (hour >= 15 && hour <= 16)) {
      riskScore += 0.1 // Higher volatility during market open/close
    }

    // Portfolio concentration risk
    const concentrationRisk = this.openPositions / this.maxOpenPositions
    riskScore += concentrationRisk * 0.2

    return Math.min(1, riskScore)
  }

  calculatePositionSize(signal, riskAssessment) {
    // Base position size on portfolio percentage
    let baseSize = this.portfolioValue * this.maxPositionSize

    // Adjust for risk per trade
    const riskAmount = this.portfolioValue * this.riskPerTrade
    const stopLossDistance = Math.abs(signal.price - signal.stopLoss)

    if (stopLossDistance > 0) {
      const riskBasedSize = riskAmount / stopLossDistance
      baseSize = Math.min(baseSize, riskBasedSize)
    }

    // Adjust for signal confidence
    baseSize *= signal.confidence

    // Adjust for risk score
    baseSize *= 1 - riskAssessment.riskScore

    // Adjust for market regime
    if (signal.marketRegime === "VOLATILE") {
      baseSize *= 0.7
    } else if (signal.marketRegime === "BEAR") {
      baseSize *= 0.8
    }

    // Convert to shares/units
    const quantity = Math.floor(baseSize / signal.price)

    return Math.max(1, quantity) // Minimum 1 unit
  }

  updatePortfolioMetrics(trades) {
    if (!trades || trades.length === 0) return

    const completedTrades = trades.filter((t) => t.status === "CLOSED")

    if (completedTrades.length === 0) return

    // Calculate win rate
    const winningTrades = completedTrades.filter((t) => t.pnl > 0)
    this.riskMetrics.winRate = (winningTrades.length / completedTrades.length) * 100

    // Calculate average win/loss
    const wins = winningTrades.map((t) => t.pnl)
    const losses = completedTrades.filter((t) => t.pnl < 0).map((t) => Math.abs(t.pnl))

    this.riskMetrics.averageWin = wins.length > 0 ? wins.reduce((a, b) => a + b, 0) / wins.length : 0

    this.riskMetrics.averageLoss = losses.length > 0 ? losses.reduce((a, b) => a + b, 0) / losses.length : 0

    // Calculate profit factor
    const totalWins = wins.reduce((a, b) => a + b, 0)
    const totalLosses = losses.reduce((a, b) => a + b, 0)

    this.riskMetrics.profitFactor = totalLosses > 0 ? totalWins / totalLosses : 0

    // Calculate Sharpe ratio (simplified)
    const returns = completedTrades.map((t) => t.pnl / this.portfolioValue)
    const avgReturn = returns.reduce((a, b) => a + b, 0) / returns.length
    const returnStdDev = this.calculateStandardDeviation(returns)

    this.riskMetrics.sharpeRatio = returnStdDev > 0 ? avgReturn / returnStdDev : 0

    console.log("📊 Risk metrics updated:", this.riskMetrics)
  }

  calculateStandardDeviation(values) {
    const mean = values.reduce((a, b) => a + b, 0) / values.length
    const squaredDiffs = values.map((value) => Math.pow(value - mean, 2))
    const avgSquaredDiff = squaredDiffs.reduce((a, b) => a + b, 0) / squaredDiffs.length
    return Math.sqrt(avgSquaredDiff)
  }

  checkRiskLimits(currentPnl, positions) {
    const alerts = []

    // Check daily loss
    if (currentPnl < -this.maxDailyLoss) {
      alerts.push({
        type: "DAILY_LOSS_LIMIT",
        severity: "CRITICAL",
        message: `Daily loss limit exceeded: ${currentPnl}`,
        action: "STOP_TRADING",
      })
    }

    // Check drawdown
    const drawdown = this.calculateCurrentDrawdown(currentPnl)
    if (drawdown > this.maxDrawdown) {
      alerts.push({
        type: "DRAWDOWN_LIMIT",
        severity: "HIGH",
        message: `Drawdown limit exceeded: ${(drawdown * 100).toFixed(2)}%`,
        action: "REDUCE_POSITION_SIZE",
      })
    }

    // Check position concentration
    const totalExposure = positions.reduce((sum, pos) => sum + Math.abs(pos.quantity * pos.avgPrice), 0)

    const exposureRatio = totalExposure / this.portfolioValue
    if (exposureRatio > 0.8) {
      alerts.push({
        type: "HIGH_EXPOSURE",
        severity: "MEDIUM",
        message: `High portfolio exposure: ${(exposureRatio * 100).toFixed(2)}%`,
        action: "REDUCE_NEW_POSITIONS",
      })
    }

    return alerts
  }

  calculateCurrentDrawdown(currentPnl) {
    // Simplified drawdown calculation
    const peak = Math.max(this.portfolioValue + currentPnl, this.portfolioValue)
    const current = this.portfolioValue + currentPnl
    return (peak - current) / peak
  }

  adjustStopLoss(position, currentPrice, atr) {
    if (!position.stopLoss) return null

    const isLong = position.quantity > 0
    let newStopLoss

    if (isLong) {
      // Trailing stop for long positions
      newStopLoss = currentPrice - atr * this.stopLossMultiplier
      return newStopLoss > position.stopLoss ? newStopLoss : position.stopLoss
    } else {
      // Trailing stop for short positions
      newStopLoss = currentPrice + atr * this.stopLossMultiplier
      return newStopLoss < position.stopLoss ? newStopLoss : position.stopLoss
    }
  }

  shouldHedgePosition(position, marketData) {
    // Simple hedging logic
    const unrealizedLoss = position.unrealizedPnl
    const positionValue = Math.abs(position.quantity * position.avgPrice)
    const lossPercentage = Math.abs(unrealizedLoss) / positionValue

    // Hedge if loss exceeds 10% of position value
    return lossPercentage > 0.1
  }

  reducePositionSize(factor = 0.5) {
    this.maxPositionSize *= factor
    console.log(`🛡️ Position size reduced by ${(1 - factor) * 100}% to ${this.maxPositionSize}`)
  }

  resetDailyMetrics() {
    this.dailyLoss = 0
    console.log("🛡️ Daily risk metrics reset")
  }

  getRiskMetrics() {
    return {
      ...this.riskMetrics,
      dailyLoss: this.dailyLoss,
      currentDrawdown: this.currentDrawdown,
      openPositions: this.openPositions,
      maxDailyLoss: this.maxDailyLoss,
      maxDrawdown: this.maxDrawdown,
      maxPositionSize: this.maxPositionSize,
      portfolioValue: this.portfolioValue,
    }
  }
}

module.exports = RiskManager
