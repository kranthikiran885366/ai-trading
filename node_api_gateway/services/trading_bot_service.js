const EventEmitter = require("events")
const BrokerAPI = require("./broker_api_service")
const RiskManager = require("./risk_manager")
const StrategyManager = require("./strategy_manager")
const TradeLogger = require("./trade_logger")
const AlertService = require("./alert_service")

class TradingBotService extends EventEmitter {
  constructor(io) {
    super()
    this.io = io
    this.isActive = false
    this.positions = new Map()
    this.orders = new Map()
    this.pnl = 0
    this.dailyPnl = 0
    this.totalTrades = 0
    this.winningTrades = 0

    // Initialize components
    this.broker = new BrokerAPI()
    this.riskManager = new RiskManager()
    this.strategyManager = new StrategyManager()
    this.tradeLogger = new TradeLogger()
    this.alertService = new AlertService()

    this.setupEventHandlers()
    this.startMonitoring()
  }

  setupEventHandlers() {
    this.on("signal", this.handleTradingSignal.bind(this))
    this.on("positionUpdate", this.handlePositionUpdate.bind(this))
    this.on("orderFilled", this.handleOrderFilled.bind(this))
    this.on("riskAlert", this.handleRiskAlert.bind(this))
  }

  async start(config = {}) {
    try {
      console.log("🤖 Starting Trading Bot...")

      // Validate configuration
      if (!this.validateConfig(config)) {
        throw new Error("Invalid configuration")
      }

      // Initialize broker connection
      await this.broker.connect(config.brokerConfig)

      // Set risk parameters
      this.riskManager.setParameters(config.riskConfig)

      // Load strategies
      await this.strategyManager.loadStrategies(config.strategies)

      this.isActive = true

      // Start strategy execution
      this.strategyManager.start()

      // Emit status update
      this.emitStatusUpdate()

      console.log("✅ Trading Bot started successfully")
      return { success: true, message: "Bot started successfully" }
    } catch (error) {
      console.error("❌ Failed to start trading bot:", error)
      await this.alertService.sendAlert("CRITICAL", `Bot failed to start: ${error.message}`)
      throw error
    }
  }

  async stop() {
    try {
      console.log("🛑 Stopping Trading Bot...")

      this.isActive = false

      // Close all open positions
      await this.closeAllPositions()

      // Cancel pending orders
      await this.cancelAllOrders()

      // Stop strategies
      this.strategyManager.stop()

      // Disconnect broker
      await this.broker.disconnect()

      this.emitStatusUpdate()

      console.log("✅ Trading Bot stopped successfully")
      return { success: true, message: "Bot stopped successfully" }
    } catch (error) {
      console.error("❌ Error stopping trading bot:", error)
      throw error
    }
  }

  async handleTradingSignal(signal) {
    if (!this.isActive) return

    try {
      console.log("📊 Processing trading signal:", signal)

      // Risk assessment
      const riskAssessment = await this.riskManager.assessSignal(signal)
      if (!riskAssessment.approved) {
        console.log("⚠️ Signal rejected by risk manager:", riskAssessment.reason)
        return
      }

      // Calculate position size
      const positionSize = this.riskManager.calculatePositionSize(signal, riskAssessment)

      // Create order
      const order = {
        symbol: signal.symbol,
        side: signal.action, // 'BUY' or 'SELL'
        quantity: positionSize,
        type: signal.orderType || "MARKET",
        price: signal.price,
        stopLoss: signal.stopLoss,
        takeProfit: signal.takeProfit,
        strategy: signal.strategy,
        timestamp: new Date(),
        signalId: signal.id,
      }

      // Execute order
      const executedOrder = await this.broker.placeOrder(order)

      if (executedOrder.success) {
        this.orders.set(executedOrder.orderId, executedOrder)

        // Log trade
        await this.tradeLogger.logTrade({
          ...executedOrder,
          signal: signal,
          riskAssessment: riskAssessment,
        })

        // Send notification
        await this.alertService.sendAlert(
          "INFO",
          `Order placed: ${signal.action} ${positionSize} ${signal.symbol} at ${signal.price}`,
        )

        this.emitTradeUpdate(executedOrder)
      }
    } catch (error) {
      console.error("❌ Error handling trading signal:", error)
      await this.alertService.sendAlert("ERROR", `Signal processing failed: ${error.message}`)
    }
  }

  async handleOrderFilled(orderData) {
    try {
      const order = this.orders.get(orderData.orderId)
      if (!order) return

      // Update position
      const position = this.positions.get(orderData.symbol) || {
        symbol: orderData.symbol,
        quantity: 0,
        avgPrice: 0,
        unrealizedPnl: 0,
        realizedPnl: 0,
      }

      if (orderData.side === "BUY") {
        const totalCost = position.quantity * position.avgPrice + orderData.quantity * orderData.price
        const totalQuantity = position.quantity + orderData.quantity
        position.avgPrice = totalCost / totalQuantity
        position.quantity = totalQuantity
      } else {
        position.quantity -= orderData.quantity
        position.realizedPnl += (orderData.price - position.avgPrice) * orderData.quantity
      }

      this.positions.set(orderData.symbol, position)

      // Update statistics
      this.totalTrades++
      if (position.realizedPnl > 0) {
        this.winningTrades++
      }

      this.emitPositionUpdate(position)
    } catch (error) {
      console.error("❌ Error handling order fill:", error)
    }
  }

  async closeAllPositions() {
    const closePromises = []

    for (const [symbol, position] of this.positions) {
      if (position.quantity !== 0) {
        const closeOrder = {
          symbol: symbol,
          side: position.quantity > 0 ? "SELL" : "BUY",
          quantity: Math.abs(position.quantity),
          type: "MARKET",
        }

        closePromises.push(this.broker.placeOrder(closeOrder))
      }
    }

    await Promise.all(closePromises)
  }

  async cancelAllOrders() {
    const cancelPromises = []

    for (const [orderId, order] of this.orders) {
      if (order.status === "PENDING") {
        cancelPromises.push(this.broker.cancelOrder(orderId))
      }
    }

    await Promise.all(cancelPromises)
  }

  startMonitoring() {
    // Monitor positions every 5 seconds
    setInterval(() => {
      if (this.isActive) {
        this.updatePositions()
        this.checkRiskLimits()
      }
    }, 5000)

    // Daily PnL reset
    setInterval(
      () => {
        this.resetDailyPnl()
      },
      24 * 60 * 60 * 1000,
    )
  }

  async updatePositions() {
    try {
      const marketData = await this.broker.getMarketData([...this.positions.keys()])

      for (const [symbol, position] of this.positions) {
        const currentPrice = marketData[symbol]?.price
        if (currentPrice && position.quantity !== 0) {
          position.unrealizedPnl = (currentPrice - position.avgPrice) * position.quantity
          this.emitPositionUpdate(position)
        }
      }
    } catch (error) {
      console.error("❌ Error updating positions:", error)
    }
  }

  checkRiskLimits() {
    const totalUnrealizedPnl = Array.from(this.positions.values()).reduce((sum, pos) => sum + pos.unrealizedPnl, 0)

    const totalPnl = this.dailyPnl + totalUnrealizedPnl

    // Check daily loss limit
    if (totalPnl < -this.riskManager.maxDailyLoss) {
      this.emit("riskAlert", {
        type: "DAILY_LOSS_LIMIT",
        message: `Daily loss limit exceeded: ${totalPnl}`,
        action: "STOP_TRADING",
      })
    }

    // Check drawdown limit
    const drawdown = this.calculateDrawdown()
    if (drawdown > this.riskManager.maxDrawdown) {
      this.emit("riskAlert", {
        type: "DRAWDOWN_LIMIT",
        message: `Maximum drawdown exceeded: ${drawdown}%`,
        action: "REDUCE_POSITION_SIZE",
      })
    }
  }

  calculateDrawdown() {
    // Simplified drawdown calculation
    const peak = Math.max(this.pnl, 0)
    const current = this.pnl
    return peak > 0 ? ((peak - current) / peak) * 100 : 0
  }

  async handleRiskAlert(alert) {
    console.log("🚨 Risk Alert:", alert)

    await this.alertService.sendAlert("CRITICAL", alert.message)

    if (alert.action === "STOP_TRADING") {
      await this.stop()
    } else if (alert.action === "REDUCE_POSITION_SIZE") {
      this.riskManager.reducePositionSize(0.5)
    }
  }

  validateConfig(config) {
    return config.brokerConfig && config.riskConfig && config.strategies && Array.isArray(config.strategies)
  }

  resetDailyPnl() {
    this.dailyPnl = 0
    console.log("📅 Daily PnL reset")
  }

  emitStatusUpdate() {
    this.io.emit("botStatus", {
      isActive: this.isActive,
      totalTrades: this.totalTrades,
      winRate: this.totalTrades > 0 ? (this.winningTrades / this.totalTrades) * 100 : 0,
      pnl: this.pnl,
      dailyPnl: this.dailyPnl,
      positions: Array.from(this.positions.values()),
      timestamp: new Date(),
    })
  }

  emitTradeUpdate(trade) {
    this.io.emit("tradeUpdate", trade)
  }

  emitPositionUpdate(position) {
    this.io.emit("positionUpdate", position)
  }

  isRunning() {
    return this.isActive
  }

  getStatus() {
    return {
      isActive: this.isActive,
      totalTrades: this.totalTrades,
      winRate: this.totalTrades > 0 ? (this.winningTrades / this.totalTrades) * 100 : 0,
      pnl: this.pnl,
      dailyPnl: this.dailyPnl,
      positions: Array.from(this.positions.values()),
      orders: Array.from(this.orders.values()),
    }
  }
}

module.exports = TradingBotService
