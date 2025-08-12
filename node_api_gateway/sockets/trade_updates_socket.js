const socketIo = require("socket.io")

class TradeUpdatesSocket {
  constructor(server) {
    this.io = socketIo(server, {
      cors: {
        origin: "*",
        methods: ["GET", "POST"],
      },
    })

    this.connectedClients = new Map()
    this.setupSocketHandlers()

    console.log("🔌 Trade Updates Socket initialized")
  }

  setupSocketHandlers() {
    this.io.on("connection", (socket) => {
      console.log(`📱 Client connected: ${socket.id}`)

      // Store client info
      this.connectedClients.set(socket.id, {
        id: socket.id,
        connectedAt: new Date(),
        subscriptions: new Set(),
      })

      // Handle client authentication
      socket.on("authenticate", (data) => {
        const client = this.connectedClients.get(socket.id)
        if (client) {
          client.userId = data.userId
          client.authenticated = true
          socket.emit("authenticated", { success: true })
          console.log(`🔐 Client ${socket.id} authenticated as user ${data.userId}`)
        }
      })

      // Handle subscription to specific data feeds
      socket.on("subscribe", (data) => {
        const { feeds } = data
        const client = this.connectedClients.get(socket.id)

        if (client && feeds) {
          feeds.forEach((feed) => {
            client.subscriptions.add(feed)
            socket.join(feed)
          })

          socket.emit("subscribed", { feeds })
          console.log(`📡 Client ${socket.id} subscribed to: ${feeds.join(", ")}`)
        }
      })

      // Handle unsubscription
      socket.on("unsubscribe", (data) => {
        const { feeds } = data
        const client = this.connectedClients.get(socket.id)

        if (client && feeds) {
          feeds.forEach((feed) => {
            client.subscriptions.delete(feed)
            socket.leave(feed)
          })

          socket.emit("unsubscribed", { feeds })
          console.log(`📡 Client ${socket.id} unsubscribed from: ${feeds.join(", ")}`)
        }
      })

      // Handle real-time trade requests
      socket.on("requestTradeUpdate", (data) => {
        const { symbol } = data
        if (symbol) {
          // Send latest trade data for the symbol
          this.sendTradeUpdate(socket.id, symbol)
        }
      })

      // Handle portfolio update requests
      socket.on("requestPortfolioUpdate", () => {
        this.sendPortfolioUpdate(socket.id)
      })

      // Handle market data requests
      socket.on("requestMarketData", (data) => {
        const { symbols } = data
        if (symbols && Array.isArray(symbols)) {
          this.sendMarketData(socket.id, symbols)
        }
      })

      // Handle bot status requests
      socket.on("requestBotStatus", () => {
        this.sendBotStatus(socket.id)
      })

      // Handle disconnection
      socket.on("disconnect", (reason) => {
        console.log(`📱 Client disconnected: ${socket.id}, reason: ${reason}`)
        this.connectedClients.delete(socket.id)
      })

      // Handle errors
      socket.on("error", (error) => {
        console.error(`❌ Socket error for client ${socket.id}:`, error)
      })

      // Send initial connection data
      socket.emit("connected", {
        clientId: socket.id,
        timestamp: new Date(),
        availableFeeds: this.getAvailableFeeds(),
      })
    })
  }

  // Broadcast trade updates to all subscribed clients
  broadcastTradeUpdate(tradeData) {
    this.io.to("trades").emit("tradeUpdate", {
      ...tradeData,
      timestamp: new Date(),
    })

    console.log(`📊 Broadcasted trade update: ${tradeData.symbol} ${tradeData.action}`)
  }

  // Broadcast portfolio updates
  broadcastPortfolioUpdate(portfolioData) {
    this.io.to("portfolio").emit("portfolioUpdate", {
      ...portfolioData,
      timestamp: new Date(),
    })

    console.log("💼 Broadcasted portfolio update")
  }

  // Broadcast market data updates
  broadcastMarketData(marketData) {
    this.io.to("market").emit("marketData", {
      ...marketData,
      timestamp: new Date(),
    })
  }

  // Broadcast bot status updates
  broadcastBotStatus(statusData) {
    this.io.emit("botStatus", {
      ...statusData,
      timestamp: new Date(),
    })

    console.log(`🤖 Broadcasted bot status: ${statusData.status}`)
  }

  // Broadcast alerts
  broadcastAlert(alertData) {
    this.io.emit("alert", {
      ...alertData,
      timestamp: new Date(),
    })

    console.log(`🚨 Broadcasted alert: ${alertData.type} - ${alertData.message}`)
  }

  // Broadcast performance updates
  broadcastPerformanceUpdate(performanceData) {
    this.io.to("performance").emit("performanceUpdate", {
      ...performanceData,
      timestamp: new Date(),
    })

    console.log("📈 Broadcasted performance update")
  }

  // Broadcast AI signals
  broadcastAISignal(signalData) {
    this.io.to("signals").emit("aiSignal", {
      ...signalData,
      timestamp: new Date(),
    })

    console.log(`🧠 Broadcasted AI signal: ${signalData.symbol} ${signalData.action}`)
  }

  // Broadcast risk alerts
  broadcastRiskAlert(riskData) {
    this.io.emit("riskAlert", {
      ...riskData,
      timestamp: new Date(),
    })

    console.log(`⚠️ Broadcasted risk alert: ${riskData.type}`)
  }

  // Send trade update to specific client
  sendTradeUpdate(clientId, symbol) {
    const socket = this.io.sockets.sockets.get(clientId)
    if (socket) {
      // Get latest trade data for symbol (mock data for now)
      const tradeData = {
        symbol,
        price: Math.random() * 1000 + 100,
        volume: Math.random() * 10000,
        change: (Math.random() - 0.5) * 10,
        timestamp: new Date(),
      }

      socket.emit("tradeUpdate", tradeData)
    }
  }

  // Send portfolio update to specific client
  sendPortfolioUpdate(clientId) {
    const socket = this.io.sockets.sockets.get(clientId)
    if (socket) {
      // Get portfolio data (mock data for now)
      const portfolioData = {
        totalValue: 50000 + Math.random() * 10000,
        totalPnL: (Math.random() - 0.5) * 5000,
        positions: [
          { symbol: "AAPL", quantity: 10, value: 1500 },
          { symbol: "GOOGL", quantity: 5, value: 12500 },
        ],
        timestamp: new Date(),
      }

      socket.emit("portfolioUpdate", portfolioData)
    }
  }

  // Send market data to specific client
  sendMarketData(clientId, symbols) {
    const socket = this.io.sockets.sockets.get(clientId)
    if (socket) {
      const marketData = symbols.map((symbol) => ({
        symbol,
        price: Math.random() * 1000 + 100,
        change: (Math.random() - 0.5) * 10,
        volume: Math.random() * 1000000,
        timestamp: new Date(),
      }))

      socket.emit("marketData", marketData)
    }
  }

  // Send bot status to specific client
  sendBotStatus(clientId) {
    const socket = this.io.sockets.sockets.get(clientId)
    if (socket) {
      const statusData = {
        isActive: true,
        uptime: process.uptime(),
        totalTrades: Math.floor(Math.random() * 100),
        winRate: Math.random() * 100,
        timestamp: new Date(),
      }

      socket.emit("botStatus", statusData)
    }
  }

  // Get available data feeds
  getAvailableFeeds() {
    return ["trades", "portfolio", "market", "performance", "signals", "alerts"]
  }

  // Get connected clients count
  getConnectedClientsCount() {
    return this.connectedClients.size
  }

  // Get client statistics
  getClientStatistics() {
    const stats = {
      totalClients: this.connectedClients.size,
      authenticatedClients: 0,
      subscriptionStats: {},
    }

    this.connectedClients.forEach((client) => {
      if (client.authenticated) {
        stats.authenticatedClients++
      }

      client.subscriptions.forEach((subscription) => {
        stats.subscriptionStats[subscription] = (stats.subscriptionStats[subscription] || 0) + 1
      })
    })

    return stats
  }

  // Broadcast system notification
  broadcastSystemNotification(notification) {
    this.io.emit("systemNotification", {
      ...notification,
      timestamp: new Date(),
    })

    console.log(`📢 Broadcasted system notification: ${notification.message}`)
  }

  // Send private message to specific user
  sendPrivateMessage(userId, message) {
    this.connectedClients.forEach((client, socketId) => {
      if (client.userId === userId) {
        const socket = this.io.sockets.sockets.get(socketId)
        if (socket) {
          socket.emit("privateMessage", {
            message,
            timestamp: new Date(),
          })
        }
      }
    })
  }

  // Broadcast emergency notification
  broadcastEmergencyNotification(notification) {
    this.io.emit("emergency", {
      ...notification,
      timestamp: new Date(),
    })

    console.log(`🚨 Broadcasted emergency notification: ${notification.message}`)
  }
}

module.exports = TradeUpdatesSocket
