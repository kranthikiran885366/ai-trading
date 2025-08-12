const express = require("express")
const http = require("http")
const socketIo = require("socket.io")
const cors = require("cors")
const helmet = require("helmet")
const rateLimit = require("express-rate-limit")
const mongoose = require("mongoose")
const jwt = require("jsonwebtoken")
require("dotenv").config()

const app = express()
const server = http.createServer(app)
const io = socketIo(server, {
  cors: {
    origin: process.env.FRONTEND_URL || "http://localhost:3000",
    methods: ["GET", "POST"],
  },
})

// Middleware
app.use(helmet())
app.use(cors())
app.use(express.json({ limit: "10mb" }))
app.use(express.urlencoded({ extended: true }))

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // limit each IP to 100 requests per windowMs
})
app.use("/api/", limiter)

// Database connection
mongoose.connect(process.env.MONGODB_URI || "mongodb://localhost:27017/trading_bot", {
  useNewUrlParser: true,
  useUnifiedTopology: true,
})

// Import routes
const botControlRoutes = require("./routes/bot_control")
const tradeHistoryRoutes = require("./routes/trade_history")
const riskSettingsRoutes = require("./routes/risk_settings")
const aiServiceRoutes = require("./routes/ai_service")

// Import services
const TradingBotService = require("./services/trading_bot_service")
const AIService = require("./services/ai_service")
const DataStreamService = require("./services/data_stream_service")

// Initialize services
const tradingBot = new TradingBotService(io)
const aiService = new AIService()
const dataStream = new DataStreamService(io)

// Make services available to routes
app.locals.tradingBot = tradingBot
app.locals.aiService = aiService
app.locals.dataStream = dataStream

// Routes
app.use("/api/bot", botControlRoutes)
app.use("/api/trades", tradeHistoryRoutes)
app.use("/api/risk", riskSettingsRoutes)
app.use("/api/ai", aiServiceRoutes)

// WebSocket handling
require("./sockets/trade_updates_socket")(io, tradingBot, aiService)

// Error handling middleware
app.use((err, req, res, next) => {
  console.error("Error:", err)
  res.status(500).json({
    error: "Internal server error",
    message: process.env.NODE_ENV === "development" ? err.message : "Something went wrong",
  })
})

// Health check
app.get("/health", (req, res) => {
  res.json({
    status: "OK",
    timestamp: new Date().toISOString(),
    services: {
      database: mongoose.connection.readyState === 1 ? "connected" : "disconnected",
      tradingBot: tradingBot.isRunning() ? "running" : "stopped",
      aiService: aiService.isReady() ? "ready" : "initializing",
    },
  })
})

const PORT = process.env.PORT || 5000
server.listen(PORT, () => {
  console.log(`🚀 Trading Bot API Server running on port ${PORT}`)

  // Initialize services
  aiService.initialize()
  dataStream.connect()
})

// Graceful shutdown
process.on("SIGTERM", () => {
  console.log("SIGTERM received, shutting down gracefully")
  tradingBot.stop()
  dataStream.disconnect()
  server.close(() => {
    mongoose.connection.close()
    process.exit(0)
  })
})

module.exports = app
