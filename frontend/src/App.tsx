"use client"

import type React from "react"
import { useState, useEffect } from "react"
import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom"
import { ThemeProvider, createTheme } from "@mui/material/styles"
import CssBaseline from "@mui/material/CssBaseline"
import { Box, AppBar, Toolbar, Typography, Alert, Snackbar } from "@mui/material"
import Dashboard from "./components/Dashboard"
import TradingView from "./components/TradingView"
import Portfolio from "./components/Portfolio"
import Settings from "./components/Settings"
import Navigation from "./components/Navigation"
import WebSocketService from "./services/WebSocketService"
import ApiService from "./services/ApiService"

const theme = createTheme({
  palette: {
    mode: "dark",
    primary: {
      main: "#00ff88",
    },
    secondary: {
      main: "#ff4444",
    },
    background: {
      default: "#0a0a0a",
      paper: "#1a1a1a",
    },
  },
  typography: {
    fontFamily: '"Roboto Mono", "Helvetica", "Arial", sans-serif',
  },
})

interface BotStatus {
  is_running: boolean
  success_rate: number
  total_trades: number
  portfolio_value: number
  daily_pnl: number
  uptime: number
  active_positions: number
}

interface RealTimeData {
  bot_status: BotStatus
  portfolio: any
  latest_signals: any[]
  market_overview: any
  timestamp: string
}

const App: React.FC = () => {
  const [botStatus, setBotStatus] = useState<BotStatus>({
    is_running: false,
    success_rate: 0,
    total_trades: 0,
    portfolio_value: 100000,
    daily_pnl: 0,
    uptime: 0,
    active_positions: 0,
  })

  const [realTimeData, setRealTimeData] = useState<RealTimeData | null>(null)
  const [connected, setConnected] = useState(false)
  const [notification, setNotification] = useState<{
    open: boolean
    message: string
    severity: "success" | "error" | "warning" | "info"
  }>({
    open: false,
    message: "",
    severity: "info",
  })

  useEffect(() => {
    // Initialize WebSocket connection
    const wsService = WebSocketService.getInstance()

    wsService.connect("ws://localhost:8000/ws")

    wsService.onMessage((data: RealTimeData) => {
      setRealTimeData(data)
      if (data.bot_status) {
        setBotStatus(data.bot_status)
      }
    })

    wsService.onConnect(() => {
      setConnected(true)
      showNotification("Connected to trading bot", "success")
    })

    wsService.onDisconnect(() => {
      setConnected(false)
      showNotification("Disconnected from trading bot", "warning")
    })

    wsService.onError((error) => {
      showNotification(`WebSocket error: ${error}`, "error")
    })

    // Fetch initial data
    fetchInitialData()

    return () => {
      wsService.disconnect()
    }
  }, [])

  const fetchInitialData = async () => {
    try {
      const status = await ApiService.getBotStatus()
      setBotStatus(status)
    } catch (error) {
      console.error("Error fetching initial data:", error)
      showNotification("Failed to fetch initial data", "error")
    }
  }

  const handleStartBot = async () => {
    try {
      await ApiService.startBot()
      const status = await ApiService.getBotStatus()
      setBotStatus(status)
      showNotification("Trading bot started successfully", "success")
    } catch (error) {
      console.error("Error starting bot:", error)
      showNotification("Failed to start trading bot", "error")
    }
  }

  const handleStopBot = async () => {
    try {
      await ApiService.stopBot()
      const status = await ApiService.getBotStatus()
      setBotStatus(status)
      showNotification("Trading bot stopped successfully", "success")
    } catch (error) {
      console.error("Error stopping bot:", error)
      showNotification("Failed to stop trading bot", "error")
    }
  }

  const showNotification = (message: string, severity: "success" | "error" | "warning" | "info") => {
    setNotification({
      open: true,
      message,
      severity,
    })
  }

  const handleCloseNotification = () => {
    setNotification((prev) => ({ ...prev, open: false }))
  }

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Box sx={{ display: "flex", flexDirection: "column", minHeight: "100vh" }}>
          <AppBar position="static" sx={{ backgroundColor: "#1a1a1a" }}>
            <Toolbar>
              <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
                🤖 Advanced AI Trading Bot
              </Typography>
              <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
                <Typography variant="body2" color={connected ? "success.main" : "error.main"}>
                  {connected ? "🟢 Connected" : "🔴 Disconnected"}
                </Typography>
                <Typography variant="body2">Success Rate: {(botStatus.success_rate * 100).toFixed(1)}%</Typography>
                <Typography variant="body2">Portfolio: ${botStatus.portfolio_value.toLocaleString()}</Typography>
                <Typography variant="body2" color={botStatus.daily_pnl >= 0 ? "success.main" : "error.main"}>
                  Daily P&L: ${botStatus.daily_pnl.toFixed(2)}
                </Typography>
                <Typography variant="body2">Status: {botStatus.is_running ? "🟢 Running" : "🔴 Stopped"}</Typography>
              </Box>
            </Toolbar>
          </AppBar>

          <Box sx={{ display: "flex", flex: 1 }}>
            <Navigation />

            <Box component="main" sx={{ flexGrow: 1, p: 3 }}>
              <Routes>
                <Route
                  path="/"
                  element={
                    <Dashboard
                      botStatus={botStatus}
                      realTimeData={realTimeData}
                      onStartBot={handleStartBot}
                      onStopBot={handleStopBot}
                      onShowNotification={showNotification}
                    />
                  }
                />
                <Route
                  path="/trading"
                  element={<TradingView realTimeData={realTimeData} onShowNotification={showNotification} />}
                />
                <Route
                  path="/portfolio"
                  element={<Portfolio realTimeData={realTimeData} onShowNotification={showNotification} />}
                />
                <Route path="/settings" element={<Settings onShowNotification={showNotification} />} />
                <Route path="*" element={<Navigate to="/" replace />} />
              </Routes>
            </Box>
          </Box>

          <Snackbar
            open={notification.open}
            autoHideDuration={6000}
            onClose={handleCloseNotification}
            anchorOrigin={{ vertical: "bottom", horizontal: "right" }}
          >
            <Alert onClose={handleCloseNotification} severity={notification.severity} sx={{ width: "100%" }}>
              {notification.message}
            </Alert>
          </Snackbar>
        </Box>
      </Router>
    </ThemeProvider>
  )
}

export default App
