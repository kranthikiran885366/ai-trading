"use client"

import type React from "react"
import { useState, useEffect } from "react"
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  Box,
  LinearProgress,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
} from "@mui/material"
import { TrendingUp, TrendingDown, PlayArrow, Stop, AccountBalance, ShowChart, Assessment } from "@mui/icons-material"
import { XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from "recharts"
import ApiService from "../services/ApiService"

interface DashboardProps {
  botStatus: any
  realTimeData: any
  onStartBot: () => void
  onStopBot: () => void
  onShowNotification: (message: string, severity: "success" | "error" | "warning" | "info") => void
}

const Dashboard: React.FC<DashboardProps> = ({
  botStatus,
  realTimeData,
  onStartBot,
  onStopBot,
  onShowNotification,
}) => {
  const [performanceData, setPerformanceData] = useState<any[]>([])
  const [recentTrades, setRecentTrades] = useState<any[]>([])
  const [aiInsights, setAiInsights] = useState<any>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    fetchDashboardData()
    const interval = setInterval(fetchDashboardData, 30000) // Update every 30 seconds
    return () => clearInterval(interval)
  }, [])

  const fetchDashboardData = async () => {
    try {
      setLoading(true)
      const [performance, trades, insights] = await Promise.all([
        ApiService.getPerformance(),
        ApiService.getTrades(10),
        ApiService.getAiInsights(),
      ])

      setPerformanceData(performance.daily_performance || [])
      setRecentTrades(trades)
      setAiInsights(insights)
    } catch (error) {
      console.error("Error fetching dashboard data:", error)
      onShowNotification("Failed to fetch dashboard data", "error")
    } finally {
      setLoading(false)
    }
  }

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
    }).format(value)
  }

  const formatPercentage = (value: number) => {
    return `${(value * 100).toFixed(2)}%`
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case "completed":
        return "success"
      case "failed":
        return "error"
      case "pending":
        return "warning"
      default:
        return "default"
    }
  }

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Typography variant="h4" gutterBottom>
        Trading Bot Dashboard
      </Typography>

      {loading && <LinearProgress sx={{ mb: 2 }} />}

      {/* Control Panel */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <Typography variant="h6">Bot Control</Typography>
            <Box sx={{ display: "flex", gap: 2 }}>
              <Button
                variant="contained"
                color="success"
                startIcon={<PlayArrow />}
                onClick={onStartBot}
                disabled={botStatus.is_running}
              >
                Start Bot
              </Button>
              <Button
                variant="contained"
                color="error"
                startIcon={<Stop />}
                onClick={onStopBot}
                disabled={!botStatus.is_running}
              >
                Stop Bot
              </Button>
            </Box>
          </Box>
        </CardContent>
      </Card>

      {/* Key Metrics */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: "flex", alignItems: "center", mb: 1 }}>
                <AccountBalance sx={{ mr: 1, color: "primary.main" }} />
                <Typography variant="h6">Portfolio Value</Typography>
              </Box>
              <Typography variant="h4" color="primary">
                {formatCurrency(botStatus.portfolio_value)}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Total portfolio value
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: "flex", alignItems: "center", mb: 1 }}>
                <ShowChart sx={{ mr: 1, color: "success.main" }} />
                <Typography variant="h6">Success Rate</Typography>
              </Box>
              <Typography variant="h4" color="success.main">
                {formatPercentage(botStatus.success_rate)}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Trading success rate
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: "flex", alignItems: "center", mb: 1 }}>
                {botStatus.daily_pnl >= 0 ? (
                  <TrendingUp sx={{ mr: 1, color: "success.main" }} />
                ) : (
                  <TrendingDown sx={{ mr: 1, color: "error.main" }} />
                )}
                <Typography variant="h6">Daily P&L</Typography>
              </Box>
              <Typography variant="h4" color={botStatus.daily_pnl >= 0 ? "success.main" : "error.main"}>
                {formatCurrency(botStatus.daily_pnl)}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Today's profit/loss
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: "flex", alignItems: "center", mb: 1 }}>
                <Assessment sx={{ mr: 1, color: "info.main" }} />
                <Typography variant="h6">Active Positions</Typography>
              </Box>
              <Typography variant="h4" color="info.main">
                {botStatus.active_positions}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Currently open positions
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Performance Chart */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Portfolio Performance
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={performanceData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip formatter={(value) => formatCurrency(Number(value))} />
                  <Area type="monotone" dataKey="portfolio_value" stroke="#00ff88" fill="#00ff88" fillOpacity={0.3} />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                AI Insights
              </Typography>
              {aiInsights ? (
                <Box>
                  <Typography variant="body2" gutterBottom>
                    Model Confidence: {formatPercentage(aiInsights.confidence || 0)}
                  </Typography>
                  <Typography variant="body2" gutterBottom>
                    Market Sentiment: {aiInsights.market_sentiment || "Neutral"}
                  </Typography>
                  <Typography variant="body2" gutterBottom>
                    Recommended Action: {aiInsights.recommendation || "Hold"}
                  </Typography>
                  <Typography variant="body2" gutterBottom>
                    Risk Level: {aiInsights.risk_level || "Medium"}
                  </Typography>
                </Box>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  Loading AI insights...
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Recent Trades */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Recent Trades
          </Typography>
          <TableContainer component={Paper} sx={{ backgroundColor: "background.paper" }}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Symbol</TableCell>
                  <TableCell>Type</TableCell>
                  <TableCell>Quantity</TableCell>
                  <TableCell>Price</TableCell>
                  <TableCell>P&L</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Time</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {recentTrades.map((trade, index) => (
                  <TableRow key={index}>
                    <TableCell>{trade.symbol}</TableCell>
                    <TableCell>
                      <Chip label={trade.side} color={trade.side === "buy" ? "success" : "error"} size="small" />
                    </TableCell>
                    <TableCell>{trade.quantity}</TableCell>
                    <TableCell>{formatCurrency(trade.price)}</TableCell>
                    <TableCell
                      sx={{
                        color: trade.pnl >= 0 ? "success.main" : "error.main",
                      }}
                    >
                      {formatCurrency(trade.pnl)}
                    </TableCell>
                    <TableCell>
                      <Chip label={trade.status} color={getStatusColor(trade.status)} size="small" />
                    </TableCell>
                    <TableCell>{new Date(trade.timestamp).toLocaleTimeString()}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>
    </Box>
  )
}

export default Dashboard
