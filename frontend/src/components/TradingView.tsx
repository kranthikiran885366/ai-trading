"use client"

import type React from "react"
import { useState, useEffect } from "react"
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from "@mui/material"
import ApiService from "../services/ApiService"

interface TradingViewProps {
  realTimeData: any
  onShowNotification: (message: string, severity: "success" | "error" | "warning" | "info") => void
}

const TradingView: React.FC<TradingViewProps> = ({ realTimeData, onShowNotification }) => {
  const [signals, setSignals] = useState<any[]>([])
  const [marketData, setMarketData] = useState<any>({})
  const [manualTradeOpen, setManualTradeOpen] = useState(false)
  const [manualTrade, setManualTrade] = useState({
    symbol: "",
    side: "buy",
    quantity: "",
    price: "",
  })

  useEffect(() => {
    fetchTradingData()
    const interval = setInterval(fetchTradingData, 10000) // Update every 10 seconds
    return () => clearInterval(interval)
  }, [])

  const fetchTradingData = async () => {
    try {
      const [signalsData] = await Promise.all([ApiService.getSignals(20)])
      setSignals(signalsData)
    } catch (error) {
      console.error("Error fetching trading data:", error)
    }
  }

  const handleManualTrade = async () => {
    try {
      await ApiService.executeManualTrade({
        symbol: manualTrade.symbol,
        side: manualTrade.side,
        quantity: Number.parseFloat(manualTrade.quantity),
        price: Number.parseFloat(manualTrade.price),
      })
      onShowNotification("Manual trade executed successfully", "success")
      setManualTradeOpen(false)
      setManualTrade({ symbol: "", side: "buy", quantity: "", price: "" })
    } catch (error) {
      onShowNotification("Failed to execute manual trade", "error")
    }
  }

  const getSignalColor = (signal: string) => {
    switch (signal.toLowerCase()) {
      case "buy":
        return "success"
      case "sell":
        return "error"
      case "hold":
        return "warning"
      default:
        return "default"
    }
  }

  const formatConfidence = (confidence: number) => {
    return `${(confidence * 100).toFixed(1)}%`
  }

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 3 }}>
        <Typography variant="h4">Trading Signals</Typography>
        <Button variant="contained" color="primary" onClick={() => setManualTradeOpen(true)}>
          Manual Trade
        </Button>
      </Box>

      {/* Market Overview */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Market Overview
              </Typography>
              {realTimeData?.market_overview ? (
                <Grid container spacing={2}>
                  <Grid item xs={12} sm={6} md={3}>
                    <Typography variant="body2" color="text.secondary">
                      Market Trend
                    </Typography>
                    <Typography variant="h6">{realTimeData.market_overview.trend || "Neutral"}</Typography>
                  </Grid>
                  <Grid item xs={12} sm={6} md={3}>
                    <Typography variant="body2" color="text.secondary">
                      Volatility
                    </Typography>
                    <Typography variant="h6">{realTimeData.market_overview.volatility || "Medium"}</Typography>
                  </Grid>
                  <Grid item xs={12} sm={6} md={3}>
                    <Typography variant="body2" color="text.secondary">
                      Volume
                    </Typography>
                    <Typography variant="h6">{realTimeData.market_overview.volume || "Normal"}</Typography>
                  </Grid>
                  <Grid item xs={12} sm={6} md={3}>
                    <Typography variant="body2" color="text.secondary">
                      Sentiment
                    </Typography>
                    <Typography variant="h6">{realTimeData.market_overview.sentiment || "Neutral"}</Typography>
                  </Grid>
                </Grid>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  Loading market data...
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Trading Signals */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Recent Trading Signals
          </Typography>
          <TableContainer component={Paper} sx={{ backgroundColor: "background.paper" }}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Symbol</TableCell>
                  <TableCell>Signal</TableCell>
                  <TableCell>Confidence</TableCell>
                  <TableCell>Price</TableCell>
                  <TableCell>Target</TableCell>
                  <TableCell>Stop Loss</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Time</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {signals.map((signal, index) => (
                  <TableRow key={index}>
                    <TableCell>{signal.symbol}</TableCell>
                    <TableCell>
                      <Chip label={signal.signal} color={getSignalColor(signal.signal)} size="small" />
                    </TableCell>
                    <TableCell>{formatConfidence(signal.confidence)}</TableCell>
                    <TableCell>${signal.price?.toFixed(2)}</TableCell>
                    <TableCell>${signal.target?.toFixed(2)}</TableCell>
                    <TableCell>${signal.stop_loss?.toFixed(2)}</TableCell>
                    <TableCell>
                      <Chip
                        label={signal.status}
                        color={signal.status === "active" ? "success" : "default"}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>{new Date(signal.timestamp).toLocaleTimeString()}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>

      {/* Manual Trade Dialog */}
      <Dialog open={manualTradeOpen} onClose={() => setManualTradeOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Execute Manual Trade</DialogTitle>
        <DialogContent>
          <Box sx={{ display: "flex", flexDirection: "column", gap: 2, mt: 1 }}>
            <TextField
              label="Symbol"
              value={manualTrade.symbol}
              onChange={(e) => setManualTrade({ ...manualTrade, symbol: e.target.value })}
              fullWidth
            />
            <FormControl fullWidth>
              <InputLabel>Side</InputLabel>
              <Select
                value={manualTrade.side}
                onChange={(e) => setManualTrade({ ...manualTrade, side: e.target.value })}
              >
                <MenuItem value="buy">Buy</MenuItem>
                <MenuItem value="sell">Sell</MenuItem>
              </Select>
            </FormControl>
            <TextField
              label="Quantity"
              type="number"
              value={manualTrade.quantity}
              onChange={(e) => setManualTrade({ ...manualTrade, quantity: e.target.value })}
              fullWidth
            />
            <TextField
              label="Price"
              type="number"
              value={manualTrade.price}
              onChange={(e) => setManualTrade({ ...manualTrade, price: e.target.value })}
              fullWidth
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setManualTradeOpen(false)}>Cancel</Button>
          <Button onClick={handleManualTrade} variant="contained">
            Execute Trade
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  )
}

export default TradingView
