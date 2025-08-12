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
} from "@mui/material"
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip } from "recharts"
import ApiService from "../services/ApiService"

interface PortfolioProps {
  realTimeData: any
  onShowNotification: (message: string, severity: "success" | "error" | "warning" | "info") => void
}

const Portfolio: React.FC<PortfolioProps> = ({ realTimeData, onShowNotification }) => {
  const [portfolio, setPortfolio] = useState<any>(null)
  const [positions, setPositions] = useState<any[]>([])

  useEffect(() => {
    fetchPortfolioData()
    const interval = setInterval(fetchPortfolioData, 15000) // Update every 15 seconds
    return () => clearInterval(interval)
  }, [])

  const fetchPortfolioData = async () => {
    try {
      const portfolioData = await ApiService.getPortfolio()
      setPortfolio(portfolioData)
      setPositions(portfolioData.positions || [])
    } catch (error) {
      console.error("Error fetching portfolio data:", error)
      onShowNotification("Failed to fetch portfolio data", "error")
    }
  }

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
    }).format(value)
  }

  const formatPercentage = (value: number) => {
    return `${value.toFixed(2)}%`
  }

  const COLORS = ["#00ff88", "#ff4444", "#ffaa00", "#00aaff", "#aa00ff"]

  const pieData = positions.map((position, index) => ({
    name: position.symbol,
    value: position.market_value,
    color: COLORS[index % COLORS.length],
  }))

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Typography variant="h4" gutterBottom>
        Portfolio Overview
      </Typography>

      {/* Portfolio Summary */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" color="primary">
                Total Value
              </Typography>
              <Typography variant="h4">{formatCurrency(portfolio?.total_value || 0)}</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" color="success.main">
                Cash Balance
              </Typography>
              <Typography variant="h4">{formatCurrency(portfolio?.cash_balance || 0)}</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" color="info.main">
                Total P&L
              </Typography>
              <Typography variant="h4" color={portfolio?.total_pnl >= 0 ? "success.main" : "error.main"}>
                {formatCurrency(portfolio?.total_pnl || 0)}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" color="warning.main">
                Day Change
              </Typography>
              <Typography variant="h4" color={portfolio?.day_change >= 0 ? "success.main" : "error.main"}>
                {formatPercentage(portfolio?.day_change_percent || 0)}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Charts */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Portfolio Allocation
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {pieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value) => formatCurrency(Number(value))} />
                </PieChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Position Performance
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={positions}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="symbol" />
                  <YAxis />
                  <Tooltip formatter={(value) => formatPercentage(Number(value))} />
                  <Bar dataKey="pnl_percent" fill="#00ff88" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Positions Table */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Current Positions
          </Typography>
          <TableContainer component={Paper} sx={{ backgroundColor: "background.paper" }}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Symbol</TableCell>
                  <TableCell>Quantity</TableCell>
                  <TableCell>Avg Price</TableCell>
                  <TableCell>Current Price</TableCell>
                  <TableCell>Market Value</TableCell>
                  <TableCell>P&L</TableCell>
                  <TableCell>P&L %</TableCell>
                  <TableCell>Status</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {positions.map((position, index) => (
                  <TableRow key={index}>
                    <TableCell>{position.symbol}</TableCell>
                    <TableCell>{position.quantity}</TableCell>
                    <TableCell>{formatCurrency(position.avg_price)}</TableCell>
                    <TableCell>{formatCurrency(position.current_price)}</TableCell>
                    <TableCell>{formatCurrency(position.market_value)}</TableCell>
                    <TableCell
                      sx={{
                        color: position.pnl >= 0 ? "success.main" : "error.main",
                      }}
                    >
                      {formatCurrency(position.pnl)}
                    </TableCell>
                    <TableCell
                      sx={{
                        color: position.pnl_percent >= 0 ? "success.main" : "error.main",
                      }}
                    >
                      {formatPercentage(position.pnl_percent)}
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={position.status}
                        color={position.status === "open" ? "success" : "default"}
                        size="small"
                      />
                    </TableCell>
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

export default Portfolio
