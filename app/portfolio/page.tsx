"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { TrendingUp, TrendingDown, Wallet, PieChart } from "lucide-react"

interface Position {
  symbol: string
  quantity: number
  avgPrice: number
  currentPrice: number
  pnl: number
  pnlPercent: number
}

export default function PortfolioPage() {
  const positions: Position[] = [
    { symbol: "BTC/USDT", quantity: 0.5, avgPrice: 65000, currentPrice: 67234.50, pnl: 1117.25, pnlPercent: 3.44 },
    { symbol: "ETH/USDT", quantity: 2.0, avgPrice: 3500, currentPrice: 3456.78, pnl: -86.44, pnlPercent: -1.23 },
    { symbol: "AAPL", quantity: 10, avgPrice: 175.50, currentPrice: 178.23, pnl: 27.30, pnlPercent: 1.56 },
    { symbol: "TSLA", quantity: 5, avgPrice: 250.00, currentPrice: 245.67, pnl: -21.65, pnlPercent: -1.73 },
  ]

  const totalValue = positions.reduce((sum, pos) => sum + (pos.currentPrice * pos.quantity), 0)
  const totalPnL = positions.reduce((sum, pos) => sum + pos.pnl, 0)
  const totalPnLPercent = (totalPnL / (totalValue - totalPnL)) * 100

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      <div className="container mx-auto">
        <h1 className="text-3xl font-bold mb-6">Portfolio</h1>

        <div className="grid md:grid-cols-4 gap-6 mb-6">
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium text-gray-400">Total Value</CardTitle>
              <Wallet className="w-4 h-4 text-blue-500" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-white">
                ${totalValue.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </div>
              <p className="text-xs text-gray-400 mt-1">Portfolio value</p>
            </CardContent>
          </Card>

          <Card className="bg-gray-800 border-gray-700">
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium text-gray-400">Total P&L</CardTitle>
              {totalPnL >= 0 ? (
                <TrendingUp className="w-4 h-4 text-green-500" />
              ) : (
                <TrendingDown className="w-4 h-4 text-red-500" />
              )}
            </CardHeader>
            <CardContent>
              <div className={`text-2xl font-bold ${totalPnL >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                {totalPnL >= 0 ? '+' : ''}${totalPnL.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </div>
              <p className="text-xs text-gray-400 mt-1">
                {totalPnLPercent >= 0 ? '+' : ''}{totalPnLPercent.toFixed(2)}%
              </p>
            </CardContent>
          </Card>

          <Card className="bg-gray-800 border-gray-700">
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium text-gray-400">Positions</CardTitle>
              <PieChart className="w-4 h-4 text-purple-500" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-white">{positions.length}</div>
              <p className="text-xs text-gray-400 mt-1">Active positions</p>
            </CardContent>
          </Card>

          <Card className="bg-gray-800 border-gray-700">
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium text-gray-400">Win Rate</CardTitle>
              <TrendingUp className="w-4 h-4 text-green-500" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-white">68.5%</div>
              <p className="text-xs text-gray-400 mt-1">156 total trades</p>
            </CardContent>
          </Card>
        </div>

        <Card className="bg-gray-800 border-gray-700">
          <CardHeader>
            <CardTitle className="text-white">Current Positions</CardTitle>
            <CardDescription className="text-gray-400">Real-time portfolio tracking</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {positions.map((position) => (
                <div key={position.symbol} className="p-4 bg-gray-700 rounded-lg">
                  <div className="flex items-center justify-between mb-3">
                    <div>
                      <div className="font-bold text-white text-lg">{position.symbol}</div>
                      <div className="text-sm text-gray-400">
                        {position.quantity} units @ ${position.avgPrice.toFixed(2)}
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-white font-semibold">
                        ${position.currentPrice.toFixed(2)}
                      </div>
                      <div className="text-sm text-gray-400">Current price</div>
                    </div>
                  </div>
                  <div className="flex items-center justify-between pt-3 border-t border-gray-600">
                    <div className="text-sm text-gray-400">
                      Value: ${(position.quantity * position.currentPrice).toFixed(2)}
                    </div>
                    <div className={`flex items-center font-semibold ${
                      position.pnl >= 0 ? 'text-green-500' : 'text-red-500'
                    }`}>
                      {position.pnl >= 0 ? (
                        <TrendingUp className="w-4 h-4 mr-1" />
                      ) : (
                        <TrendingDown className="w-4 h-4 mr-1" />
                      )}
                      <span>
                        {position.pnl >= 0 ? '+' : ''}${position.pnl.toFixed(2)} ({position.pnlPercent >= 0 ? '+' : ''}{position.pnlPercent.toFixed(2)}%)
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <div className="grid md:grid-cols-2 gap-6 mt-6">
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader>
              <CardTitle className="text-white">Performance Metrics</CardTitle>
              <CardDescription className="text-gray-400">Key portfolio statistics</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex justify-between">
                  <span className="text-gray-400">Sharpe Ratio</span>
                  <span className="text-white font-semibold">2.34</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Max Drawdown</span>
                  <span className="text-red-500 font-semibold">-8.5%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Total Trades</span>
                  <span className="text-white font-semibold">156</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Average Trade</span>
                  <span className="text-green-500 font-semibold">+$79.81</span>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gray-800 border-gray-700">
            <CardHeader>
              <CardTitle className="text-white">Risk Metrics</CardTitle>
              <CardDescription className="text-gray-400">Portfolio risk analysis</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex justify-between">
                  <span className="text-gray-400">Portfolio Beta</span>
                  <span className="text-white font-semibold">1.12</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Value at Risk (95%)</span>
                  <span className="text-yellow-500 font-semibold">$1,234</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Correlation Score</span>
                  <span className="text-white font-semibold">0.67</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Exposure</span>
                  <span className="text-white font-semibold">78%</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
