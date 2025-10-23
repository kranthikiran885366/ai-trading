"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { TrendingUp, TrendingDown, Activity, DollarSign, BarChart3 } from "lucide-react"

interface MarketData {
  symbol: string
  price: number
  change: number
  changePercent: number
}

interface BotStatus {
  active: boolean
  totalTrades: number
  winRate: number
  totalProfit: number
}

export default function DashboardPage() {
  const [marketData, setMarketData] = useState<MarketData[]>([
    { symbol: "BTC/USDT", price: 67234.50, change: 1234.50, changePercent: 1.87 },
    { symbol: "ETH/USDT", price: 3456.78, change: -45.32, changePercent: -1.29 },
    { symbol: "AAPL", price: 178.23, change: 2.45, changePercent: 1.39 },
    { symbol: "TSLA", price: 245.67, change: -3.21, changePercent: -1.29 },
  ])

  const [botStatus, setBotStatus] = useState<BotStatus>({
    active: true,
    totalTrades: 156,
    winRate: 68.5,
    totalProfit: 12450.75,
  })

  const [recentSignals, setRecentSignals] = useState([
    { symbol: "BTC/USDT", action: "BUY", confidence: 0.89, time: "2 mins ago" },
    { symbol: "ETH/USDT", action: "SELL", confidence: 0.76, time: "5 mins ago" },
    { symbol: "AAPL", action: "BUY", confidence: 0.92, time: "8 mins ago" },
  ])

  useEffect(() => {
    const interval = setInterval(() => {
      setMarketData(prev => prev.map(item => ({
        ...item,
        price: item.price + (Math.random() - 0.5) * 10,
        change: item.change + (Math.random() - 0.5) * 5,
        changePercent: item.changePercent + (Math.random() - 0.5) * 0.5,
      })))
    }, 3000)

    return () => clearInterval(interval)
  }, [])

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      <div className="container mx-auto">
        <h1 className="text-3xl font-bold mb-6">Trading Dashboard</h1>

        <div className="grid md:grid-cols-4 gap-6 mb-6">
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium text-gray-400">Bot Status</CardTitle>
              <Activity className={`w-4 h-4 ${botStatus.active ? 'text-green-500' : 'text-red-500'}`} />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-white">
                {botStatus.active ? 'Active' : 'Inactive'}
              </div>
              <p className="text-xs text-gray-400 mt-1">
                {botStatus.totalTrades} trades executed
              </p>
            </CardContent>
          </Card>

          <Card className="bg-gray-800 border-gray-700">
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium text-gray-400">Win Rate</CardTitle>
              <BarChart3 className="w-4 h-4 text-blue-500" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-white">{botStatus.winRate}%</div>
              <p className="text-xs text-gray-400 mt-1">
                Above 95% target
              </p>
            </CardContent>
          </Card>

          <Card className="bg-gray-800 border-gray-700">
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium text-gray-400">Total Profit</CardTitle>
              <DollarSign className="w-4 h-4 text-green-500" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-500">
                ${botStatus.totalProfit.toLocaleString()}
              </div>
              <p className="text-xs text-gray-400 mt-1">
                +12.4% this month
              </p>
            </CardContent>
          </Card>

          <Card className="bg-gray-800 border-gray-700">
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium text-gray-400">Active Positions</CardTitle>
              <TrendingUp className="w-4 h-4 text-purple-500" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-white">8</div>
              <p className="text-xs text-gray-400 mt-1">
                $45,230 invested
              </p>
            </CardContent>
          </Card>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader>
              <CardTitle className="text-white">Live Market Data</CardTitle>
              <CardDescription className="text-gray-400">Real-time price updates</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {marketData.map((item) => (
                  <div key={item.symbol} className="flex items-center justify-between p-3 bg-gray-700 rounded-lg">
                    <div>
                      <div className="font-semibold text-white">{item.symbol}</div>
                      <div className="text-sm text-gray-400">${item.price.toFixed(2)}</div>
                    </div>
                    <div className={`flex items-center ${item.change >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                      {item.change >= 0 ? <TrendingUp className="w-4 h-4 mr-1" /> : <TrendingDown className="w-4 h-4 mr-1" />}
                      <span className="font-semibold">{item.changePercent.toFixed(2)}%</span>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gray-800 border-gray-700">
            <CardHeader>
              <CardTitle className="text-white">Recent AI Signals</CardTitle>
              <CardDescription className="text-gray-400">Latest trading opportunities</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {recentSignals.map((signal, idx) => (
                  <div key={idx} className="flex items-center justify-between p-3 bg-gray-700 rounded-lg">
                    <div>
                      <div className="font-semibold text-white">{signal.symbol}</div>
                      <div className="text-sm text-gray-400">{signal.time}</div>
                    </div>
                    <div className="text-right">
                      <div className={`font-bold ${signal.action === 'BUY' ? 'text-green-500' : 'text-red-500'}`}>
                        {signal.action}
                      </div>
                      <div className="text-sm text-gray-400">
                        {(signal.confidence * 100).toFixed(0)}% confidence
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
