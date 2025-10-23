"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { ArrowUpCircle, ArrowDownCircle, Clock, CheckCircle, XCircle } from "lucide-react"

interface Order {
  id: string
  symbol: string
  type: "BUY" | "SELL"
  price: number
  quantity: number
  status: "pending" | "filled" | "cancelled"
  timestamp: string
}

export default function TradingPage() {
  const [selectedSymbol, setSelectedSymbol] = useState("BTC/USDT")
  const [quantity, setQuantity] = useState("")
  const [price, setPrice] = useState("")
  const [orderType, setOrderType] = useState<"market" | "limit">("market")

  const [orders, setOrders] = useState<Order[]>([
    { id: "1", symbol: "BTC/USDT", type: "BUY", price: 67234.50, quantity: 0.5, status: "filled", timestamp: "10 mins ago" },
    { id: "2", symbol: "ETH/USDT", type: "SELL", price: 3456.78, quantity: 2.0, status: "filled", timestamp: "25 mins ago" },
    { id: "3", symbol: "AAPL", type: "BUY", price: 178.23, quantity: 10, status: "pending", timestamp: "2 mins ago" },
  ])

  const symbols = ["BTC/USDT", "ETH/USDT", "AAPL", "TSLA", "GOOGL"]

  const handleBuy = () => {
    const newOrder: Order = {
      id: String(Date.now()),
      symbol: selectedSymbol,
      type: "BUY",
      price: price ? parseFloat(price) : 0,
      quantity: parseFloat(quantity) || 0,
      status: "pending",
      timestamp: "Just now",
    }
    setOrders([newOrder, ...orders])
    setQuantity("")
    setPrice("")
  }

  const handleSell = () => {
    const newOrder: Order = {
      id: String(Date.now()),
      symbol: selectedSymbol,
      type: "SELL",
      price: price ? parseFloat(price) : 0,
      quantity: parseFloat(quantity) || 0,
      status: "pending",
      timestamp: "Just now",
    }
    setOrders([newOrder, ...orders])
    setQuantity("")
    setPrice("")
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      <div className="container mx-auto">
        <h1 className="text-3xl font-bold mb-6">Trading Console</h1>

        <div className="grid md:grid-cols-3 gap-6">
          <div className="md:col-span-2">
            <Card className="bg-gray-800 border-gray-700 mb-6">
              <CardHeader>
                <CardTitle className="text-white">Place Order</CardTitle>
                <CardDescription className="text-gray-400">Execute manual trades</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-400 mb-2">Symbol</label>
                    <select
                      value={selectedSymbol}
                      onChange={(e) => setSelectedSymbol(e.target.value)}
                      className="w-full bg-gray-700 border border-gray-600 rounded-md px-3 py-2 text-white"
                    >
                      {symbols.map((symbol) => (
                        <option key={symbol} value={symbol}>
                          {symbol}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-400 mb-2">Order Type</label>
                    <div className="flex space-x-4">
                      <button
                        onClick={() => setOrderType("market")}
                        className={`flex-1 py-2 rounded-md ${
                          orderType === "market"
                            ? "bg-blue-600 text-white"
                            : "bg-gray-700 text-gray-400"
                        }`}
                      >
                        Market
                      </button>
                      <button
                        onClick={() => setOrderType("limit")}
                        className={`flex-1 py-2 rounded-md ${
                          orderType === "limit"
                            ? "bg-blue-600 text-white"
                            : "bg-gray-700 text-gray-400"
                        }`}
                      >
                        Limit
                      </button>
                    </div>
                  </div>

                  {orderType === "limit" && (
                    <div>
                      <label className="block text-sm font-medium text-gray-400 mb-2">Price</label>
                      <input
                        type="number"
                        value={price}
                        onChange={(e) => setPrice(e.target.value)}
                        placeholder="Enter price"
                        className="w-full bg-gray-700 border border-gray-600 rounded-md px-3 py-2 text-white"
                      />
                    </div>
                  )}

                  <div>
                    <label className="block text-sm font-medium text-gray-400 mb-2">Quantity</label>
                    <input
                      type="number"
                      value={quantity}
                      onChange={(e) => setQuantity(e.target.value)}
                      placeholder="Enter quantity"
                      className="w-full bg-gray-700 border border-gray-600 rounded-md px-3 py-2 text-white"
                    />
                  </div>

                  <div className="flex space-x-4">
                    <Button
                      onClick={handleBuy}
                      className="flex-1 bg-green-600 hover:bg-green-700 text-white"
                    >
                      <ArrowUpCircle className="w-4 h-4 mr-2" />
                      Buy
                    </Button>
                    <Button
                      onClick={handleSell}
                      className="flex-1 bg-red-600 hover:bg-red-700 text-white"
                    >
                      <ArrowDownCircle className="w-4 h-4 mr-2" />
                      Sell
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-gray-800 border-gray-700">
              <CardHeader>
                <CardTitle className="text-white">Order History</CardTitle>
                <CardDescription className="text-gray-400">Recent and pending orders</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {orders.map((order) => (
                    <div key={order.id} className="flex items-center justify-between p-3 bg-gray-700 rounded-lg">
                      <div className="flex items-center space-x-4">
                        <div className={`p-2 rounded-full ${
                          order.type === "BUY" ? "bg-green-500/20" : "bg-red-500/20"
                        }`}>
                          {order.type === "BUY" ? (
                            <ArrowUpCircle className="w-5 h-5 text-green-500" />
                          ) : (
                            <ArrowDownCircle className="w-5 h-5 text-red-500" />
                          )}
                        </div>
                        <div>
                          <div className="font-semibold text-white">{order.symbol}</div>
                          <div className="text-sm text-gray-400">
                            {order.quantity} @ ${order.price.toFixed(2)}
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="flex items-center space-x-2">
                          {order.status === "filled" && (
                            <>
                              <CheckCircle className="w-4 h-4 text-green-500" />
                              <span className="text-green-500 text-sm">Filled</span>
                            </>
                          )}
                          {order.status === "pending" && (
                            <>
                              <Clock className="w-4 h-4 text-yellow-500" />
                              <span className="text-yellow-500 text-sm">Pending</span>
                            </>
                          )}
                          {order.status === "cancelled" && (
                            <>
                              <XCircle className="w-4 h-4 text-red-500" />
                              <span className="text-red-500 text-sm">Cancelled</span>
                            </>
                          )}
                        </div>
                        <div className="text-xs text-gray-400 mt-1">{order.timestamp}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          <div>
            <Card className="bg-gray-800 border-gray-700">
              <CardHeader>
                <CardTitle className="text-white">AI Recommendations</CardTitle>
                <CardDescription className="text-gray-400">ML-powered insights</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="p-4 bg-green-500/10 border border-green-500/30 rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-semibold text-green-500">BUY Signal</span>
                      <span className="text-sm text-gray-400">92% confidence</span>
                    </div>
                    <div className="text-white font-bold">BTC/USDT</div>
                    <div className="text-sm text-gray-400 mt-1">
                      Strong bullish momentum detected
                    </div>
                  </div>

                  <div className="p-4 bg-blue-500/10 border border-blue-500/30 rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-semibold text-blue-500">HOLD Signal</span>
                      <span className="text-sm text-gray-400">78% confidence</span>
                    </div>
                    <div className="text-white font-bold">ETH/USDT</div>
                    <div className="text-sm text-gray-400 mt-1">
                      Consolidation pattern forming
                    </div>
                  </div>

                  <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-semibold text-red-500">SELL Signal</span>
                      <span className="text-sm text-gray-400">85% confidence</span>
                    </div>
                    <div className="text-white font-bold">TSLA</div>
                    <div className="text-sm text-gray-400 mt-1">
                      Bearish divergence identified
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}
