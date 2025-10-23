"use client"

import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 to-gray-800 text-white">
      <div className="container mx-auto px-4 py-16">
        <div className="text-center mb-16">
          <h1 className="text-5xl font-bold mb-4">AI Trading Bot Platform</h1>
          <p className="text-xl text-gray-300 mb-8">
            Advanced algorithmic trading with AI-powered decision making
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-6 mb-16">
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader>
              <CardTitle className="text-white">Multi-Broker Support</CardTitle>
              <CardDescription className="text-gray-400">
                Connect to Binance, Alpaca, and more
              </CardDescription>
            </CardHeader>
            <CardContent className="text-gray-300">
              Seamlessly trade across multiple exchanges with unified API
            </CardContent>
          </Card>

          <Card className="bg-gray-800 border-gray-700">
            <CardHeader>
              <CardTitle className="text-white">AI-Powered Analysis</CardTitle>
              <CardDescription className="text-gray-400">
                Machine learning for market predictions
              </CardDescription>
            </CardHeader>
            <CardContent className="text-gray-300">
              Advanced neural networks analyze market trends in real-time
            </CardContent>
          </Card>

          <Card className="bg-gray-800 border-gray-700">
            <CardHeader>
              <CardTitle className="text-white">Risk Management</CardTitle>
              <CardDescription className="text-gray-400">
                Automated position sizing and stop-loss
              </CardDescription>
            </CardHeader>
            <CardContent className="text-gray-300">
              Protect your portfolio with intelligent risk controls
            </CardContent>
          </Card>
        </div>

        <div className="text-center mb-12">
          <p className="text-gray-400 mb-4">
            Status: <span className="text-green-400">Platform Ready</span>
          </p>
          <p className="text-sm text-gray-500 mb-6">
            Configure your API keys in the environment settings to start trading
          </p>
          <div className="flex justify-center space-x-4">
            <Link href="/dashboard">
              <Button className="bg-blue-600 hover:bg-blue-700 text-white">
                Go to Dashboard
              </Button>
            </Link>
            <Link href="/settings">
              <Button variant="outline" className="border-gray-600 bg-gray-700 hover:bg-gray-600 text-white">
                Configure Settings
              </Button>
            </Link>
          </div>
        </div>

        <div className="grid md:grid-cols-4 gap-4 max-w-4xl mx-auto">
          <Link href="/dashboard" className="p-6 bg-gray-800 border border-gray-700 rounded-lg hover:bg-gray-700 transition-colors text-center">
            <div className="text-3xl mb-2">📊</div>
            <div className="font-semibold text-white">Dashboard</div>
            <div className="text-sm text-gray-400 mt-1">Real-time monitoring</div>
          </Link>
          <Link href="/trading" className="p-6 bg-gray-800 border border-gray-700 rounded-lg hover:bg-gray-700 transition-colors text-center">
            <div className="text-3xl mb-2">💹</div>
            <div className="font-semibold text-white">Trading</div>
            <div className="text-sm text-gray-400 mt-1">Execute trades</div>
          </Link>
          <Link href="/portfolio" className="p-6 bg-gray-800 border border-gray-700 rounded-lg hover:bg-gray-700 transition-colors text-center">
            <div className="text-3xl mb-2">💼</div>
            <div className="font-semibold text-white">Portfolio</div>
            <div className="text-sm text-gray-400 mt-1">Track positions</div>
          </Link>
          <Link href="/settings" className="p-6 bg-gray-800 border border-gray-700 rounded-lg hover:bg-gray-700 transition-colors text-center">
            <div className="text-3xl mb-2">⚙️</div>
            <div className="font-semibold text-white">Settings</div>
            <div className="text-sm text-gray-400 mt-1">Configure bot</div>
          </Link>
        </div>
      </div>
    </div>
  )
}
