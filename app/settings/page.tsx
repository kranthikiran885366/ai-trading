"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Save, RefreshCw } from "lucide-react"

export default function SettingsPage() {
  const [botActive, setBotActive] = useState(true)
  const [maxDailyLoss, setMaxDailyLoss] = useState("2000")
  const [maxDrawdown, setMaxDrawdown] = useState("10")
  const [maxPositionSize, setMaxPositionSize] = useState("5")
  const [confidenceThreshold, setConfidenceThreshold] = useState("85")
  const [riskLevel, setRiskLevel] = useState<"low" | "medium" | "high">("medium")

  const handleSave = () => {
    alert("Settings saved successfully!")
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      <div className="container mx-auto max-w-4xl">
        <h1 className="text-3xl font-bold mb-6">Settings</h1>

        <div className="space-y-6">
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader>
              <CardTitle className="text-white">Bot Control</CardTitle>
              <CardDescription className="text-gray-400">Enable or disable the trading bot</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-between">
                <div>
                  <div className="font-semibold text-white">Trading Bot Status</div>
                  <div className="text-sm text-gray-400">
                    {botActive ? "Bot is currently active and trading" : "Bot is currently stopped"}
                  </div>
                </div>
                <button
                  onClick={() => setBotActive(!botActive)}
                  className={`relative inline-flex h-8 w-14 items-center rounded-full transition-colors ${
                    botActive ? "bg-green-600" : "bg-gray-600"
                  }`}
                >
                  <span
                    className={`inline-block h-6 w-6 transform rounded-full bg-white transition-transform ${
                      botActive ? "translate-x-7" : "translate-x-1"
                    }`}
                  />
                </button>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gray-800 border-gray-700">
            <CardHeader>
              <CardTitle className="text-white">Risk Management</CardTitle>
              <CardDescription className="text-gray-400">Configure risk parameters</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-400 mb-2">
                  Max Daily Loss ($)
                </label>
                <input
                  type="number"
                  value={maxDailyLoss}
                  onChange={(e) => setMaxDailyLoss(e.target.value)}
                  className="w-full bg-gray-700 border border-gray-600 rounded-md px-3 py-2 text-white"
                />
                <p className="text-xs text-gray-500 mt-1">Maximum loss allowed per day</p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-400 mb-2">
                  Max Drawdown (%)
                </label>
                <input
                  type="number"
                  value={maxDrawdown}
                  onChange={(e) => setMaxDrawdown(e.target.value)}
                  className="w-full bg-gray-700 border border-gray-600 rounded-md px-3 py-2 text-white"
                />
                <p className="text-xs text-gray-500 mt-1">Maximum portfolio drawdown percentage</p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-400 mb-2">
                  Max Position Size (%)
                </label>
                <input
                  type="number"
                  value={maxPositionSize}
                  onChange={(e) => setMaxPositionSize(e.target.value)}
                  className="w-full bg-gray-700 border border-gray-600 rounded-md px-3 py-2 text-white"
                />
                <p className="text-xs text-gray-500 mt-1">Maximum percentage of portfolio per position</p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-400 mb-2">Risk Level</label>
                <div className="flex space-x-4">
                  {(["low", "medium", "high"] as const).map((level) => (
                    <button
                      key={level}
                      onClick={() => setRiskLevel(level)}
                      className={`flex-1 py-2 rounded-md capitalize ${
                        riskLevel === level
                          ? "bg-blue-600 text-white"
                          : "bg-gray-700 text-gray-400"
                      }`}
                    >
                      {level}
                    </button>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gray-800 border-gray-700">
            <CardHeader>
              <CardTitle className="text-white">AI Configuration</CardTitle>
              <CardDescription className="text-gray-400">Adjust AI model parameters</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-400 mb-2">
                  AI Confidence Threshold (%)
                </label>
                <input
                  type="number"
                  value={confidenceThreshold}
                  onChange={(e) => setConfidenceThreshold(e.target.value)}
                  className="w-full bg-gray-700 border border-gray-600 rounded-md px-3 py-2 text-white"
                />
                <p className="text-xs text-gray-500 mt-1">
                  Minimum confidence required for AI to generate signals
                </p>
              </div>

              <div className="flex items-center justify-between p-4 bg-gray-700 rounded-lg">
                <div>
                  <div className="font-semibold text-white">Model Retraining</div>
                  <div className="text-sm text-gray-400">Last trained: 2 hours ago</div>
                </div>
                <Button className="bg-blue-600 hover:bg-blue-700 text-white">
                  <RefreshCw className="w-4 h-4 mr-2" />
                  Retrain Now
                </Button>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gray-800 border-gray-700">
            <CardHeader>
              <CardTitle className="text-white">API Connections</CardTitle>
              <CardDescription className="text-gray-400">Manage exchange connections</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between p-4 bg-gray-700 rounded-lg">
                <div>
                  <div className="font-semibold text-white">Binance</div>
                  <div className="text-sm text-green-500">Connected</div>
                </div>
                <Button variant="outline" className="bg-gray-600 hover:bg-gray-500 text-white border-gray-500">
                  Configure
                </Button>
              </div>

              <div className="flex items-center justify-between p-4 bg-gray-700 rounded-lg">
                <div>
                  <div className="font-semibold text-white">Alpaca</div>
                  <div className="text-sm text-green-500">Connected</div>
                </div>
                <Button variant="outline" className="bg-gray-600 hover:bg-gray-500 text-white border-gray-500">
                  Configure
                </Button>
              </div>

              <div className="flex items-center justify-between p-4 bg-gray-700 rounded-lg">
                <div>
                  <div className="font-semibold text-white">MongoDB</div>
                  <div className="text-sm text-gray-400">Not configured</div>
                </div>
                <Button variant="outline" className="bg-gray-600 hover:bg-gray-500 text-white border-gray-500">
                  Setup
                </Button>
              </div>
            </CardContent>
          </Card>

          <div className="flex justify-end">
            <Button onClick={handleSave} className="bg-blue-600 hover:bg-blue-700 text-white">
              <Save className="w-4 h-4 mr-2" />
              Save All Settings
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}
