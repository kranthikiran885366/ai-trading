import axios from "axios"

const API_BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:8000"

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    "Content-Type": "application/json",
  },
})

// Request interceptor
apiClient.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem("auth_token")
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => {
    return Promise.reject(error)
  },
)

// Response interceptor
apiClient.interceptors.response.use(
  (response) => {
    return response.data
  },
  (error) => {
    console.error("API Error:", error)
    return Promise.reject(error)
  },
)

class ApiService {
  // Bot Control
  static async getBotStatus() {
    return apiClient.get("/api/status")
  }

  static async startBot() {
    return apiClient.post("/api/start")
  }

  static async stopBot() {
    return apiClient.post("/api/stop")
  }

  // Portfolio
  static async getPortfolio() {
    return apiClient.get("/api/portfolio")
  }

  // Trading
  static async getTrades(limit = 100) {
    return apiClient.get(`/api/trades?limit=${limit}`)
  }

  static async getSignals(limit = 50) {
    return apiClient.get(`/api/signals?limit=${limit}`)
  }

  static async executeManualTrade(tradeData: any) {
    return apiClient.post("/api/manual-trade", tradeData)
  }

  // Performance
  static async getPerformance() {
    return apiClient.get("/api/performance")
  }

  // Market Data
  static async getMarketData(symbol: string) {
    return apiClient.get(`/api/market-data/${symbol}`)
  }

  // AI Insights
  static async getAiInsights() {
    return apiClient.get("/api/ai-insights")
  }

  // Settings
  static async getSettings() {
    return apiClient.get("/api/settings")
  }

  static async saveSettings(settings: any) {
    return apiClient.post("/api/settings", settings)
  }

  // Health Check
  static async healthCheck() {
    return apiClient.get("/health")
  }
}

export default ApiService
