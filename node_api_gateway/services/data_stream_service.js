const WebSocket = require("ws")
const EventEmitter = require("events")
const axios = require("axios")

class DataStreamService extends EventEmitter {
  constructor(io) {
    super()
    this.io = io
    this.connections = new Map()
    this.subscriptions = new Set()
    this.marketData = new Map()
    this.isConnected = false

    // Data providers
    this.providers = {
      binance: {
        ws: "wss://stream.binance.com:9443/ws/",
        rest: "https://api.binance.com/api/v3",
      },
      alphaVantage: {
        rest: "https://www.alphavantage.co/query",
        apiKey: process.env.ALPHA_VANTAGE_API_KEY,
      },
      finnhub: {
        ws: "wss://ws.finnhub.io",
        rest: "https://finnhub.io/api/v1",
        apiKey: process.env.FINNHUB_API_KEY,
      },
    }

    this.setupEventHandlers()
  }

  setupEventHandlers() {
    this.on("marketData", this.handleMarketData.bind(this))
    this.on("newsUpdate", this.handleNewsUpdate.bind(this))
    this.on("connectionError", this.handleConnectionError.bind(this))
  }

  async connect() {
    try {
      console.log("📡 Connecting to data streams...")

      // Connect to multiple data sources
      await Promise.all([this.connectBinance(), this.connectFinnhub(), this.startNewsPolling()])

      this.isConnected = true
      console.log("✅ Data streams connected successfully")
    } catch (error) {
      console.error("❌ Failed to connect data streams:", error)
      throw error
    }
  }

  async connectBinance() {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(this.providers.binance.ws + "btcusdt@ticker")

      ws.on("open", () => {
        console.log("🔗 Connected to Binance WebSocket")
        this.connections.set("binance", ws)

        // Subscribe to multiple streams
        const subscribeMsg = {
          method: "SUBSCRIBE",
          params: ["btcusdt@ticker", "ethusdt@ticker", "adausdt@ticker", "dotusdt@ticker", "linkusdt@ticker"],
          id: 1,
        }

        ws.send(JSON.stringify(subscribeMsg))
        resolve()
      })

      ws.on("message", (data) => {
        try {
          const message = JSON.parse(data)
          if (message.e === "24hrTicker") {
            this.processBinanceTickerData(message)
          }
        } catch (error) {
          console.error("Error parsing Binance message:", error)
        }
      })

      ws.on("error", (error) => {
        console.error("Binance WebSocket error:", error)
        this.emit("connectionError", { provider: "binance", error })
        reject(error)
      })

      ws.on("close", () => {
        console.log("🔌 Binance WebSocket disconnected")
        this.connections.delete("binance")
        // Attempt reconnection
        setTimeout(() => this.connectBinance(), 5000)
      })
    })
  }

  async connectFinnhub() {
    if (!this.providers.finnhub.apiKey) {
      console.log("⚠️ Finnhub API key not provided, skipping connection")
      return
    }

    return new Promise((resolve, reject) => {
      const ws = new WebSocket(`${this.providers.finnhub.ws}?token=${this.providers.finnhub.apiKey}`)

      ws.on("open", () => {
        console.log("🔗 Connected to Finnhub WebSocket")
        this.connections.set("finnhub", ws)

        // Subscribe to stock symbols
        const symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
        symbols.forEach((symbol) => {
          ws.send(JSON.stringify({ type: "subscribe", symbol }))
        })

        resolve()
      })

      ws.on("message", (data) => {
        try {
          const message = JSON.parse(data)
          if (message.type === "trade") {
            this.processFinnhubTradeData(message)
          }
        } catch (error) {
          console.error("Error parsing Finnhub message:", error)
        }
      })

      ws.on("error", (error) => {
        console.error("Finnhub WebSocket error:", error)
        this.emit("connectionError", { provider: "finnhub", error })
        reject(error)
      })

      ws.on("close", () => {
        console.log("🔌 Finnhub WebSocket disconnected")
        this.connections.delete("finnhub")
        setTimeout(() => this.connectFinnhub(), 5000)
      })
    })
  }

  processBinanceTickerData(data) {
    const marketData = {
      symbol: data.s,
      price: Number.parseFloat(data.c),
      change: Number.parseFloat(data.P),
      volume: Number.parseFloat(data.v),
      high: Number.parseFloat(data.h),
      low: Number.parseFloat(data.l),
      timestamp: new Date(data.E),
      provider: "binance",
    }

    this.marketData.set(data.s, marketData)
    this.emit("marketData", marketData)

    // Emit to connected clients
    this.io.emit("marketUpdate", marketData)
  }

  processFinnhubTradeData(data) {
    if (data.data && data.data.length > 0) {
      data.data.forEach((trade) => {
        const marketData = {
          symbol: trade.s,
          price: trade.p,
          volume: trade.v,
          timestamp: new Date(trade.t),
          provider: "finnhub",
        }

        this.marketData.set(trade.s, marketData)
        this.emit("marketData", marketData)
        this.io.emit("marketUpdate", marketData)
      })
    }
  }

  async startNewsPolling() {
    // Poll for news every 5 minutes
    setInterval(
      async () => {
        try {
          await this.fetchLatestNews()
        } catch (error) {
          console.error("Error fetching news:", error)
        }
      },
      5 * 60 * 1000,
    )

    // Initial fetch
    await this.fetchLatestNews()
  }

  async fetchLatestNews() {
    try {
      // Fetch from multiple news sources
      const newsPromises = [this.fetchAlphaVantageNews(), this.fetchFinnhubNews(), this.fetchCryptoNews()]

      const newsResults = await Promise.allSettled(newsPromises)

      newsResults.forEach((result, index) => {
        if (result.status === "fulfilled" && result.value) {
          this.emit("newsUpdate", result.value)
        }
      })
    } catch (error) {
      console.error("Error in news polling:", error)
    }
  }

  async fetchAlphaVantageNews() {
    if (!this.providers.alphaVantage.apiKey) return null

    try {
      const response = await axios.get(this.providers.alphaVantage.rest, {
        params: {
          function: "NEWS_SENTIMENT",
          tickers: "AAPL,GOOGL,MSFT,TSLA",
          apikey: this.providers.alphaVantage.apiKey,
          limit: 10,
        },
      })

      return {
        provider: "alphavantage",
        articles: response.data.feed || [],
        timestamp: new Date(),
      }
    } catch (error) {
      console.error("Alpha Vantage news fetch error:", error)
      return null
    }
  }

  async fetchFinnhubNews() {
    if (!this.providers.finnhub.apiKey) return null

    try {
      const response = await axios.get(`${this.providers.finnhub.rest}/news`, {
        params: {
          category: "general",
          token: this.providers.finnhub.apiKey,
        },
      })

      return {
        provider: "finnhub",
        articles: response.data || [],
        timestamp: new Date(),
      }
    } catch (error) {
      console.error("Finnhub news fetch error:", error)
      return null
    }
  }

  async fetchCryptoNews() {
    try {
      // Using a free crypto news API
      const response = await axios.get("https://api.coingecko.com/api/v3/news")

      return {
        provider: "coingecko",
        articles: response.data.data || [],
        timestamp: new Date(),
      }
    } catch (error) {
      console.error("Crypto news fetch error:", error)
      return null
    }
  }

  async getHistoricalData(symbol, interval = "1h", limit = 100) {
    try {
      // Fetch from Binance for crypto
      if (symbol.includes("USDT")) {
        const response = await axios.get(`${this.providers.binance.rest}/klines`, {
          params: {
            symbol: symbol,
            interval: interval,
            limit: limit,
          },
        })

        return response.data.map((candle) => ({
          timestamp: new Date(candle[0]),
          open: Number.parseFloat(candle[1]),
          high: Number.parseFloat(candle[2]),
          low: Number.parseFloat(candle[3]),
          close: Number.parseFloat(candle[4]),
          volume: Number.parseFloat(candle[5]),
        }))
      }

      // Fetch from Alpha Vantage for stocks
      if (this.providers.alphaVantage.apiKey) {
        const response = await axios.get(this.providers.alphaVantage.rest, {
          params: {
            function: "TIME_SERIES_INTRADAY",
            symbol: symbol,
            interval: "60min",
            apikey: this.providers.alphaVantage.apiKey,
            outputsize: "compact",
          },
        })

        const timeSeries = response.data["Time Series (60min)"]
        if (timeSeries) {
          return Object.entries(timeSeries)
            .map(([time, data]) => ({
              timestamp: new Date(time),
              open: Number.parseFloat(data["1. open"]),
              high: Number.parseFloat(data["2. high"]),
              low: Number.parseFloat(data["3. low"]),
              close: Number.parseFloat(data["4. close"]),
              volume: Number.parseFloat(data["5. volume"]),
            }))
            .reverse()
        }
      }

      return []
    } catch (error) {
      console.error(`Error fetching historical data for ${symbol}:`, error)
      return []
    }
  }

  async getCurrentPrice(symbol) {
    const cached = this.marketData.get(symbol)
    if (cached && Date.now() - cached.timestamp.getTime() < 60000) {
      return cached.price
    }

    try {
      // Try Binance first for crypto
      if (symbol.includes("USDT")) {
        const response = await axios.get(`${this.providers.binance.rest}/ticker/price`, {
          params: { symbol },
        })
        return Number.parseFloat(response.data.price)
      }

      // Try Alpha Vantage for stocks
      if (this.providers.alphaVantage.apiKey) {
        const response = await axios.get(this.providers.alphaVantage.rest, {
          params: {
            function: "GLOBAL_QUOTE",
            symbol: symbol,
            apikey: this.providers.alphaVantage.apiKey,
          },
        })

        const quote = response.data["Global Quote"]
        if (quote) {
          return Number.parseFloat(quote["05. price"])
        }
      }

      return null
    } catch (error) {
      console.error(`Error fetching current price for ${symbol}:`, error)
      return null
    }
  }

  handleMarketData(data) {
    // Process and store market data
    console.log(`📊 Market Update: ${data.symbol} - $${data.price}`)

    // Emit to AI service for signal generation
    this.emit("aiAnalysis", {
      symbol: data.symbol,
      price: data.price,
      timestamp: data.timestamp,
    })
  }

  handleNewsUpdate(newsData) {
    console.log(`📰 News Update: ${newsData.articles.length} articles from ${newsData.provider}`)

    // Emit to clients
    this.io.emit("newsUpdate", newsData)

    // Process for sentiment analysis
    this.emit("sentimentAnalysis", newsData)
  }

  handleConnectionError(error) {
    console.error(`🔌 Connection error from ${error.provider}:`, error.error)

    // Implement reconnection logic
    setTimeout(() => {
      if (error.provider === "binance") {
        this.connectBinance()
      } else if (error.provider === "finnhub") {
        this.connectFinnhub()
      }
    }, 5000)
  }

  subscribe(symbol) {
    this.subscriptions.add(symbol)
    console.log(`📡 Subscribed to ${symbol}`)
  }

  unsubscribe(symbol) {
    this.subscriptions.delete(symbol)
    console.log(`📡 Unsubscribed from ${symbol}`)
  }

  getMarketData(symbol) {
    return this.marketData.get(symbol)
  }

  getAllMarketData() {
    return Array.from(this.marketData.values())
  }

  disconnect() {
    console.log("🔌 Disconnecting data streams...")

    this.connections.forEach((ws, provider) => {
      ws.close()
      console.log(`Disconnected from ${provider}`)
    })

    this.connections.clear()
    this.isConnected = false
  }

  isConnectedToProvider(provider) {
    return this.connections.has(provider)
  }

  getConnectionStatus() {
    return {
      isConnected: this.isConnected,
      providers: Array.from(this.connections.keys()),
      subscriptions: Array.from(this.subscriptions),
      lastUpdate: new Date(),
    }
  }
}

module.exports = DataStreamService
