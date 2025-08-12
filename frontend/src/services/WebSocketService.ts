class WebSocketService {
  private static instance: WebSocketService
  private ws: WebSocket | null = null
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5
  private reconnectInterval = 5000
  private messageHandlers: ((data: any) => void)[] = []
  private connectHandlers: (() => void)[] = []
  private disconnectHandlers: (() => void)[] = []
  private errorHandlers: ((error: any) => void)[] = []

  private constructor() {}

  static getInstance(): WebSocketService {
    if (!WebSocketService.instance) {
      WebSocketService.instance = new WebSocketService()
    }
    return WebSocketService.instance
  }

  connect(url: string) {
    try {
      this.ws = new WebSocket(url)

      this.ws.onopen = () => {
        console.log("WebSocket connected")
        this.reconnectAttempts = 0
        this.connectHandlers.forEach((handler) => handler())
      }

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          this.messageHandlers.forEach((handler) => handler(data))
        } catch (error) {
          console.error("Error parsing WebSocket message:", error)
        }
      }

      this.ws.onclose = () => {
        console.log("WebSocket disconnected")
        this.disconnectHandlers.forEach((handler) => handler())
        this.attemptReconnect(url)
      }

      this.ws.onerror = (error) => {
        console.error("WebSocket error:", error)
        this.errorHandlers.forEach((handler) => handler(error))
      }
    } catch (error) {
      console.error("Error connecting to WebSocket:", error)
      this.errorHandlers.forEach((handler) => handler(error))
    }
  }

  private attemptReconnect(url: string) {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++
      console.log(`Attempting to reconnect... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`)

      setTimeout(() => {
        this.connect(url)
      }, this.reconnectInterval)
    } else {
      console.error("Max reconnection attempts reached")
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
  }

  send(data: any) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data))
    } else {
      console.error("WebSocket is not connected")
    }
  }

  onMessage(handler: (data: any) => void) {
    this.messageHandlers.push(handler)
  }

  onConnect(handler: () => void) {
    this.connectHandlers.push(handler)
  }

  onDisconnect(handler: () => void) {
    this.disconnectHandlers.push(handler)
  }

  onError(handler: (error: any) => void) {
    this.errorHandlers.push(handler)
  }

  isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN
  }
}

export default WebSocketService
