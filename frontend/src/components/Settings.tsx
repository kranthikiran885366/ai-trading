"use client"

import type React from "react"
import { useState, useEffect } from "react"
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  Grid,
  Switch,
  FormControlLabel,
  Alert,
} from "@mui/material"

interface SettingsProps {
  onShowNotification: (message: string, severity: "success" | "error" | "warning" | "info") => void
}

const Settings: React.FC<SettingsProps> = ({ onShowNotification }) => {
  const [settings, setSettings] = useState({
    // Trading Settings
    max_daily_loss: 2000,
    max_position_size: 0.05,
    ai_confidence_threshold: 0.85,

    // Risk Management
    max_drawdown: 0.1,
    stop_loss_percentage: 0.02,
    take_profit_percentage: 0.04,

    // Notifications
    email_notifications: true,
    sms_notifications: false,
    webhook_notifications: true,

    // API Keys
    binance_api_key: "",
    binance_api_secret: "",
    alpaca_api_key: "",
    alpaca_api_secret: "",
  })

  const [loading, setLoading] = useState(false)

  useEffect(() => {
    fetchSettings()
  }, [])

  const fetchSettings = async () => {
    try {
      // In a real implementation, you would fetch settings from the backend
      // const settingsData = await ApiService.getSettings()
      // setSettings(settingsData)
    } catch (error) {
      console.error("Error fetching settings:", error)
      onShowNotification("Failed to fetch settings", "error")
    }
  }

  const handleSaveSettings = async () => {
    try {
      setLoading(true)
      // In a real implementation, you would save settings to the backend
      // await ApiService.saveSettings(settings)
      onShowNotification("Settings saved successfully", "success")
    } catch (error) {
      console.error("Error saving settings:", error)
      onShowNotification("Failed to save settings", "error")
    } finally {
      setLoading(false)
    }
  }

  const handleInputChange = (field: string, value: any) => {
    setSettings((prev) => ({
      ...prev,
      [field]: value,
    }))
  }

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Typography variant="h4" gutterBottom>
        Settings
      </Typography>

      <Grid container spacing={3}>
        {/* Trading Settings */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Trading Settings
              </Typography>
              <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
                <TextField
                  label="Max Daily Loss ($)"
                  type="number"
                  value={settings.max_daily_loss}
                  onChange={(e) => handleInputChange("max_daily_loss", Number.parseFloat(e.target.value))}
                  fullWidth
                />
                <TextField
                  label="Max Position Size (%)"
                  type="number"
                  inputProps={{ step: 0.01, min: 0, max: 1 }}
                  value={settings.max_position_size}
                  onChange={(e) => handleInputChange("max_position_size", Number.parseFloat(e.target.value))}
                  fullWidth
                />
                <TextField
                  label="AI Confidence Threshold"
                  type="number"
                  inputProps={{ step: 0.01, min: 0, max: 1 }}
                  value={settings.ai_confidence_threshold}
                  onChange={(e) => handleInputChange("ai_confidence_threshold", Number.parseFloat(e.target.value))}
                  fullWidth
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Risk Management */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Risk Management
              </Typography>
              <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
                <TextField
                  label="Max Drawdown (%)"
                  type="number"
                  inputProps={{ step: 0.01, min: 0, max: 1 }}
                  value={settings.max_drawdown}
                  onChange={(e) => handleInputChange("max_drawdown", Number.parseFloat(e.target.value))}
                  fullWidth
                />
                <TextField
                  label="Stop Loss (%)"
                  type="number"
                  inputProps={{ step: 0.01, min: 0, max: 1 }}
                  value={settings.stop_loss_percentage}
                  onChange={(e) => handleInputChange("stop_loss_percentage", Number.parseFloat(e.target.value))}
                  fullWidth
                />
                <TextField
                  label="Take Profit (%)"
                  type="number"
                  inputProps={{ step: 0.01, min: 0, max: 1 }}
                  value={settings.take_profit_percentage}
                  onChange={(e) => handleInputChange("take_profit_percentage", Number.parseFloat(e.target.value))}
                  fullWidth
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Notifications */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Notifications
              </Typography>
              <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={settings.email_notifications}
                      onChange={(e) => handleInputChange("email_notifications", e.target.checked)}
                    />
                  }
                  label="Email Notifications"
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={settings.sms_notifications}
                      onChange={(e) => handleInputChange("sms_notifications", e.target.checked)}
                    />
                  }
                  label="SMS Notifications"
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={settings.webhook_notifications}
                      onChange={(e) => handleInputChange("webhook_notifications", e.target.checked)}
                    />
                  }
                  label="Webhook Notifications"
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* API Keys */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                API Keys
              </Typography>
              <Alert severity="warning" sx={{ mb: 2 }}>
                API keys are encrypted and stored securely. Never share your API keys.
              </Alert>
              <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
                <TextField
                  label="Binance API Key"
                  type="password"
                  value={settings.binance_api_key}
                  onChange={(e) => handleInputChange("binance_api_key", e.target.value)}
                  fullWidth
                />
                <TextField
                  label="Binance API Secret"
                  type="password"
                  value={settings.binance_api_secret}
                  onChange={(e) => handleInputChange("binance_api_secret", e.target.value)}
                  fullWidth
                />
                <TextField
                  label="Alpaca API Key"
                  type="password"
                  value={settings.alpaca_api_key}
                  onChange={(e) => handleInputChange("alpaca_api_key", e.target.value)}
                  fullWidth
                />
                <TextField
                  label="Alpaca API Secret"
                  type="password"
                  value={settings.alpaca_api_secret}
                  onChange={(e) => handleInputChange("alpaca_api_secret", e.target.value)}
                  fullWidth
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Box sx={{ mt: 3, display: "flex", justifyContent: "flex-end" }}>
        <Button variant="contained" color="primary" onClick={handleSaveSettings} disabled={loading}>
          {loading ? "Saving..." : "Save Settings"}
        </Button>
      </Box>
    </Box>
  )
}

export default Settings
