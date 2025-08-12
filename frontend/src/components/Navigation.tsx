"use client"

import type React from "react"
import { useLocation, useNavigate } from "react-router-dom"
import { Drawer, List, ListItem, ListItemButton, ListItemIcon, ListItemText, Toolbar } from "@mui/material"
import { Dashboard, TrendingUp, AccountBalance, Settings } from "@mui/icons-material"

const drawerWidth = 240

const Navigation: React.FC = () => {
  const location = useLocation()
  const navigate = useNavigate()

  const menuItems = [
    { text: "Dashboard", icon: <Dashboard />, path: "/" },
    { text: "Trading", icon: <TrendingUp />, path: "/trading" },
    { text: "Portfolio", icon: <AccountBalance />, path: "/portfolio" },
    { text: "Settings", icon: <Settings />, path: "/settings" },
  ]

  return (
    <Drawer
      variant="permanent"
      sx={{
        width: drawerWidth,
        flexShrink: 0,
        "& .MuiDrawer-paper": {
          width: drawerWidth,
          boxSizing: "border-box",
          backgroundColor: "background.paper",
        },
      }}
    >
      <Toolbar />
      <List>
        {menuItems.map((item) => (
          <ListItem key={item.text} disablePadding>
            <ListItemButton selected={location.pathname === item.path} onClick={() => navigate(item.path)}>
              <ListItemIcon sx={{ color: location.pathname === item.path ? "primary.main" : "inherit" }}>
                {item.icon}
              </ListItemIcon>
              <ListItemText primary={item.text} />
            </ListItemButton>
          </ListItem>
        ))}
      </List>
    </Drawer>
  )
}

export default Navigation
