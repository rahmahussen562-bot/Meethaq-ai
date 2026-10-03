import React from 'react'
import ReactDOM from 'react-dom/client'
import App, { ErrorBoundary } from './App'
import './index.css'

const rootElement = document.getElementById('root')

if (!rootElement) {
  console.error("FATAL: #root element not found in DOM!")
} else {
  ReactDOM.createRoot(rootElement).render(
    <React.StrictMode>
      <ErrorBoundary>
        <App />
      </ErrorBoundary>
    </React.StrictMode>,
  )
}
