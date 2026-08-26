import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import './i18n'
import App from './App.jsx'
import PoleFigureWindow from './components/PoleFigure/PoleFigureWindow.jsx'
import { ThemeProvider } from './theme/ThemeProvider.jsx'
import { installGlobalErrorReporter } from './services/errorReporter'

// Before render, so even a crash in App/ThemeProvider (blank window — there
// is deliberately no boundary above App) still lands in the backend log.
installGlobalErrorReporter()

const isPoleFigure = new URLSearchParams(window.location.search).get('view') === 'polefigure'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ThemeProvider>
      {isPoleFigure ? <PoleFigureWindow /> : <App />}
    </ThemeProvider>
  </StrictMode>,
)

console.log(`Orienta${isPoleFigure ? ' (pole-figure window)' : ''}`)
