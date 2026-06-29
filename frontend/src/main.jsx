import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import './i18n'
import App from './App.jsx'
import PoleFigureWindow from './components/PoleFigure/PoleFigureWindow.jsx'
import { ThemeProvider } from './theme/ThemeProvider.jsx'

const APP_VERSION = '1.0.0'
const isPoleFigure = new URLSearchParams(window.location.search).get('view') === 'polefigure'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ThemeProvider>
      {isPoleFigure ? <PoleFigureWindow /> : <App />}
    </ThemeProvider>
  </StrictMode>,
)

console.log(`Orienta v${APP_VERSION}${isPoleFigure ? ' (pole-figure window)' : ''}`)
