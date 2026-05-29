# StockBot — AI Technical Analysis Chatbot

A terminal-aesthetic stock analysis chatbot that gives you deep technical analysis on any ticker or company. No API keys required.

![StockBot Preview](https://img.shields.io/badge/StockBot-Technical%20Analysis-00d4aa?style=flat-square&labelColor=0d1117)

## Features

- **Autocomplete Search** — type a ticker or company name, get instant suggestions
- **1-Month Candlestick Chart** — OHLC candles with volume, SMA20, EMA21, Bollinger Bands, and S/R lines overlaid
- **52-Week Range** — visual slider showing where price sits between yearly extremes
- **Moving Averages** — SMA 20, SMA 50, EMA 9, EMA 21 with ABOVE/BELOW price position
- **RSI (14)** — visual gauge with overbought/oversold zones
- **MACD** — MACD line, signal, histogram, and crossover signal
- **Bollinger Bands** — upper/middle/lower bands with bandwidth and price position %
- **Support & Resistance** — swing high/low detection with clustering and strength scoring
- **Level Testing** — highlights levels the price is currently testing (within 1.5%)
- **Key Stats** — market cap, P/E, beta, volume, day range, and more
- **Trend Score** — aggregated bullish/bearish signal across all indicators

## Data Source

Uses Yahoo Finance public endpoints via [allorigins.win](https://api.allorigins.win) CORS proxy.  
**No API key required.**

## Setup

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

## Build for Production

```bash
npm run build
npm run preview
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Framework | React 18 + Vite |
| Charts | [lightweight-charts](https://tradingview.github.io/lightweight-charts/) v4 |
| Icons | lucide-react |
| Styling | CSS Modules |
| Data | Yahoo Finance (via allorigins proxy) |
| Fonts | Syne + Space Mono |

## Project Structure

```
stock-analyst/
├── src/
│   ├── main.jsx          # React entry point
│   ├── App.jsx           # Main chatbot UI + message renderers
│   ├── App.module.css    # CSS Modules (dark terminal aesthetic)
│   ├── StockChart.jsx    # TradingView lightweight-charts wrapper
│   ├── stockService.js   # Data fetching + all TA calculations
│   └── index.css         # Global CSS variables & resets
├── index.html
├── vite.config.js
└── package.json
```

## Technical Indicators

All indicators are calculated from scratch in `stockService.js` — no external TA library:

- **SMA** — Simple Moving Average (20, 50 period)
- **EMA** — Exponential Moving Average (9, 21 period)  
- **RSI** — Relative Strength Index (14 period, Wilder smoothing)
- **MACD** — 12/26/9 with histogram
- **Bollinger Bands** — 20 period, 2 standard deviations, bandwidth
- **Support/Resistance** — Swing high/low detection + proximity clustering

## Usage

1. Type a ticker (`AAPL`, `NVDA`) or company name (`Tesla`, `Microsoft`)
2. Select from autocomplete or press Enter
3. Full technical analysis renders with chart and all indicators
4. Type another ticker to analyze more stocks

## Limitations

- Data via public Yahoo Finance endpoints — may occasionally be rate limited
- 1-month chart only (can be extended by changing `range=1mo` in `stockService.js`)
- 50+ tickers in autocomplete; add more in the `SEARCH_SUGGESTIONS` array
- No real-time streaming — data fetched on demand

## License

MIT
