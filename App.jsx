import { useState, useRef, useEffect, useCallback } from 'react';
import StockChart from './StockChart.jsx';
import {
  searchTickers,
  fetchStockData,
  analyzeStock,
} from './stockService.js';
import styles from './App.module.css';

function fmt(n, decimals = 2) {
  if (n == null || isNaN(n)) return 'N/A';
  return n.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
}
function fmtLarge(n) {
  if (n == null) return 'N/A';
  if (n >= 1e12) return `$${(n / 1e12).toFixed(2)}T`;
  if (n >= 1e9) return `$${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `$${(n / 1e6).toFixed(2)}M`;
  return `$${n.toLocaleString()}`;
}
function fmtVol(n) {
  if (n == null) return 'N/A';
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return n.toString();
}

function buildAnalysisMessage(symbol, analysis) {
  const { trend, rsi, rsiSignal, macd, movingAverages, bollingerBands,
    supports, resistances, testingLevels, week52High, week52Low,
    currentPrice, changePercent, marketCap, volume, avgVolume, peRatio,
    beta, bbSignal, bbPosition } = analysis;

  const pct = changePercent ? `${changePercent >= 0 ? '+' : ''}${changePercent.toFixed(2)}%` : 'N/A';
  const trendEmoji = trend === 'BULLISH' ? '🟢' : trend === 'BEARISH' ? '🔴' : '🟡';

  const rsiBar = rsi ? Math.round(rsi) : 50;
  const from52High = week52High ? ((currentPrice - week52High) / week52High * 100).toFixed(1) : null;
  const from52Low = week52Low ? ((currentPrice - week52Low) / week52Low * 100).toFixed(1) : null;

  const lines = [
    { type: 'header', text: `${symbol} — $${fmt(currentPrice)} (${pct})` },
    { type: 'divider' },

    { type: 'section', text: `${trendEmoji} TREND: ${trend}` },
    { type: 'text', text: `Price action signals ${analysis.trendSignals}/4 bullish conditions. ` +
      `Trading ${currentPrice > (movingAverages.sma20 || 0) ? 'ABOVE' : 'BELOW'} SMA20, ` +
      `${currentPrice > (movingAverages.ema21 || 0) ? 'ABOVE' : 'BELOW'} EMA21.` },

    { type: 'divider' },
    { type: 'section', text: '📊 52-WEEK RANGE' },
    {
      type: 'range52',
      high: week52High,
      low: week52Low,
      current: currentPrice,
      fromHigh: from52High,
      fromLow: from52Low,
    },

    { type: 'divider' },
    { type: 'section', text: '📉 MOVING AVERAGES' },
    {
      type: 'maTable',
      rows: [
        { label: 'SMA 20', value: movingAverages.sma20, pos: movingAverages.sma20 ? (currentPrice > movingAverages.sma20 ? 'ABOVE' : 'BELOW') : null },
        { label: 'SMA 50', value: movingAverages.sma50, pos: movingAverages.sma50 ? (currentPrice > movingAverages.sma50 ? 'ABOVE' : 'BELOW') : null },
        { label: 'EMA 9', value: movingAverages.ema9, pos: movingAverages.ema9 ? (currentPrice > movingAverages.ema9 ? 'ABOVE' : 'BELOW') : null },
        { label: 'EMA 21', value: movingAverages.ema21, pos: movingAverages.ema21 ? (currentPrice > movingAverages.ema21 ? 'ABOVE' : 'BELOW') : null },
      ]
    },

    { type: 'divider' },
    { type: 'section', text: `⚡ RSI (14): ${rsi ?? 'N/A'} — ${rsiSignal}` },
    { type: 'rsiBar', value: rsi },
    { type: 'text', text: rsi < 30 ? '⚠ Oversold — potential reversal zone. Watch for bounce.' :
      rsi > 70 ? '⚠ Overbought — momentum extended. Risk of pullback.' :
      rsi > 55 ? 'Momentum favors bulls, not yet overbought.' :
      rsi < 45 ? 'Momentum favors bears, not yet oversold.' :
      'RSI in neutral territory — no clear signal.' },

    macd && { type: 'divider' },
    macd && { type: 'section', text: '🔀 MACD' },
    macd && {
      type: 'macdRow',
      macd: macd.macd,
      signal: macd.signal,
      hist: macd.histogram,
      bullish: macd.macd > macd.signal,
    },

    bollingerBands && { type: 'divider' },
    bollingerBands && { type: 'section', text: `〰 BOLLINGER BANDS (20,2)` },
    bollingerBands && {
      type: 'bbData',
      upper: bollingerBands.upper,
      middle: bollingerBands.middle,
      lower: bollingerBands.lower,
      bandwidth: bollingerBands.bandwidth,
      position: bbPosition,
      signal: bbSignal,
    },

    { type: 'divider' },
    { type: 'section', text: '🧱 SUPPORT LEVELS' },
    supports.length > 0
      ? { type: 'levels', levels: supports, color: 'green' }
      : { type: 'text', text: 'No significant support levels identified in 1-month data.' },

    { type: 'divider' },
    { type: 'section', text: '🔒 RESISTANCE LEVELS' },
    resistances.length > 0
      ? { type: 'levels', levels: resistances, color: 'red' }
      : { type: 'text', text: 'No significant resistance levels identified in 1-month data.' },

    testingLevels.length > 0 && { type: 'divider' },
    testingLevels.length > 0 && { type: 'section', text: '🎯 CURRENTLY TESTING' },
    testingLevels.length > 0 && {
      type: 'testing',
      levels: testingLevels,
      currentPrice,
    },

    { type: 'divider' },
    { type: 'section', text: '📋 KEY STATS' },
    {
      type: 'statsGrid',
      stats: [
        { label: 'Market Cap', value: fmtLarge(marketCap) },
        { label: 'P/E Ratio', value: peRatio ? fmt(peRatio) : 'N/A' },
        { label: 'Beta', value: beta ? fmt(beta) : 'N/A' },
        { label: 'Volume', value: fmtVol(volume) },
        { label: 'Avg Vol', value: fmtVol(avgVolume) },
        { label: 'Day High', value: analysis.dayHigh ? `$${fmt(analysis.dayHigh)}` : 'N/A' },
        { label: 'Day Low', value: analysis.dayLow ? `$${fmt(analysis.dayLow)}` : 'N/A' },
        { label: 'Prev Close', value: analysis.prevClose ? `$${fmt(analysis.prevClose)}` : 'N/A' },
      ]
    },
  ].filter(Boolean);

  return lines;
}

// ── Message Renderer ──────────────────────────────────────────────────────────
function MessageBlock({ msg }) {
  if (msg.type === 'user') {
    return (
      <div className={styles.userMessage}>
        <span className={styles.userPrompt}>▶</span>
        <span>{msg.text}</span>
      </div>
    );
  }

  if (msg.type === 'error') {
    return (
      <div className={styles.errorMessage}>
        <span className={styles.errorIcon}>✕</span>
        {msg.text}
      </div>
    );
  }

  if (msg.type === 'loading') {
    return (
      <div className={styles.loadingMessage}>
        <div className={styles.loadingDots}>
          <span /><span /><span />
        </div>
        <span className={styles.loadingText}>{msg.text}</span>
      </div>
    );
  }

  if (msg.type === 'analysis') {
    const { symbol, analysis, candles, blocks } = msg;
    return (
      <div className={styles.analysisMessage}>
        <div className={styles.chartWrapper}>
          <div className={styles.chartTitle}>
            <span className={styles.symbolBadge}>{symbol}</span>
            <span className={styles.chartLabel}>1-Month Price Chart · OHLC · SMA20 · EMA21 · BB</span>
          </div>
          <StockChart candles={candles} analysis={analysis} />
        </div>
        <div className={styles.analysisBlocks}>
          {blocks.map((block, i) => <BlockRenderer key={i} block={block} />)}
        </div>
      </div>
    );
  }

  return (
    <div className={styles.botMessage}>
      {msg.text}
    </div>
  );
}

function BlockRenderer({ block }) {
  if (block.type === 'header') {
    return <div className={styles.blockHeader}>{block.text}</div>;
  }
  if (block.type === 'divider') {
    return <div className={styles.blockDivider} />;
  }
  if (block.type === 'section') {
    return <div className={styles.blockSection}>{block.text}</div>;
  }
  if (block.type === 'text') {
    return <div className={styles.blockText}>{block.text}</div>;
  }
  if (block.type === 'range52') {
    const { high, low, current, fromHigh, fromLow } = block;
    const pct = high && low ? ((current - low) / (high - low) * 100) : 50;
    return (
      <div className={styles.range52}>
        <div className={styles.rangeBar}>
          <span className={styles.rangeLow}>${fmt(low)}</span>
          <div className={styles.rangeTrack}>
            <div className={styles.rangeFill} style={{ width: `${Math.max(2, Math.min(98, pct))}%` }} />
            <div className={styles.rangeMarker} style={{ left: `${Math.max(2, Math.min(98, pct))}%` }} />
          </div>
          <span className={styles.rangeHigh}>${fmt(high)}</span>
        </div>
        <div className={styles.rangeStats}>
          <span>From 52w High: <strong className={styles.red}>{fromHigh}%</strong></span>
          <span>From 52w Low: <strong className={styles.green}>+{fromLow}%</strong></span>
        </div>
      </div>
    );
  }
  if (block.type === 'maTable') {
    return (
      <div className={styles.maTable}>
        {block.rows.map((row, i) => (
          <div key={i} className={styles.maRow}>
            <span className={styles.maLabel}>{row.label}</span>
            <span className={styles.maValue}>{row.value ? `$${fmt(row.value)}` : '—'}</span>
            {row.pos && (
              <span className={row.pos === 'ABOVE' ? styles.posAbove : styles.posBelow}>
                {row.pos}
              </span>
            )}
          </div>
        ))}
      </div>
    );
  }
  if (block.type === 'rsiBar') {
    const v = block.value ?? 50;
    return (
      <div className={styles.rsiBar}>
        <div className={styles.rsiTrack}>
          <div className={styles.rsiZoneOversold} />
          <div className={styles.rsiZoneNeutral} />
          <div className={styles.rsiZoneOverbought} />
          <div className={styles.rsiMarker} style={{ left: `${v}%` }} />
        </div>
        <div className={styles.rsiLabels}>
          <span>Oversold (30)</span>
          <span>Neutral</span>
          <span>Overbought (70)</span>
        </div>
      </div>
    );
  }
  if (block.type === 'macdRow') {
    const { macd, signal, hist, bullish } = block;
    return (
      <div className={styles.macdRow}>
        <div className={styles.macdItem}>
          <span className={styles.macdLabel}>MACD</span>
          <span className={macd >= 0 ? styles.green : styles.red}>{fmt(macd, 4)}</span>
        </div>
        <div className={styles.macdItem}>
          <span className={styles.macdLabel}>Signal</span>
          <span className={styles.text}>{fmt(signal, 4)}</span>
        </div>
        <div className={styles.macdItem}>
          <span className={styles.macdLabel}>Histogram</span>
          <span className={hist >= 0 ? styles.green : styles.red}>{fmt(hist, 4)}</span>
        </div>
        <div className={styles.macdItem}>
          <span className={styles.macdLabel}>Signal</span>
          <span className={bullish ? styles.green : styles.red}>{bullish ? 'BULLISH CROSS' : 'BEARISH CROSS'}</span>
        </div>
      </div>
    );
  }
  if (block.type === 'bbData') {
    const { upper, middle, lower, bandwidth, position, signal } = block;
    const sigLabel = {
      AT_UPPER: '⚠ At Upper Band — stretched',
      AT_LOWER: '⚠ At Lower Band — potential bounce',
      UPPER_HALF: 'Price in upper half of bands',
      LOWER_HALF: 'Price in lower half of bands',
    }[signal] || '';
    return (
      <div className={styles.bbData}>
        <div className={styles.bbRow}>
          <span className={styles.red}>Upper</span><span>${fmt(upper)}</span>
          <span className={styles.textMuted}>Middle</span><span>${fmt(middle)}</span>
          <span className={styles.green}>Lower</span><span>${fmt(lower)}</span>
        </div>
        <div className={styles.bbRow2}>
          <span>Bandwidth: <strong>{bandwidth}%</strong></span>
          <span>Position: <strong>{position}th percentile</strong></span>
        </div>
        {sigLabel && <div className={styles.bbSignal}>{sigLabel}</div>}
      </div>
    );
  }
  if (block.type === 'levels') {
    const isGreen = block.color === 'green';
    return (
      <div className={styles.levelsList}>
        {block.levels.map((l, i) => (
          <div key={i} className={styles.levelRow}>
            <span className={isGreen ? styles.green : styles.red}>
              ${fmt(l.price)}
            </span>
            <div className={styles.levelBar}>
              <div
                className={isGreen ? styles.levelBarFillGreen : styles.levelBarFillRed}
                style={{ width: `${Math.min(100, l.strength * 25)}%` }}
              />
            </div>
            <span className={styles.levelStrength}>
              {'●'.repeat(Math.min(l.strength, 4))} strength {l.strength}
            </span>
          </div>
        ))}
      </div>
    );
  }
  if (block.type === 'testing') {
    return (
      <div className={styles.testingLevels}>
        <div className={styles.testingAlert}>
          🎯 Currently testing {block.levels.length} level{block.levels.length > 1 ? 's' : ''}:
        </div>
        {block.levels.map((l, i) => {
          const dist = ((block.currentPrice - l.price) / l.price * 100).toFixed(2);
          return (
            <div key={i} className={styles.testingRow}>
              <span className={l.type === 'support' ? styles.green : styles.red}>
                {l.label}
              </span>
              <span className={styles.testingPrice}>${fmt(l.price)}</span>
              <span className={styles.textMuted}>{dist > 0 ? `+${dist}` : dist}% away</span>
            </div>
          );
        })}
      </div>
    );
  }
  if (block.type === 'statsGrid') {
    return (
      <div className={styles.statsGrid}>
        {block.stats.map((s, i) => (
          <div key={i} className={styles.statCell}>
            <span className={styles.statLabel}>{s.label}</span>
            <span className={styles.statValue}>{s.value}</span>
          </div>
        ))}
      </div>
    );
  }
  return null;
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [input, setInput] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const [messages, setMessages] = useState([
    {
      type: 'bot',
      text: 'Enter a stock ticker or company name to get a full technical analysis — price chart, RSI, MACD, Bollinger Bands, moving averages, support/resistance levels, and more.',
    }
  ]);
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleInput = (val) => {
    setInput(val);
    setSuggestions(val.length >= 1 ? searchTickers(val) : []);
  };

  const analyze = useCallback(async (symbol) => {
    if (!symbol || loading) return;
    setInput('');
    setSuggestions([]);
    setLoading(true);

    setMessages(prev => [
      ...prev,
      { type: 'user', text: symbol },
      { type: 'loading', text: `Fetching ${symbol} data...` },
    ]);

    try {
      const { quote, candles } = await fetchStockData(symbol.toUpperCase());
      const analysis = analyzeStock(candles, quote);
      const blocks = buildAnalysisMessage(symbol.toUpperCase(), analysis);

      setMessages(prev => {
        const without = prev.filter(m => m.type !== 'loading');
        return [
          ...without,
          {
            type: 'analysis',
            symbol: symbol.toUpperCase(),
            analysis,
            candles,
            blocks,
          }
        ];
      });
    } catch (err) {
      setMessages(prev => {
        const without = prev.filter(m => m.type !== 'loading');
        return [
          ...without,
          {
            type: 'error',
            text: `Could not fetch data for "${symbol}". Check the ticker and try again. (${err.message})`,
          }
        ];
      });
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  }, [loading]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && input.trim()) {
      analyze(input.trim().toUpperCase());
    }
    if (e.key === 'Escape') {
      setSuggestions([]);
    }
  };

  return (
    <div className={styles.app}>
      {/* Header */}
      <header className={styles.header}>
        <div className={styles.headerInner}>
          <div className={styles.logo}>
            <span className={styles.logoIcon}>◈</span>
            <span className={styles.logoText}>STOCK<span className={styles.logoAccent}>BOT</span></span>
          </div>
          <div className={styles.headerMeta}>
            <span className={styles.metaBadge}>TECHNICAL ANALYSIS</span>
            <span className={styles.metaBadge}>REAL-TIME DATA</span>
          </div>
        </div>
      </header>

      {/* Chat */}
      <main className={styles.main}>
        <div className={styles.messages}>
          {messages.map((msg, i) => (
            <MessageBlock key={i} msg={msg} />
          ))}
          <div ref={bottomRef} />
        </div>
      </main>

      {/* Input */}
      <footer className={styles.footer}>
        <div className={styles.inputWrapper}>
          <div className={styles.inputRow}>
            <span className={styles.inputPrefix}>$</span>
            <input
              ref={inputRef}
              className={styles.input}
              value={input}
              onChange={e => handleInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Enter ticker or company... (e.g. AAPL, NVDA, Tesla)"
              disabled={loading}
              autoComplete="off"
              spellCheck={false}
            />
            <button
              className={styles.analyzeBtn}
              onClick={() => input.trim() && analyze(input.trim().toUpperCase())}
              disabled={loading || !input.trim()}
            >
              {loading ? '...' : 'ANALYZE'}
            </button>
          </div>

          {suggestions.length > 0 && (
            <div className={styles.suggestions}>
              {suggestions.map((s, i) => (
                <button
                  key={i}
                  className={styles.suggestion}
                  onClick={() => analyze(s.symbol)}
                >
                  <span className={styles.sugSymbol}>{s.symbol}</span>
                  <span className={styles.sugName}>{s.name}</span>
                </button>
              ))}
            </div>
          )}
        </div>
      </footer>
    </div>
  );
}
