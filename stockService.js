// Stock data fetched from Yahoo Finance via public endpoints
// No API key required for basic quote/chart data

const SEARCH_SUGGESTIONS = [
  { symbol: 'AAPL', name: 'Apple Inc.' },
  { symbol: 'MSFT', name: 'Microsoft Corporation' },
  { symbol: 'GOOGL', name: 'Alphabet Inc.' },
  { symbol: 'AMZN', name: 'Amazon.com Inc.' },
  { symbol: 'NVDA', name: 'NVIDIA Corporation' },
  { symbol: 'META', name: 'Meta Platforms Inc.' },
  { symbol: 'TSLA', name: 'Tesla Inc.' },
  { symbol: 'NFLX', name: 'Netflix Inc.' },
  { symbol: 'AMD', name: 'Advanced Micro Devices' },
  { symbol: 'INTC', name: 'Intel Corporation' },
  { symbol: 'CRM', name: 'Salesforce Inc.' },
  { symbol: 'ORCL', name: 'Oracle Corporation' },
  { symbol: 'ADBE', name: 'Adobe Inc.' },
  { symbol: 'PYPL', name: 'PayPal Holdings' },
  { symbol: 'SHOP', name: 'Shopify Inc.' },
  { symbol: 'UBER', name: 'Uber Technologies' },
  { symbol: 'LYFT', name: 'Lyft Inc.' },
  { symbol: 'SQ', name: 'Block Inc.' },
  { symbol: 'COIN', name: 'Coinbase Global' },
  { symbol: 'HOOD', name: 'Robinhood Markets' },
  { symbol: 'SPY', name: 'SPDR S&P 500 ETF' },
  { symbol: 'QQQ', name: 'Invesco QQQ Trust' },
  { symbol: 'IWM', name: 'iShares Russell 2000 ETF' },
  { symbol: 'DIA', name: 'SPDR Dow Jones ETF' },
  { symbol: 'GLD', name: 'SPDR Gold Shares' },
  { symbol: 'PLTR', name: 'Palantir Technologies' },
  { symbol: 'SOFI', name: 'SoFi Technologies' },
  { symbol: 'RIVN', name: 'Rivian Automotive' },
  { symbol: 'LCID', name: 'Lucid Group' },
  { symbol: 'NIO', name: 'NIO Inc.' },
  { symbol: 'BABA', name: 'Alibaba Group' },
  { symbol: 'JNJ', name: 'Johnson & Johnson' },
  { symbol: 'PFE', name: 'Pfizer Inc.' },
  { symbol: 'MRNA', name: 'Moderna Inc.' },
  { symbol: 'JPM', name: 'JPMorgan Chase' },
  { symbol: 'BAC', name: 'Bank of America' },
  { symbol: 'GS', name: 'Goldman Sachs' },
  { symbol: 'MS', name: 'Morgan Stanley' },
  { symbol: 'V', name: 'Visa Inc.' },
  { symbol: 'MA', name: 'Mastercard Inc.' },
  { symbol: 'BRK.B', name: 'Berkshire Hathaway' },
  { symbol: 'XOM', name: 'Exxon Mobil' },
  { symbol: 'CVX', name: 'Chevron Corporation' },
  { symbol: 'WMT', name: 'Walmart Inc.' },
  { symbol: 'TGT', name: 'Target Corporation' },
  { symbol: 'COST', name: 'Costco Wholesale' },
  { symbol: 'DIS', name: 'The Walt Disney Company' },
  { symbol: 'T', name: 'AT&T Inc.' },
  { symbol: 'VZ', name: 'Verizon Communications' },
  { symbol: 'ABNB', name: 'Airbnb Inc.' },
  { symbol: 'SNAP', name: 'Snap Inc.' },
  { symbol: 'PINS', name: 'Pinterest Inc.' },
  { symbol: 'TWTR', name: 'Twitter/X Corp' },
  { symbol: 'ZM', name: 'Zoom Video Communications' },
  { symbol: 'DOCU', name: 'DocuSign Inc.' },
  { symbol: 'NOW', name: 'ServiceNow Inc.' },
  { symbol: 'SNOW', name: 'Snowflake Inc.' },
  { symbol: 'DDOG', name: 'Datadog Inc.' },
  { symbol: 'NET', name: 'Cloudflare Inc.' },
  { symbol: 'MDB', name: 'MongoDB Inc.' },
  { symbol: 'OKTA', name: 'Okta Inc.' },
];

export function searchTickers(query) {
  if (!query || query.length < 1) return [];
  const q = query.toUpperCase();
  const ql = query.toLowerCase();
  return SEARCH_SUGGESTIONS.filter(s =>
    s.symbol.startsWith(q) ||
    s.name.toLowerCase().includes(ql)
  ).slice(0, 8);
}

// Fetch from Yahoo Finance via allorigins CORS proxy
async function yahooFetch(path) {
  const url = `https://query1.finance.yahoo.com${path}`;
  const proxy = `https://api.allorigins.win/get?url=${encodeURIComponent(url)}`;
  const res = await fetch(proxy);
  const outer = await res.json();
  return JSON.parse(outer.contents);
}

export async function fetchStockData(symbol) {
  const [quoteData, chartData] = await Promise.all([
    yahooFetch(`/v8/finance/quote?symbols=${symbol}`),
    yahooFetch(`/v8/finance/chart/${symbol}?interval=1d&range=1mo`)
  ]);

  const quote = quoteData?.quoteResponse?.result?.[0];
  const chart = chartData?.chart?.result?.[0];

  if (!quote || !chart) throw new Error(`No data found for ${symbol}`);

  const timestamps = chart.timestamp;
  const ohlcv = chart.indicators.quote[0];

  const candles = timestamps.map((t, i) => ({
    time: t,
    open: parseFloat(ohlcv.open[i]?.toFixed(2)),
    high: parseFloat(ohlcv.high[i]?.toFixed(2)),
    low: parseFloat(ohlcv.low[i]?.toFixed(2)),
    close: parseFloat(ohlcv.close[i]?.toFixed(2)),
    volume: ohlcv.volume[i] || 0,
  })).filter(c => c.open && c.high && c.low && c.close);

  return { quote, candles };
}

// ── Technical Analysis Calculations ──────────────────────────────────────────

export function calcSMA(closes, period) {
  const result = [];
  for (let i = period - 1; i < closes.length; i++) {
    const slice = closes.slice(i - period + 1, i + 1);
    result.push(slice.reduce((a, b) => a + b, 0) / period);
  }
  return result;
}

export function calcEMA(closes, period) {
  const k = 2 / (period + 1);
  const result = [closes[0]];
  for (let i = 1; i < closes.length; i++) {
    result.push(closes[i] * k + result[i - 1] * (1 - k));
  }
  return result;
}

export function calcRSI(closes, period = 14) {
  if (closes.length < period + 1) return null;
  let gains = 0, losses = 0;
  for (let i = 1; i <= period; i++) {
    const diff = closes[i] - closes[i - 1];
    if (diff >= 0) gains += diff; else losses -= diff;
  }
  let avgGain = gains / period;
  let avgLoss = losses / period;
  for (let i = period + 1; i < closes.length; i++) {
    const diff = closes[i] - closes[i - 1];
    avgGain = (avgGain * (period - 1) + Math.max(diff, 0)) / period;
    avgLoss = (avgLoss * (period - 1) + Math.max(-diff, 0)) / period;
  }
  if (avgLoss === 0) return 100;
  const rs = avgGain / avgLoss;
  return parseFloat((100 - 100 / (1 + rs)).toFixed(2));
}

export function calcMACD(closes) {
  if (closes.length < 26) return null;
  const ema12 = calcEMA(closes, 12);
  const ema26 = calcEMA(closes, 26);
  const macdLine = ema12.slice(ema26.length - ema12.length).map((v, i) => v - ema26[i]);
  const signal = calcEMA(macdLine, 9);
  const histogram = macdLine.slice(macdLine.length - signal.length).map((v, i) => v - signal[i]);
  return {
    macd: parseFloat(macdLine[macdLine.length - 1].toFixed(4)),
    signal: parseFloat(signal[signal.length - 1].toFixed(4)),
    histogram: parseFloat(histogram[histogram.length - 1].toFixed(4)),
  };
}

export function calcBollingerBands(closes, period = 20, stdDev = 2) {
  if (closes.length < period) return null;
  const slice = closes.slice(-period);
  const mean = slice.reduce((a, b) => a + b, 0) / period;
  const variance = slice.reduce((s, v) => s + Math.pow(v - mean, 2), 0) / period;
  const sd = Math.sqrt(variance);
  return {
    upper: parseFloat((mean + stdDev * sd).toFixed(2)),
    middle: parseFloat(mean.toFixed(2)),
    lower: parseFloat((mean - stdDev * sd).toFixed(2)),
    bandwidth: parseFloat(((stdDev * 2 * sd) / mean * 100).toFixed(2)),
  };
}

export function findSupportResistance(candles) {
  if (candles.length < 10) return { supports: [], resistances: [] };
  
  const levels = [];
  const prices = candles.map(c => c.close);
  const highs = candles.map(c => c.high);
  const lows = candles.map(c => c.low);
  
  // Find local swing highs/lows
  for (let i = 2; i < candles.length - 2; i++) {
    // Swing high
    if (highs[i] > highs[i-1] && highs[i] > highs[i-2] &&
        highs[i] > highs[i+1] && highs[i] > highs[i+2]) {
      levels.push({ price: highs[i], type: 'resistance' });
    }
    // Swing low
    if (lows[i] < lows[i-1] && lows[i] < lows[i-2] &&
        lows[i] < lows[i+1] && lows[i] < lows[i+2]) {
      levels.push({ price: lows[i], type: 'support' });
    }
  }

  // Cluster nearby levels (within 1%)
  const clustered = [];
  const used = new Set();
  for (let i = 0; i < levels.length; i++) {
    if (used.has(i)) continue;
    const group = [levels[i]];
    for (let j = i + 1; j < levels.length; j++) {
      if (!used.has(j) && Math.abs(levels[j].price - levels[i].price) / levels[i].price < 0.01) {
        group.push(levels[j]);
        used.add(j);
      }
    }
    const avgPrice = group.reduce((s, l) => s + l.price, 0) / group.length;
    const dominantType = group.filter(l => l.type === 'support').length >= group.filter(l => l.type === 'resistance').length
      ? 'support' : 'resistance';
    clustered.push({ price: parseFloat(avgPrice.toFixed(2)), type: dominantType, strength: group.length });
    used.add(i);
  }

  const currentPrice = prices[prices.length - 1];
  const supports = clustered
    .filter(l => l.price < currentPrice && l.type === 'support')
    .sort((a, b) => b.price - a.price)
    .slice(0, 4);
  const resistances = clustered
    .filter(l => l.price > currentPrice && l.type === 'resistance')
    .sort((a, b) => a.price - b.price)
    .slice(0, 4);

  return { supports, resistances };
}

export function analyzeStock(candles, quote) {
  const closes = candles.map(c => c.close);
  const currentPrice = closes[closes.length - 1];

  const sma20 = calcSMA(closes, Math.min(20, closes.length));
  const sma50 = calcSMA(closes, Math.min(50, closes.length));
  const ema9 = calcEMA(closes, Math.min(9, closes.length));
  const ema21 = calcEMA(closes, Math.min(21, closes.length));

  const rsi = calcRSI(closes, 14);
  const macd = calcMACD(closes);
  const bb = calcBollingerBands(closes);
  const { supports, resistances } = findSupportResistance(candles);

  const sma20val = sma20[sma20.length - 1];
  const sma50val = sma50[sma50.length - 1];
  const ema9val = ema9[ema9.length - 1];
  const ema21val = ema21[ema21.length - 1];

  // Determine trend
  let trend = 'NEUTRAL';
  let trendSignals = 0;
  if (currentPrice > sma20val) trendSignals++;
  if (sma20val > sma50val) trendSignals++;
  if (ema9val > ema21val) trendSignals++;
  if (macd && macd.macd > macd.signal) trendSignals++;
  if (trendSignals >= 3) trend = 'BULLISH';
  else if (trendSignals <= 1) trend = 'BEARISH';

  // RSI interpretation
  let rsiSignal = 'NEUTRAL';
  if (rsi !== null) {
    if (rsi >= 70) rsiSignal = 'OVERBOUGHT';
    else if (rsi <= 30) rsiSignal = 'OVERSOLD';
    else if (rsi >= 55) rsiSignal = 'BULLISH';
    else if (rsi <= 45) rsiSignal = 'BEARISH';
  }

  // BB position
  let bbSignal = null;
  let bbPosition = null;
  if (bb) {
    const range = bb.upper - bb.lower;
    bbPosition = parseFloat(((currentPrice - bb.lower) / range * 100).toFixed(1));
    if (currentPrice >= bb.upper) bbSignal = 'AT_UPPER';
    else if (currentPrice <= bb.lower) bbSignal = 'AT_LOWER';
    else if (bbPosition > 60) bbSignal = 'UPPER_HALF';
    else bbSignal = 'LOWER_HALF';
  }

  // Key level testing
  const allLevels = [
    ...supports.map(s => ({ ...s, label: 'Support' })),
    ...resistances.map(r => ({ ...r, label: 'Resistance' })),
    bb && { price: bb.upper, label: 'BB Upper', type: 'resistance' },
    bb && { price: bb.lower, label: 'BB Lower', type: 'support' },
    sma20val && { price: parseFloat(sma20val.toFixed(2)), label: 'SMA 20', type: currentPrice > sma20val ? 'support' : 'resistance' },
    ema21val && { price: parseFloat(ema21val.toFixed(2)), label: 'EMA 21', type: currentPrice > ema21val ? 'support' : 'resistance' },
  ].filter(Boolean);

  const testingLevels = allLevels.filter(l =>
    Math.abs(l.price - currentPrice) / currentPrice < 0.015
  );

  return {
    currentPrice,
    trend,
    trendSignals,
    movingAverages: {
      sma20: sma20val ? parseFloat(sma20val.toFixed(2)) : null,
      sma50: sma50val ? parseFloat(sma50val.toFixed(2)) : null,
      ema9: ema9val ? parseFloat(ema9val.toFixed(2)) : null,
      ema21: ema21val ? parseFloat(ema21val.toFixed(2)) : null,
    },
    rsi,
    rsiSignal,
    macd,
    bollingerBands: bb,
    bbSignal,
    bbPosition,
    supports,
    resistances,
    testingLevels,
    week52High: quote.fiftyTwoWeekHigh,
    week52Low: quote.fiftyTwoWeekLow,
    marketCap: quote.marketCap,
    volume: quote.regularMarketVolume,
    avgVolume: quote.averageDailyVolume3Month,
    peRatio: quote.trailingPE,
    eps: quote.epsTrailingTwelveMonths,
    beta: quote.beta,
    shortName: quote.shortName || quote.longName,
    exchange: quote.fullExchangeName,
    currency: quote.currency,
    dayHigh: quote.regularMarketDayHigh,
    dayLow: quote.regularMarketDayLow,
    openPrice: quote.regularMarketOpen,
    prevClose: quote.regularMarketPreviousClose,
    changePercent: quote.regularMarketChangePercent,
  };
}
