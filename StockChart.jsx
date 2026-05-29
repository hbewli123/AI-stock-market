import { useEffect, useRef } from 'react';

// Dynamic import of lightweight-charts
let chartLib = null;

export default function StockChart({ candles, analysis }) {
  const chartRef = useRef(null);
  const containerRef = useRef(null);

  useEffect(() => {
    if (!candles || candles.length === 0 || !containerRef.current) return;

    let chart, candleSeries, volumeSeries;

    async function initChart() {
      if (!chartLib) {
        chartLib = await import('lightweight-charts');
      }
      const { createChart, CandlestickSeries, HistogramSeries, LineSeries } = chartLib;

      // Clear previous chart
      containerRef.current.innerHTML = '';

      chart = createChart(containerRef.current, {
        width: containerRef.current.clientWidth,
        height: 380,
        layout: {
          background: { color: '#0d1117' },
          textColor: '#8b949e',
          fontFamily: "'Space Mono', monospace",
          fontSize: 11,
        },
        grid: {
          vertLines: { color: '#1e2d3d' },
          horzLines: { color: '#1e2d3d' },
        },
        crosshair: {
          mode: 1,
          vertLine: { color: '#00d4aa', width: 1, style: 3 },
          horzLine: { color: '#00d4aa', width: 1, style: 3 },
        },
        rightPriceScale: {
          borderColor: '#1e2d3d',
          scaleMargins: { top: 0.1, bottom: 0.25 },
        },
        timeScale: {
          borderColor: '#1e2d3d',
          timeVisible: true,
          secondsVisible: false,
        },
      });

      // Candlestick series
      candleSeries = chart.addSeries(CandlestickSeries, {
        upColor: '#00d4aa',
        downColor: '#ff4d6d',
        borderUpColor: '#00d4aa',
        borderDownColor: '#ff4d6d',
        wickUpColor: '#00d4aa',
        wickDownColor: '#ff4d6d',
      });
      candleSeries.setData(candles);

      // Volume
      volumeSeries = chart.addSeries(HistogramSeries, {
        color: '#00d4aa',
        priceFormat: { type: 'volume' },
        priceScaleId: 'volume',
      });
      chart.priceScale('volume').applyOptions({
        scaleMargins: { top: 0.8, bottom: 0 },
      });
      volumeSeries.setData(candles.map(c => ({
        time: c.time,
        value: c.volume,
        color: c.close >= c.open ? 'rgba(0,212,170,0.3)' : 'rgba(255,77,109,0.3)',
      })));

      // SMA 20
      if (analysis?.movingAverages?.sma20) {
        const smaData = candles.slice(-20).map((c, i) => {
          if (i < 19) return null;
          const slice = candles.slice(-20 + i - 19, -20 + i + 1);
          const avg = slice.reduce((s, x) => s + x.close, 0) / 20;
          return { time: c.time, value: parseFloat(avg.toFixed(2)) };
        }).filter(Boolean);
        
        // Simple approach: just show the last known SMA value as a line
        const sma20Series = chart.addSeries(LineSeries, {
          color: '#ffd166',
          lineWidth: 1,
          title: 'SMA20',
          priceLineVisible: false,
          lastValueVisible: true,
        });

        // Calculate all SMAs
        const allSmaData = [];
        for (let i = 19; i < candles.length; i++) {
          const slice = candles.slice(i - 19, i + 1);
          const avg = slice.reduce((s, x) => s + x.close, 0) / 20;
          allSmaData.push({ time: candles[i].time, value: parseFloat(avg.toFixed(2)) });
        }
        sma20Series.setData(allSmaData);
      }

      // EMA 21
      if (analysis?.movingAverages?.ema21) {
        const ema21Series = chart.addSeries(LineSeries, {
          color: '#b56bff',
          lineWidth: 1,
          title: 'EMA21',
          priceLineVisible: false,
          lastValueVisible: true,
          lineStyle: 2,
        });
        const closes = candles.map(c => c.close);
        const k = 2 / 22;
        let ema = closes[0];
        const emaData = candles.map((c, i) => {
          if (i === 0) return { time: c.time, value: parseFloat(ema.toFixed(2)) };
          ema = closes[i] * k + ema * (1 - k);
          return { time: c.time, value: parseFloat(ema.toFixed(2)) };
        });
        ema21Series.setData(emaData);
      }

      // BB Upper/Lower
      if (analysis?.bollingerBands) {
        const { upper, lower } = analysis.bollingerBands;
        const bbUpperSeries = chart.addSeries(LineSeries, {
          color: 'rgba(77,166,255,0.5)',
          lineWidth: 1,
          title: 'BB+',
          priceLineVisible: false,
          lastValueVisible: false,
          lineStyle: 3,
        });
        const bbLowerSeries = chart.addSeries(LineSeries, {
          color: 'rgba(77,166,255,0.5)',
          lineWidth: 1,
          title: 'BB-',
          priceLineVisible: false,
          lastValueVisible: false,
          lineStyle: 3,
        });

        // Calculate BB for each point
        const bbUpperData = [], bbLowerData = [];
        for (let i = 19; i < candles.length; i++) {
          const slice = candles.slice(i - 19, i + 1).map(c => c.close);
          const mean = slice.reduce((s, v) => s + v, 0) / 20;
          const variance = slice.reduce((s, v) => s + Math.pow(v - mean, 2), 0) / 20;
          const sd = Math.sqrt(variance);
          bbUpperData.push({ time: candles[i].time, value: parseFloat((mean + 2 * sd).toFixed(2)) });
          bbLowerData.push({ time: candles[i].time, value: parseFloat((mean - 2 * sd).toFixed(2)) });
        }
        bbUpperSeries.setData(bbUpperData);
        bbLowerSeries.setData(bbLowerData);
      }

      // Support/Resistance price lines
      if (analysis?.supports) {
        analysis.supports.slice(0, 2).forEach(s => {
          candleSeries.createPriceLine({
            price: s.price,
            color: 'rgba(0,212,170,0.6)',
            lineWidth: 1,
            lineStyle: 2,
            title: `S ${s.price}`,
          });
        });
      }
      if (analysis?.resistances) {
        analysis.resistances.slice(0, 2).forEach(r => {
          candleSeries.createPriceLine({
            price: r.price,
            color: 'rgba(255,77,109,0.6)',
            lineWidth: 1,
            lineStyle: 2,
            title: `R ${r.price}`,
          });
        });
      }

      chart.timeScale().fitContent();

      // Resize handler
      const resizeObserver = new ResizeObserver(() => {
        if (containerRef.current) {
          chart.resize(containerRef.current.clientWidth, 380);
        }
      });
      resizeObserver.observe(containerRef.current);
      chartRef.current = { chart, resizeObserver };
    }

    initChart();

    return () => {
      if (chartRef.current) {
        chartRef.current.resizeObserver?.disconnect();
        chartRef.current.chart?.remove();
        chartRef.current = null;
      }
    };
  }, [candles, analysis]);

  return (
    <div
      ref={containerRef}
      style={{
        width: '100%',
        height: '380px',
        borderRadius: '4px',
        overflow: 'hidden',
      }}
    />
  );
}
