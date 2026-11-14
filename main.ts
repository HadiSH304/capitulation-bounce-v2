import { runBars, BarsState, type Batch, type Fill } from './vendor/makefun-bars.ts';

const SYMBOL = 'CBBTC';
const WARMUP = 360;
const MAX_SEQ = 1300;
const RANGE_PCT = 96;        // 96th percentile
const DROP_PCT_NUM = 962n;   // 3.8% drop: close * 1000 <= high * 962
const DROP_PCT_DEN = 1000n;
const TP_NUM = 1010n;        // +1% take profit
const TP_DEN = 1000n;
const SL_NUM = 988n;         // -1.2% stop loss
const SL_DEN = 1000n;
const TIME_STOP = 300;       // 300 bars max hold
const BUY_PCT = 95n;         // 95% of cash
const MAX_SLIPPAGE = 100;

// State
const ranges: bigint[] = [];
let rollingHigh = 0n;
let entryPrice = 0n;
let entrySeq = -1;

function percentile(arr: bigint[], p: number): bigint {
  if (arr.length === 0) return 0n;
  const sorted = [...arr].sort((a, b) => (a > b ? 1 : -1));
  const idx = Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length));
  return sorted[idx];
}

async function onBatch(state: BarsState, batch: Batch): Promise<readonly { side: 'buy' | 'sell'; symbol: string; notional_quote_units?: bigint; qty?: bigint; max_slippage_bps: number }[] | void> {
  const bar = state.latest.get(SYMBOL);
  if (!bar) return;
  const close = BigInt(bar.c);
  const range = BigInt(bar.h) - BigInt(bar.l);
  ranges.push(range);

  rollingHigh = rollingHigh === 0n ? close : (close > rollingHigh ? close : rollingHigh);

  const pos = state.positions.get(SYMBOL);
  const holding = pos && pos.qty > 0n;

  // Exit logic
  if (holding) {
    const seq = batch.seq;
    if (close * TP_DEN >= entryPrice * TP_NUM ||
        close * SL_DEN <= entryPrice * SL_NUM ||
        seq - entrySeq >= TIME_STOP) {
      return [{ side: 'sell', symbol: SYMBOL, qty: pos.qty, max_slippage_bps: MAX_SLIPPAGE }];
    }
    return;
  }

  // Entry logic
  if (batch.seq < WARMUP || batch.seq >= MAX_SEQ || state.warmingUp) return;

  const pctile = percentile(ranges, RANGE_PCT);
  if (range < pctile) return;
  if (close * DROP_PCT_DEN > rollingHigh * DROP_PCT_NUM) return;

  const notional = state.cash * BUY_PCT / 100n;
  if (notional <= 0n) return;

  return [{ side: 'buy', symbol: SYMBOL, notional_quote_units: notional, max_slippage_bps: MAX_SLIPPAGE }];
}

// Track fills to capture entry price/seq
async function onBatchWrapper(state: BarsState, batch: Batch) {
  for (const fill of batch.fills) {
    if (fill.symbol === SYMBOL && fill.status === 'filled' && fill.side === 'buy') {
      entryPrice = BigInt(fill.price);
      entrySeq = batch.seq;
    }
  }
  return onBatch(state, batch);
}

await runBars({ onBatch: onBatchWrapper });