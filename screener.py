"""
Auktionsmarkt-Screener v2
Methodische Grundlage: Tom Alexander – „Trading Without Crutches" (2015)

Findet Aktien im S&P 500 mit aktiven Balance-Areas und klassifiziert Setups:
  - Responsive Buy  (Preis am unteren Extrem, Long-Bias)
  - Responsive Sell (Preis am oberen Extrem, Short-Bias)
  - Initiative Buy  (frischer Ausbruch nach oben, Long-Bias)
  - Initiative Sell (frischer Ausbruch nach unten, Short-Bias)
  - ... Watch-Varianten für Setups, die sich nähern

HVN-Näherung : VWAP der Balance-Area
Trendfilter   : Preis vs. 20-Tage-VWAP

Ausgabe: index.html (für GitHub Pages)

Benötigte Pakete:
  pip install yfinance pandas requests
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import sys
from datetime import datetime

# ─────────────────────────────────────────────────────────────────
# KONFIGURATION  — hier anpassen
# ─────────────────────────────────────────────────────────────────
MIN_BALANCE_DAYS   = 4      # Mindest-Konsolidierungstage
MAX_RANGE_PCT      = 9.0    # Max. Gesamt-Range der Balance-Area (% des Preises)
NEAR_EXTREME_PCT   = 3.0    # Ab wann gilt der Preis als "nahe am Extrem" (%)
BREAKOUT_MAX_PCT   = 5.0    # Frischer Ausbruch: Preis max. X % über/unter Extrem
LOOKBACK_DAYS      = 30     # Suche im Rückblick-Fenster (Tage)
MAX_BARS_AGO       = 5      # Balance-Area darf max. X Tage in der Vergangenheit enden
VWAP_TREND_PERIOD  = 20     # Tage für übergeordneten VWAP-Trendfilter
OUTPUT_FILE        = "index.html"   # GitHub Pages: immer index.html


# ─────────────────────────────────────────────────────────────────
# UNIVERSUM  — S&P 500 Ticker von Wikipedia; Fallback: kompakte Liste
# ─────────────────────────────────────────────────────────────────
FALLBACK_UNIVERSE = [
    ("AAPL","Information Technology","Apple Inc."),
    ("MSFT","Information Technology","Microsoft Corp."),
    ("NVDA","Information Technology","NVIDIA Corp."),
    ("AVGO","Information Technology","Broadcom Inc."),
    ("ORCL","Information Technology","Oracle Corp."),
    ("AMD","Information Technology","AMD"),
    ("ADBE","Information Technology","Adobe Inc."),
    ("CRM","Information Technology","Salesforce"),
    ("NOW","Information Technology","ServiceNow"),
    ("TXN","Information Technology","Texas Instruments"),
    ("AMZN","Consumer Discretionary","Amazon.com"),
    ("TSLA","Consumer Discretionary","Tesla Inc."),
    ("HD","Consumer Discretionary","Home Depot"),
    ("MCD","Consumer Discretionary","McDonald's"),
    ("NKE","Consumer Discretionary","Nike Inc."),
    ("SBUX","Consumer Discretionary","Starbucks"),
    ("LOW","Consumer Discretionary","Lowe's Cos."),
    ("GOOGL","Communication Services","Alphabet Inc."),
    ("META","Communication Services","Meta Platforms"),
    ("NFLX","Communication Services","Netflix Inc."),
    ("JPM","Financials","JPMorgan Chase"),
    ("V","Financials","Visa Inc."),
    ("MA","Financials","Mastercard"),
    ("BAC","Financials","Bank of America"),
    ("GS","Financials","Goldman Sachs"),
    ("MS","Financials","Morgan Stanley"),
    ("BLK","Financials","BlackRock"),
    ("SPGI","Financials","S&P Global"),
    ("JNJ","Health Care","Johnson & Johnson"),
    ("UNH","Health Care","UnitedHealth Group"),
    ("LLY","Health Care","Eli Lilly"),
    ("MRK","Health Care","Merck & Co."),
    ("ABBV","Health Care","AbbVie Inc."),
    ("PG","Consumer Staples","Procter & Gamble"),
    ("KO","Consumer Staples","Coca-Cola Co."),
    ("PEP","Consumer Staples","PepsiCo Inc."),
    ("WMT","Consumer Staples","Walmart Inc."),
    ("COST","Consumer Staples","Costco Wholesale"),
    ("XOM","Energy","Exxon Mobil"),
    ("CVX","Energy","Chevron Corp."),
    ("COP","Energy","ConocoPhillips"),
    ("NEE","Utilities","NextEra Energy"),
    ("DUK","Utilities","Duke Energy"),
    ("RTX","Industrials","RTX Corp."),
    ("CAT","Industrials","Caterpillar"),
    ("DE","Industrials","Deere & Co."),
    ("GE","Industrials","GE Aerospace"),
    ("UPS","Industrials","United Parcel Service"),
    ("LIN","Materials","Linde plc"),
    ("SHW","Materials","Sherwin-Williams"),
    ("FCX","Materials","Freeport-McMoRan"),
    ("AMT","Real Estate","American Tower"),
    ("PLD","Real Estate","Prologis"),
    ("EQIX","Real Estate","Equinix"),
]


def get_universe():
    """Versucht S&P 500 von Wikipedia zu laden; nutzt Fallback-Liste bei Fehler."""
    try:
        tables  = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", header=0)
        df      = tables[0]
        df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False)
        tickers = df['Symbol'].tolist()
        sectors = dict(zip(df['Symbol'], df['GICS Sector']))
        names   = dict(zip(df['Symbol'], df['Security']))
        print(f"  S&P 500 von Wikipedia geladen: {len(tickers)} Ticker")
        return tickers, sectors, names
    except Exception as e:
        print(f"  Wikipedia nicht erreichbar ({e}) – nutze Fallback-Liste")
        tickers = [d[0] for d in FALLBACK_UNIVERSE]
        sectors = {d[0]: d[1] for d in FALLBACK_UNIVERSE}
        names   = {d[0]: d[2] for d in FALLBACK_UNIVERSE}
        return tickers, sectors, names


# ─────────────────────────────────────────────────────────────────
# TECHNISCHE BERECHNUNGEN
# ─────────────────────────────────────────────────────────────────
def vwap(df):
    """VWAP (Volume Weighted Average Price) für einen DataFrame."""
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    return (tp * df['Volume']).sum() / df['Volume'].sum()


def find_balance_window(df):
    """
    Sucht das beste Balance-Fenster (rolling window) in den letzten LOOKBACK_DAYS Tagen.
    Gibt dict mit Balance-Area-Daten zurück, oder None wenn keine gefunden.
    """
    recent = df.tail(LOOKBACK_DAYS).copy()
    n      = len(recent)
    best   = None
    best_s = -1

    for start in range(n):
        for end in range(start + MIN_BALANCE_DAYS - 1, n):
            bars_ago = n - end - 1
            if bars_ago > MAX_BARS_AGO:
                continue
            window  = recent.iloc[start:end + 1]
            rng_pct = (window['High'].max() - window['Low'].min()) / window['Close'].mean() * 100
            if rng_pct <= MAX_RANGE_PCT:
                days  = end - start + 1
                score = days * (MAX_RANGE_PCT - rng_pct)
                if score > best_s:
                    best_s = score
                    best   = {
                        'df':         window,
                        'range_high': float(window['High'].max()),
                        'range_low':  float(window['Low'].min()),
                        'range_pct':  round(rng_pct, 2),
                        'days':       days,
                        'start_date': window.index[0].strftime('%d.%m.%Y'),
                        'bars_ago':   bars_ago,
                    }
    return best


def classify_setup(price, rh, rl, bias):
    """
    Klassifiziert Setup nach Auktionsmarkt-Prinzipien.
    Gibt (setup_type, priority) zurück – None, None wenn kein Setup.
    """
    dist_h = (rh - price) / rh * 100
    dist_l = (price - rl) / rl * 100
    broke_up   = price > rh
    broke_down = price < rl

    if broke_up and (price - rh) / rh * 100 <= BREAKOUT_MAX_PCT and bias == "Long":
        return "Initiative Buy", 1
    if broke_down and (rl - price) / rl * 100 <= BREAKOUT_MAX_PCT and bias == "Short":
        return "Initiative Sell", 1

    if not broke_up and not broke_down:
        if dist_l <= NEAR_EXTREME_PCT and bias == "Long":
            return "Responsive Buy", 1
        if dist_h <= NEAR_EXTREME_PCT and bias == "Short":
            return "Responsive Sell", 1
        if dist_l <= NEAR_EXTREME_PCT * 2.5 and bias == "Long":
            return "Responsive Buy (Watch)", 2
        if dist_h <= NEAR_EXTREME_PCT * 2.5 and bias == "Short":
            return "Responsive Sell (Watch)", 2

    return None, None


def calculate_score(balance_days, range_pct, priority, trend_aligned, bars_ago):
    """Qualitäts-Score 0–100 für das Setup."""
    s  = (30 if balance_days >= 15 else 24 if balance_days >= 10
          else 17 if balance_days >= 7 else 9)
    s += (28 if range_pct < 3 else 22 if range_pct < 5
          else 14 if range_pct < 7 else 6)
    s += 25 if priority == 1 else 10
    s += 12 if trend_aligned else 0
    s += 5  if bars_ago == 0 else 0
    return min(s, 100)


def analyze_ticker(df, ticker, sector, name):
    """Vollanalyse eines Tickers. Gibt Ergebnis-Dict zurück oder None."""
    if df is None or len(df) < 25:
        return None
    try:
        price   = float(df['Close'].iloc[-1])
        v20     = vwap(df.tail(VWAP_TREND_PERIOD))
        bias    = "Long" if price > v20 else "Short"
        atr14   = float(df['High'].sub(df['Low']).rolling(14).mean().iloc[-1])

        bal     = find_balance_window(df)
        if bal is None:
            return None

        hvn_val = float(vwap(bal['df']))
        setup, prio = classify_setup(price, bal['range_high'], bal['range_low'], bias)
        if setup is None:
            return None

        trend_aligned = (
            ("Buy"  in setup and bias == "Long") or
            ("Sell" in setup and bias == "Short")
        )
        dist_h = round((bal['range_high'] - price) / bal['range_high'] * 100, 2)
        dist_l = round((price - bal['range_low'])  / bal['range_low']  * 100, 2)
        sc     = calculate_score(bal['days'], bal['range_pct'], prio, trend_aligned, bal['bars_ago'])

        return {
            "ticker":         ticker,
            "name":           name,
            "sector":         sector,
            "current_price":  round(price, 2),
            "range_high":     round(bal['range_high'], 2),
            "range_low":      round(bal['range_low'],  2),
            "range_size_pct": bal['range_pct'],
            "hvn_vwap":       round(hvn_val, 2),
            "vwap_trend":     round(v20, 2),
            "trend_bias":     bias,
            "balance_days":   bal['days'],
            "balance_start":  bal['start_date'],
            "bars_ago":       bal['bars_ago'],
            "setup_type":     setup,
            "priority":       prio,
            "dist_from_high": dist_h,
            "dist_from_low":  dist_l,
            "trend_aligned":  trend_aligned,
            "score":          sc,
            "atr14":          round(atr14, 2),
        }
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────
# HTML-DASHBOARD
# ─────────────────────────────────────────────────────────────────
def build_html(results, date_str, screened):
    """Generiert self-contained HTML-Dashboard mit allen Ergebnissen."""

    prio1 = sum(1 for r in results if r['priority'] == 1)
    ib    = sum(1 for r in results if r['setup_type'] == 'Initiative Buy')
    isel  = sum(1 for r in results if r['setup_type'] == 'Initiative Sell')
    rb    = sum(1 for r in results if r['setup_type'] == 'Responsive Buy')
    rs    = sum(1 for r in results if r['setup_type'] == 'Responsive Sell')
    total = len(results)

    json_str = json.dumps(results, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Auktionsmarkt-Screener – {date_str}</title>
<style>
:root{{--bg:#0f1117;--s1:#1a1d27;--s2:#22263a;--bdr:#2e3350;--txt:#e2e8f0;--mut:#8892a4;--acc:#6366f1;--grn:#22c55e;--red:#ef4444;--ylw:#f59e0b}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:var(--bg);color:var(--txt);padding:24px}}
.hdr{{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:22px;gap:12px;flex-wrap:wrap}}
.hdr h1{{font-size:20px;font-weight:700}}.hdr .sub{{font-size:12px;color:var(--mut);margin-top:3px}}
.meta{{font-size:11px;color:var(--mut);text-align:right;line-height:1.7}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:10px;margin-bottom:20px}}
.card{{background:var(--s1);border:1px solid var(--bdr);border-radius:10px;padding:14px;text-align:center;cursor:pointer;transition:all .15s}}
.card:hover{{border-color:var(--acc);transform:translateY(-2px)}}
.card.active{{border-color:var(--acc);background:rgba(99,102,241,.15);box-shadow:0 0 0 1px var(--acc)}}
.card .v{{font-size:26px;font-weight:800}}.card .l{{font-size:10px;color:var(--mut);margin-top:3px;text-transform:uppercase;letter-spacing:.5px}}
.toolbar{{display:flex;gap:7px;flex-wrap:wrap;margin-bottom:13px;align-items:center}}
.btn{{padding:5px 12px;border-radius:16px;border:1px solid var(--bdr);background:var(--s1);color:var(--mut);cursor:pointer;font-size:11px;transition:all .15s}}
.btn:hover{{border-color:var(--acc);color:var(--txt)}}.btn.on{{background:var(--acc);border-color:var(--acc);color:#fff;font-weight:600}}
.search{{background:var(--s1);border:1px solid var(--bdr);border-radius:7px;padding:6px 11px;color:var(--txt);font-size:12px;width:200px;outline:none}}
.search:focus{{border-color:var(--acc)}}.search::placeholder{{color:var(--mut)}}
.sr{{display:flex;align-items:center;gap:7px;font-size:11px;color:var(--mut)}}
.sr input{{accent-color:var(--acc);width:90px}}
.tw{{background:var(--s1);border:1px solid var(--bdr);border-radius:11px;overflow:auto;max-height:65vh}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
thead{{position:sticky;top:0;z-index:5}}
thead th{{background:var(--s2);padding:9px 11px;text-align:left;font-size:10px;font-weight:700;color:var(--mut);text-transform:uppercase;letter-spacing:.4px;cursor:pointer;user-select:none;white-space:nowrap;border-bottom:1px solid var(--bdr)}}
thead th:hover{{color:var(--txt)}}.sorted{{color:var(--acc)!important}}
tbody tr{{border-bottom:1px solid var(--bdr);transition:background .1s}}
tbody tr:last-child{{border-bottom:none}}
tbody tr:hover{{background:var(--s2)}}
td{{padding:8px 11px;vertical-align:middle;white-space:nowrap}}
.tk a{{font-weight:700;font-size:13px;color:var(--acc)}}.tk a:hover{{text-decoration:underline}}
.tk .nm{{font-size:10px;color:var(--mut);margin-top:1px;max-width:115px;overflow:hidden;text-overflow:ellipsis}}
.badge{{display:inline-flex;align-items:center;gap:4px;padding:2px 8px;border-radius:9px;font-size:10px;font-weight:700;white-space:nowrap}}
.ib{{background:rgba(34,197,94,.13);color:#4ade80;border:1px solid rgba(34,197,94,.3)}}
.is{{background:rgba(239,68,68,.13);color:#f87171;border:1px solid rgba(239,68,68,.3)}}
.rb{{background:rgba(59,130,246,.13);color:#60a5fa;border:1px solid rgba(59,130,246,.3)}}
.rs{{background:rgba(168,85,247,.13);color:#c084fc;border:1px solid rgba(168,85,247,.3)}}
.rw{{background:rgba(148,163,184,.07);color:#94a3b8;border:1px solid rgba(148,163,184,.2)}}
.long{{color:var(--grn);font-weight:700}}.short{{color:var(--red);font-weight:700}}
.dot{{width:7px;height:7px;border-radius:50%;display:inline-block;margin-right:4px;flex-shrink:0}}
.p1{{background:var(--grn)}}.p2{{background:var(--ylw)}}
.sbar{{display:flex;align-items:center;gap:5px}}
.str{{width:48px;height:4px;background:var(--bdr);border-radius:2px;overflow:hidden}}
.sfil{{height:100%;border-radius:2px}}
.mb{{min-width:86px}}
.mbl{{display:flex;justify-content:space-between;font-size:9px;color:var(--mut);margin-bottom:2px}}
.mbt{{position:relative;width:86px;height:11px;border-radius:3px;overflow:visible}}
.mbp{{position:absolute;top:-2px;width:3px;height:15px;background:var(--ylw);border-radius:1px;transform:translateX(-50%);z-index:2}}
.mbh{{position:absolute;top:0;width:2px;height:11px;background:var(--acc);opacity:.8;transform:translateX(-50%)}}
.mbpr{{font-size:11px;color:var(--txt);font-weight:600;margin-top:2px}}
.empty{{text-align:center;padding:32px;color:var(--mut)}}
.leg{{display:flex;gap:12px;flex-wrap:wrap;margin-top:9px;font-size:10px;color:var(--mut)}}
.leg span{{display:flex;align-items:center;gap:4px}}
.note{{margin-top:20px;background:var(--s1);border:1px solid var(--bdr);border-radius:10px;padding:15px;font-size:11px;color:var(--mut);line-height:1.7}}
.note strong{{color:var(--txt)}}
.hlp-btn{{margin-left:auto;padding:6px 14px;border-radius:16px;border:1px solid var(--ylw);background:rgba(245,158,11,.08);color:var(--ylw);cursor:pointer;font-size:11px;font-weight:700;transition:all .15s}}
.hlp-btn:hover{{background:rgba(245,158,11,.18)}}
.modal-bg{{display:none;position:fixed;inset:0;background:rgba(0,0,0,.7);z-index:100;overflow-y:auto;padding:24px}}
.modal-bg.open{{display:flex;justify-content:center;align-items:flex-start}}
.modal{{background:#1a1d27;border:1px solid var(--bdr);border-radius:14px;max-width:720px;width:100%;padding:28px;position:relative}}
.modal h2{{font-size:17px;font-weight:800;color:var(--txt);margin-bottom:4px}}
.modal .sub{{font-size:12px;color:var(--mut);margin-bottom:22px}}
.modal-close{{position:absolute;top:16px;right:18px;background:none;border:none;color:var(--mut);font-size:20px;cursor:pointer;line-height:1}}
.modal-close:hover{{color:var(--txt)}}
.step{{display:flex;gap:14px;margin-bottom:20px}}
.step-num{{flex-shrink:0;width:28px;height:28px;border-radius:50%;background:var(--acc);color:#fff;font-weight:800;font-size:13px;display:flex;align-items:center;justify-content:center}}
.step-body h3{{font-size:13px;font-weight:700;color:var(--txt);margin-bottom:5px}}
.step-body p{{font-size:12px;color:#94a3b8;line-height:1.7;margin-bottom:6px}}
.step-body p strong{{color:var(--txt)}}
.step-body .tip{{background:rgba(99,102,241,.1);border-left:3px solid var(--acc);padding:7px 10px;border-radius:0 6px 6px 0;font-size:11px;color:#a5b4fc;margin-top:6px}}
.setup-grid{{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px}}
.sg{{background:var(--s2);border-radius:8px;padding:10px 12px;font-size:11px}}
.sg .st{{font-weight:700;color:var(--txt);margin-bottom:4px}}
.sg .sd{{color:#94a3b8;line-height:1.6}}
.divider{{border:none;border-top:1px solid var(--bdr);margin:18px 0}}
.crv{{background:rgba(34,197,94,.08);border:1px solid rgba(34,197,94,.25);border-radius:8px;padding:10px 14px;font-size:12px;color:#4ade80;margin-top:6px}}
.crv strong{{color:#86efac}}
</style>
</head>
<body>
<div class="hdr">
  <div><h1>Auktionsmarkt-Screener</h1>
  <div class="sub">Tom Alexander – Trading Without Crutches &nbsp;|&nbsp; Balance-Areas · KRAs · Initiative &amp; Responsive Setups &nbsp;|&nbsp; S&amp;P 500</div></div>
  <div class="meta">Generiert: <strong>{date_str}</strong><br>Universe: {screened} Werte &nbsp;|&nbsp; Setups: {total}</div>
</div>
<div class="cards">
  <div class="card active" data-f="all" onclick="filtCard('all',this)"><div class="v" style="color:var(--acc)">{total}</div><div class="l">Alle Setups</div></div>
  <div class="card" data-f="prio1" onclick="filtCard('prio1',this)"><div class="v" style="color:var(--grn)">{prio1}</div><div class="l">⭐ Prio 1</div></div>
  <div class="card" data-f="Initiative Buy" onclick="filtCard('Initiative Buy',this)"><div class="v" style="color:#4ade80">{ib}</div><div class="l">Init. Buy</div></div>
  <div class="card" data-f="Initiative Sell" onclick="filtCard('Initiative Sell',this)"><div class="v" style="color:#f87171">{isel}</div><div class="l">Init. Sell</div></div>
  <div class="card" data-f="Responsive Buy" onclick="filtCard('Responsive Buy',this)"><div class="v" style="color:#60a5fa">{rb}</div><div class="l">Resp. Buy</div></div>
  <div class="card" data-f="Responsive Sell" onclick="filtCard('Responsive Sell',this)"><div class="v" style="color:#c084fc">{rs}</div><div class="l">Resp. Sell</div></div>
</div>
<div class="toolbar">
  <input class="search" type="text" id="si" placeholder="Ticker, Name, Sektor…" oninput="render()">
  <button class="btn on" onclick="filt('all',this)">Alle</button>
  <button class="btn" onclick="filt('Initiative Buy',this)">🟢 Init. Buy</button>
  <button class="btn" onclick="filt('Initiative Sell',this)">🔴 Init. Sell</button>
  <button class="btn" onclick="filt('Responsive Buy',this)">🔵 Resp. Buy</button>
  <button class="btn" onclick="filt('Responsive Sell',this)">🟣 Resp. Sell</button>
  <button class="btn" onclick="filt('Watch',this)">👁 Watch</button>
  <button class="btn" onclick="filt('Long',this)">↑ Long</button>
  <button class="btn" onclick="filt('Short',this)">↓ Short</button>
  <div class="sr">Score≥<input type="range" id="sm" min="0" max="100" step="5" value="0" oninput="document.getElementById('sv').textContent=this.value;render()"><span id="sv">0</span></div>
  <button class="hlp-btn" onclick="document.getElementById('hlp').classList.add('open')">📖 Wie benutze ich das?</button>
</div>
<div class="tw">
<table>
<thead><tr>
  <th onclick="srt(0)" id="h0">Ticker</th>
  <th onclick="srt(1)" id="h1">Sektor</th>
  <th onclick="srt(2)" id="h2">Setup</th>
  <th onclick="srt(3)" id="h3" class="sorted">Score ↓</th>
  <th onclick="srt(4)" id="h4">Balance-Tage</th>
  <th>Preis &amp; Range</th>
  <th onclick="srt(6)" id="h6">Range-%</th>
  <th onclick="srt(7)" id="h7">Trend</th>
  <th onclick="srt(8)" id="h8">Abstand</th>
</tr></thead>
<tbody id="tb"></tbody>
</table>
<div class="empty" id="em" style="display:none">Kein Setup für diesen Filter.</div>
</div>
<div class="leg">
  <span><span style="display:inline-block;width:11px;height:3px;background:var(--ylw);border-radius:1px"></span>Aktueller Preis</span>
  <span><span style="display:inline-block;width:11px;height:3px;background:var(--acc);border-radius:1px"></span>HVN (VWAP Balance-Area)</span>
  <span><span class="dot p1"></span>Prio 1 – handelbares Setup</span>
  <span><span class="dot p2"></span>Prio 2 – beobachten</span>
</div>
<div id="hlp" class="modal-bg" onclick="if(event.target===this)this.classList.remove('open')">
<div class="modal">
  <button class="modal-close" onclick="document.getElementById('hlp').classList.remove('open')">✕</button>
  <h2>📖 Schritt-für-Schritt-Anleitung</h2>
  <div class="sub">Auktionsmarkt-Screener nach Tom Alexander – für Einsteiger erklärt</div>

  <div class="step">
    <div class="step-num">1</div>
    <div class="step-body">
      <h3>Dashboard filtern – nur die besten Kandidaten sehen</h3>
      <p>Klicke oben auf die Kachel <strong>„Prio 1"</strong> – damit siehst du nur Aktien, bei denen gerade ein konkretes Setup aktiv ist. Prio-2-Einträge (Watch) sind noch nicht handelbar, nur beobachtenswert.</p>
      <p>Schiebe den <strong>Score-Regler</strong> auf mindestens <strong>65</strong>. Alles darunter ist zwar rechnerisch ein Setup, aber die Qualität ist zu niedrig für einen Trade.</p>
      <div class="tip">✅ Deine Ausgangsliste: Prio 1 + Score ≥ 65. Das sind deine Kandidaten für heute.</div>
    </div>
  </div>

  <div class="step">
    <div class="step-num">2</div>
    <div class="step-body">
      <h3>Setup-Typ verstehen – was bedeuten die Farben?</h3>
      <div class="setup-grid">
        <div class="sg"><div class="st" style="color:#60a5fa">◀ Responsive Buy</div><div class="sd">Der Preis ist nahe am <strong>unteren</strong> Rand der Konsolidierungszone. Du kaufst, weil der Markt diesen Bereich bereits respektiert hat. Trend: Long (▲).</div></div>
        <div class="sg"><div class="st" style="color:#c084fc">▶ Responsive Sell</div><div class="sd">Der Preis ist nahe am <strong>oberen</strong> Rand. Du verkaufst/shortest, weil der Markt von oben abgeprallt ist. Trend: Short (▼).</div></div>
        <div class="sg"><div class="st" style="color:#4ade80">▲ Initiative Buy</div><div class="sd">Der Preis hat gerade die Konsolidierung nach <strong>oben durchbrochen</strong>. Ein frischer Ausbruch – du kaufst den Breakout. Trend muss Long sein.</div></div>
        <div class="sg"><div class="st" style="color:#f87171">▼ Initiative Sell</div><div class="sd">Der Preis hat die Zone nach <strong>unten verlassen</strong>. Ein Breakdown – du shortest. Trend muss Short sein.</div></div>
      </div>
      <div class="tip">💡 Einsteiger-Empfehlung: Fange mit <strong>Responsive Buy/Sell</strong> an. Du fadest die Extreme einer bekannten Zone – das ist klarer und kontrollierbarer als Breakout-Trading.</div>
    </div>
  </div>

  <div class="step">
    <div class="step-num">3</div>
    <div class="step-body">
      <h3>Chart öffnen und Setup bestätigen – PFLICHT vor jedem Trade</h3>
      <p>Klicke auf den <strong>Ticker-Link</strong> in der Tabelle – er öffnet TradingView direkt. Stelle den Chart auf <strong>Tageskerzen</strong> ein.</p>
      <p>Was du prüfst: <strong>(a)</strong> Siehst du die Konsolidierungszone deutlich als flachen Bereich? <strong>(b)</strong> Werden die Volumenbалken innerhalb der Zone kleiner (nachlassendes Interesse = gesunde Balance)? <strong>(c)</strong> Gibt es eine Kerzenstruktur am Extrem – z.B. einen Hammer oder einen Doji – als Bestätigungssignal?</p>
      <div class="tip">⚠️ Kein Chart-Signal = kein Trade. Der Screener findet Kandidaten, der Chart entscheidet.</div>
    </div>
  </div>

  <div class="step">
    <div class="step-num">4</div>
    <div class="step-body">
      <h3>Einstieg – wo genau kaufe/shorte ich?</h3>
      <p><strong>Responsive Buy:</strong> Einstieg am oder leicht oberhalb des unteren Extrems der Balance-Area (der „Range Low"-Wert in der Tabelle). Ideal: auf einer bullischen Kerze am Low (Hammer, bullischer Engulfing). Kein Vorausgreifen – warte auf die Bestätigungskerze.</p>
      <p><strong>Responsive Sell:</strong> Einstieg am oder leicht unterhalb des oberen Extrems (Range High). Auf einer bearischen Kerze am High (bärischer Engulfing, Shooting Star).</p>
      <p><strong>Initiative Buy:</strong> Nicht sofort beim Ausbruch einsteigen. Warte auf den ersten <strong>Pullback</strong> zurück zur gebrochenen Obergrenze (die wird zur Unterstützung). Dort einsteigen.</p>
      <p><strong>Initiative Sell:</strong> Warte auf den Pullback zurück zur gebrochenen Untergrenze (die wird zum Widerstand). Dort short gehen.</p>
    </div>
  </div>

  <div class="step">
    <div class="step-num">5</div>
    <div class="step-body">
      <h3>Stop Loss – wo ist mein maximales Risiko?</h3>
      <p><strong>Responsive Buy:</strong> Stop Loss <strong>knapp unterhalb des Range Low</strong>. Faustregel: 0,5–1× ATR14 (der ATR-Wert steht in der Tabelle) unter dem Tief der Balance-Area. Wenn der Markt darunter schließt, war die Annahme falsch.</p>
      <p><strong>Responsive Sell:</strong> Stop Loss knapp oberhalb des Range High. 0,5–1× ATR14 über dem Hoch.</p>
      <p><strong>Initiative Buy:</strong> Stop Loss zurück <strong>in die Balance-Area hinein</strong> – z.B. auf halbe Range-Breite. Wenn der Ausbruch scheitert, ist das Setup hinfällig.</p>
      <p><strong>Initiative Sell:</strong> Stop Loss zurück in die Balance-Area – auf halbe Range-Breite nach oben.</p>
      <div class="tip">📐 Den ATR14-Wert findest du in der Tabelle-Spalte – oder direkt im TradingView-Chart als Indikator „ATR(14)".</div>
    </div>
  </div>

  <div class="step">
    <div class="step-num">6</div>
    <div class="step-body">
      <h3>Kursziel – wo nehme ich Gewinne mit?</h3>
      <p><strong>Responsive Buy:</strong> Erstes Ziel = der <strong>HVN/VWAP-Wert</strong> aus der Tabelle (goldene Linie im Mini-Chart, das Gravitationszentrum der Balance-Area). Zweites Ziel = Range High.</p>
      <p><strong>Responsive Sell:</strong> Erstes Ziel = HVN/VWAP. Zweites Ziel = Range Low.</p>
      <p><strong>Initiative Buy/Sell:</strong> Projiziere die <strong>Breite der Balance-Area</strong> in Richtung des Ausbruchs. Formel: Range-Breite = Range High minus Range Low. Addiere diese Zahl zum Ausbruchspunkt (bei Buy) oder subtrahiere sie (bei Sell).</p>
      <div class="crv"><strong>Wichtigste Regel:</strong> Führe den Trade nur aus, wenn das Chance-Risiko-Verhältnis (CRV) mindestens <strong>2:1</strong> beträgt – also das Kursziel mindestens doppelt so weit entfernt ist wie der Stop Loss. Alles darunter lohnt sich statistisch nicht.</div>
    </div>
  </div>

  <hr class="divider">
  <p style="font-size:11px;color:var(--mut);line-height:1.7"><strong style="color:var(--txt)">Kurzfassung für den täglichen Ablauf:</strong> Prio 1 anklicken → Score ≥ 65 → Top 5–8 Kandidaten → Chart öffnen → Balance-Area sichtbar? → Kerzenstruktur am Extrem? → Einstieg, Stop, Ziel berechnen → CRV ≥ 2:1? → Trade ausführen oder streichen.<br><br>
  Keine Anlageberatung. Alle Signale eigenverantwortlich prüfen.</p>
</div>
</div>

<div class="note">
  <strong>Methodik:</strong> Balance-Area = Konsolidierungsfenster (min. {MIN_BALANCE_DAYS} Tage, Range &lt;{MAX_RANGE_PCT:.0f}%, max. {MAX_BARS_AGO} Tage alt).
  HVN-Näherung = VWAP der Balance-Area (kein echtes Volumenprofil — für präzise HVNs Volumenprofil-Tool nutzen).
  Trendfilter = Preis vs. {VWAP_TREND_PERIOD}d-VWAP. Initiative-Ausbruch = Schlusskurs außerhalb Extrem ≤{BREAKOUT_MAX_PCT:.0f}%.
  Responsive = Preis innerhalb {NEAR_EXTREME_PCT:.0f}% vom Extrem, Watch = innerhalb {NEAR_EXTREME_PCT*2.5:.0f}%.<br>
  <strong>Kein Anlageberatung.</strong> Jedes Signal manuell im Tages- und Wochenchart prüfen. Klick auf Ticker → TradingView.
</div>
<script>
const D={json_str};
const BC={{"Initiative Buy":"ib","Initiative Sell":"is","Responsive Buy":"rb","Responsive Sell":"rs","Responsive Buy (Watch)":"rw","Responsive Sell (Watch)":"rw"}};
const IC={{"Initiative Buy":"▲","Initiative Sell":"▼","Responsive Buy":"◀","Responsive Sell":"▶","Responsive Buy (Watch)":"◁","Responsive Sell (Watch)":"▷"}};
let cf="all",sc=3,sa=false;
function filtCard(f,card){{cf=f;document.querySelectorAll(".card[data-f]").forEach(c=>c.classList.remove("active"));card.classList.add("active");document.querySelectorAll(".btn").forEach(b=>b.classList.remove("on"));render();}}
function sc2(s){{return s>=75?"#22c55e":s>=55?"#f59e0b":"#ef4444"}}
function mb(r){{
  const t=r.range_high-r.range_low;if(t<=0)return "";
  const pp=Math.min(Math.max((r.current_price-r.range_low)/t,0),1)*100;
  const hp=Math.min(Math.max((r.hvn_vwap-r.range_low)/t,0),1)*100;
  const ext=r.current_price>r.range_high||r.current_price<r.range_low;
  return `<div class="mb"><div class="mbl"><span>${{r.range_low.toFixed(1)}}</span><span>${{r.range_high.toFixed(1)}}</span></div>
    <div class="mbt" style="background:${{ext?"rgba(245,158,11,.1)":"rgba(99,102,241,.1)"}}">
      <div class="mbp" style="left:${{Math.min(Math.max(pp,2),98)}}%"></div>
      <div class="mbh" style="left:${{Math.min(Math.max(hp,2),98)}}%"></div>
    </div><div class="mbpr">$${{r.current_price.toFixed(2)}}</div></div>`;}}
function dc(r){{return r.setup_type.includes("Buy")?`<span style="color:#60a5fa">${{r.dist_from_low.toFixed(1)}}% vom Low</span>`:`<span style="color:#c084fc">${{r.dist_from_high.toFixed(1)}}% vom High</span>`;}}
function gd(){{
  const q=document.getElementById("si").value.toLowerCase();
  const ms=parseInt(document.getElementById("sm").value)||0;
  return D.filter(r=>{{
    const mq=!q||r.ticker.toLowerCase().includes(q)||r.name.toLowerCase().includes(q)||r.sector.toLowerCase().includes(q);
    const mf=cf==="all"||(cf==="prio1"&&r.priority===1)||(cf==="Watch"&&r.setup_type.includes("Watch"))||(cf==="Long"&&r.trend_bias==="Long")||(cf==="Short"&&r.trend_bias==="Short")||r.setup_type===cf||r.setup_type===cf+" (Watch)";
    return mq&&mf&&r.score>=ms;}});}}
function sd(data){{
  const fn=[r=>r.ticker,r=>r.sector,r=>r.setup_type,r=>r.score,r=>r.balance_days,r=>r.current_price,r=>r.range_size_pct,r=>r.trend_bias,r=>r.setup_type.includes("Buy")?r.dist_from_low:r.dist_from_high][sc]||(r=>r.score);
  return [...data].sort((a,b)=>{{const va=fn(a),vb=fn(b);return va<vb?(sa?1:-1):va>vb?(sa?-1:1):0;}});}}
function render(){{
  const rows=sd(gd());const tb=document.getElementById("tb");const em=document.getElementById("em");
  if(!rows.length){{tb.innerHTML="";em.style.display="block";return;}}em.style.display="none";
  tb.innerHTML=rows.map(r=>`<tr>
    <td class="tk"><a href="https://www.tradingview.com/chart/?symbol=${{r.ticker}}" target="_blank">${{r.ticker}}</a><div class="nm">${{r.name}}</div></td>
    <td style="color:var(--mut);font-size:11px">${{r.sector}}</td>
    <td><span class="dot ${{r.priority===1?'p1':'p2'}}"></span><span class="badge ${{BC[r.setup_type]||'rw'}}">${{IC[r.setup_type]||''}} ${{r.setup_type}}</span></td>
    <td><div class="sbar"><span style="font-weight:800;color:${{sc2(r.score)}};min-width:25px">${{r.score}}</span><div class="str"><div class="sfil" style="width:${{r.score}}%;background:${{sc2(r.score)}}"></div></div></div></td>
    <td style="text-align:center"><strong>${{r.balance_days}}</strong><div style="font-size:9px;color:var(--mut)">ab ${{r.balance_start}}</div></td>
    <td>${{mb(r)}}</td>
    <td style="text-align:center;color:${{r.range_size_pct<5?"#22c55e":r.range_size_pct<7?"#f59e0b":"#ef4444"}}">${{r.range_size_pct.toFixed(1)}}%</td>
    <td class="${{r.trend_bias==='Long'?'long':'short'}}">${{r.trend_bias==='Long'?'▲ Long':'▼ Short'}}</td>
    <td>${{dc(r)}}</td></tr>`).join("");}}
function filt(f,btn){{cf=f;document.querySelectorAll(".btn").forEach(b=>b.classList.remove("on"));btn.classList.add("on");document.querySelectorAll(".card[data-f]").forEach(c=>c.classList.remove("active"));render();}}
function srt(col){{if(sc===col){{sa=!sa;}}else{{sc=col;sa=false;}}
  for(let i=0;i<=8;i++){{const h=document.getElementById("h"+i);if(h)h.classList.toggle("sorted",i===col);}}render();}}
render();
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    print("\n═══════════════════════════════════════════════════════")
    print("  Auktionsmarkt-Screener v2  –  GitHub Actions Build")
    print("  Tom Alexander – Trading Without Crutches (2015)")
    print("═══════════════════════════════════════════════════════\n")

    print("① Universe laden ...")
    tickers, sectors, names = get_universe()

    print(f"② Kursdaten laden ({len(tickers)} Ticker, Batch-Download) ...")
    raw = yf.download(
        tickers, period="3mo", auto_adjust=True,
        progress=False, group_by="ticker", threads=True
    )
    if raw is None or raw.empty:
        print("  !! Fehler beim Download. Abbruch.")
        sys.exit(1)

    print("③ Balance-Areas analysieren und Setups klassifizieren ...")
    results = []
    for i, ticker in enumerate(tickers):
        if (i + 1) % 100 == 0:
            print(f"  ... {i+1}/{len(tickers)}")
        try:
            df = raw[ticker].dropna() if len(tickers) > 1 else raw.dropna()
            r  = analyze_ticker(df, ticker, sectors.get(ticker, "—"), names.get(ticker, ticker))
            if r:
                results.append(r)
        except Exception:
            pass

    results.sort(key=lambda x: x['score'], reverse=True)

    prio1 = sum(1 for r in results if r['priority'] == 1)
    print(f"\n  Analysiert : {len(tickers)} Ticker")
    print(f"  Setups     : {len(results)} gefunden  |  Prio 1: {prio1}")
    for r in results[:10]:
        print(f"  {r['ticker']:6s} | {r['setup_type']:28s} | Score {r['score']:3d} | {r['balance_days']:2d}d | {r['trend_bias']}")

    print(f"\n④ HTML-Dashboard generieren → {OUTPUT_FILE} ...")
    date_str = datetime.now().strftime("%d.%m.%Y %H:%M UTC")
    html     = build_html(results, date_str, len(tickers))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✓ Fertig: {OUTPUT_FILE}  ({len(html)//1024} KB)\n")


if __name__ == "__main__":
    main()
