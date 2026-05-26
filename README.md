# Auktionsmarkt-Screener

Täglicher Balance-Area-Screener für S&P 500-Aktien nach der Auktionsmarkt-Theorie
von Tom Alexander (*Trading Without Crutches*, 2015).

**Live-Dashboard:** https://thblees.github.io/auktionsmarkt-screener/

---

## Was der Screener macht

- Scannt täglich ~500 S&P 500-Aktien auf aktive Balance Areas (Konsolidierungsphasen)
- Klassifiziert Setups: **Responsive Buy/Sell** (Fade der Range-Extreme) und **Initiative Buy/Sell** (Ausbrüche)
- Berechnet einen Qualitäts-Score (0–100) aus Balance-Dauer, Range-Enge, Priorität und Trend-Alignment
- Trendfilter: Preis vs. 20-Tage-VWAP (Long / Short Bias)
- HVN-Näherung: VWAP der Balance-Area (kein echtes Volumenprofil)

## Setup-Typen

| Setup | Beschreibung | Prio |
|-------|-------------|------|
| Responsive Buy | Preis nahe unterem Range-Extrem, Long-Bias | 1 |
| Responsive Sell | Preis nahe oberem Range-Extrem, Short-Bias | 1 |
| Initiative Buy | Frischer Ausbruch nach oben ≤5%, Long-Bias | 1 |
| Initiative Sell | Frischer Ausbruch nach unten ≤5%, Short-Bias | 1 |
| ...Watch | Setup nähert sich, noch nicht aktiv | 2 |

## Automatische Updates

Der GitHub Actions Workflow `.github/workflows/daily-update.yml` läuft
**montags bis freitags um 21:15 UTC** (kurz nach US-Marktschluss 17:00 ET).

Er führt `screener.py` aus, generiert `index.html` und committet sie zurück ins Repo.
GitHub Pages serviert die Datei automatisch.

Manueller Start: GitHub → Actions → "Daily Screener Update" → "Run workflow"

## Lokale Ausführung

```bash
pip install yfinance pandas requests
python screener.py
# Öffne dann index.html im Browser
```

## Parameter (in screener.py anpassen)

```python
MIN_BALANCE_DAYS  = 4     # Mindest-Konsolidierungstage
MAX_RANGE_PCT     = 9.0   # Max. Range der Balance-Area in %
NEAR_EXTREME_PCT  = 3.0   # "Nahe am Extrem" Schwelle in %
BREAKOUT_MAX_PCT  = 5.0   # Maximale Ausbruchs-Distanz in %
MAX_BARS_AGO      = 5     # Balance-Area darf max. X Tage alt sein
VWAP_TREND_PERIOD = 20    # Periode für Trendfilter-VWAP
```

---

*Keine Anlageberatung. Alle Signale manuell im Chart prüfen.*
