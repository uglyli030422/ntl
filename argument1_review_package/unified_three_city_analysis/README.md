# Unified Three-City NTL Electricity Bias Analysis

This directory contains an isolated, configuration-driven analysis for Tokyo,
Amsterdam, and London. It does not modify prior data, scripts, or results.

Run from the project root:

```powershell
python unified_three_city_analysis/scripts/unified_three_city_analysis.py
```

Main outputs are written under:

- `outputs/tokyo`
- `outputs/amsterdam`
- `outputs/london`
- `outputs/cross_city`
- `audit`

The pipeline fits one city-specific RandomForestRegressor using only the NTL
field, calculates out-of-fold predictions, derives multiple bias metrics, tests
vulnerability gradients, summarizes social indicators and urban form effects,
and evaluates low-NTL detection of socially vulnerable low-electricity areas.
