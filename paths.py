"""
Central path configuration for the CaRS project.

All project paths are derived from CARS_ROOT, which defaults to the directory
containing this file. Override by setting the CARS_ROOT environment variable.
"""

import os
from pathlib import Path

CARS_ROOT = Path(os.environ.get("CARS_ROOT", Path(__file__).resolve().parent))

# Data directories
DATA_DIR = CARS_ROOT / "data"
UNIFIED_DIR = DATA_DIR / "unified"
ENTSOE_DIR = DATA_DIR / "entsoe"
WEATHER_DIR = DATA_DIR / "weather"
COMMODITY_DIR = DATA_DIR / "commodities"
EPFTOOLBOX_DIR = DATA_DIR / "epftoolbox"
GAS_STORAGE_DIR = DATA_DIR / "gas_storage"
MACRO_DIR = DATA_DIR / "macro"
SENTIMENT_DIR = DATA_DIR / "sentiment"
OIL_FUND_DIR = DATA_DIR / "oil_fundamentals"
TRANSPORT_DIR = DATA_DIR / "transport"
TRADE_DIR = DATA_DIR / "trade"
HYDROGEN_DIR = DATA_DIR / "hydrogen"
OUTAGE_DIR = DATA_DIR / "outages"
HYDRO_RESERVOIR_DIR = DATA_DIR / "hydro_reservoirs"
GEN_FORECAST_DIR = DATA_DIR / "generation_forecasts"
DEMAND_FORECAST_DIR = DATA_DIR / "demand_forecasts"
NTC_DIR = DATA_DIR / "ntc"
ENTSOG_DIR = DATA_DIR / "entsog"
BALANCING_DIR = DATA_DIR / "balancing"
ERA5_DIR = DATA_DIR / "era5"

# Upstream benchmark repos
UPSTREAM_DIR = CARS_ROOT / "upstream"
DS3M_DIR = UPSTREAM_DIR / "DS3M"
CASTOR_DIR = UPSTREAM_DIR / "CASTOR"
FANTOM_CODE_DIR = UPSTREAM_DIR / "FANTOM" / "FANTOM_supplementary" / "fantom_code"

# Electricity module directories
ELECTRICITY_DIR = CARS_ROOT / "electricity"
SUBMISSIONS_DIR = ELECTRICITY_DIR / "submissions"

# Output directories
OUTPUT_DIR = CARS_ROOT / "outputs"
LOG_DIR = CARS_ROOT / "logs"
