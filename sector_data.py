"""
Sector Data & Heatmap Engine
Implements:
1. Definedge Platform Sectoral Classification for Indian Equities.
2. Standard NSE Sectoral Indices mapping.
3. Fast vectorized multi-timeframe performance calculation (Daily 1D, Weekly 1W, Monthly 1M, Yearly 1Y).
"""

import sqlite3
import time
import pandas as pd
from typing import Dict, List, Optional, Tuple
from config import DB_PATH

# ---------------------------------------------------------------------------
# Definedge Sector Taxonomy & Curated Stock Mappings
# ---------------------------------------------------------------------------

DEFINEDGE_SECTORS_LIST = [
    "Automobiles & Auto Ancillaries",
    "Banking - Private",
    "Banking - PSU",
    "Financial Services & NBFC",
    "Information Technology",
    "Pharmaceuticals & Healthcare",
    "Fast Moving Consumer Goods (FMCG)",
    "Consumer Durables & Electronics",
    "Metals & Mining",
    "Oil, Gas & Petroleum",
    "Power & Renewable Energy",
    "Chemicals & Petrochemicals",
    "Real Estate & Realty",
    "Capital Goods & Industrial Engineering",
    "Cement & Building Materials",
    "Media, Telecom & Entertainment",
    "Textiles & Apparel",
    "Logistics, Shipping & Aviation",
    "Defense & Aerospace",
    "Hotels, Tourism & Leisure",
    "Agriculture, Fertilizers & Sugar",
    "Retail & E-Commerce",
    "Conglomerates & Diversified"
]

# Curated High-Accuracy Symbol-to-Sector Mappings for NSE Universe
SYMBOL_TO_DEFINEDGE_SECTOR = {
    # Automobiles & Auto Ancillaries
    "MARUTI": "Automobiles & Auto Ancillaries",
    "TATAMOTORS": "Automobiles & Auto Ancillaries",
    "M&M": "Automobiles & Auto Ancillaries",
    "BAJAJ-AUTO": "Automobiles & Auto Ancillaries",
    "EICHERMOT": "Automobiles & Auto Ancillaries",
    "HEROMOTOCO": "Automobiles & Auto Ancillaries",
    "TVSMOTOR": "Automobiles & Auto Ancillaries",
    "BHARATFORG": "Automobiles & Auto Ancillaries",
    "BOSCHLTD": "Automobiles & Auto Ancillaries",
    "MOTHERSON": "Automobiles & Auto Ancillaries",
    "SAMVARDHANA": "Automobiles & Auto Ancillaries",
    "MRF": "Automobiles & Auto Ancillaries",
    "BALKRISIND": "Automobiles & Auto Ancillaries",
    "APOLLOTYRE": "Automobiles & Auto Ancillaries",
    "SONACOMS": "Automobiles & Auto Ancillaries",
    "EXIDEIND": "Automobiles & Auto Ancillaries",
    "AMARAJABAT": "Automobiles & Auto Ancillaries",
    "ARE&M": "Automobiles & Auto Ancillaries",
    "TIINDIA": "Automobiles & Auto Ancillaries",
    "ENDURANCE": "Automobiles & Auto Ancillaries",
    "ESCORTS": "Automobiles & Auto Ancillaries",
    "ASHOKLEY": "Automobiles & Auto Ancillaries",
    "CEATLTD": "Automobiles & Auto Ancillaries",
    "CRAFTSMAN": "Automobiles & Auto Ancillaries",
    "GABRIEL": "Automobiles & Auto Ancillaries",
    "SUNDRMFAST": "Automobiles & Auto Ancillaries",
    "SUBROS": "Automobiles & Auto Ancillaries",
    "UNOMINDA": "Automobiles & Auto Ancillaries",

    # Banking - Private
    "HDFCBANK": "Banking - Private",
    "ICICIBANK": "Banking - Private",
    "KOTAKBANK": "Banking - Private",
    "AXISBANK": "Banking - Private",
    "INDUSINDBK": "Banking - Private",
    "FEDERALBNK": "Banking - Private",
    "IDFCFIRSTB": "Banking - Private",
    "AUBANK": "Banking - Private",
    "BANDHANBNK": "Banking - Private",
    "RBLBANK": "Banking - Private",
    "YESBANK": "Banking - Private",
    "CITYUNIONB": "Banking - Private",
    "KARURVYSYA": "Banking - Private",
    "CUB": "Banking - Private",
    "CSBBANK": "Banking - Private",
    "SOUTHBANK": "Banking - Private",
    "DCBBANK": "Banking - Private",
    "KTKBANK": "Banking - Private",
    "J&KBANK": "Banking - Private",

    # Banking - PSU
    "SBIN": "Banking - PSU",
    "BANKBARODA": "Banking - PSU",
    "PNB": "Banking - PSU",
    "CANBK": "Banking - PSU",
    "UNIONBANK": "Banking - PSU",
    "INDIANB": "Banking - PSU",
    "IOB": "Banking - PSU",
    "BANKINDIA": "Banking - PSU",
    "CENTRALBK": "Banking - PSU",
    "MAHABANK": "Banking - PSU",
    "UCOBANK": "Banking - PSU",
    "PSB": "Banking - PSU",

    # Financial Services & NBFC
    "BAJFINANCE": "Financial Services & NBFC",
    "BAJAJFINSV": "Financial Services & NBFC",
    "SHRIRAMFIN": "Financial Services & NBFC",
    "CHOLAFIN": "Financial Services & NBFC",
    "MUTHOOTFIN": "Financial Services & NBFC",
    "JIOFIN": "Financial Services & NBFC",
    "SBICARD": "Financial Services & NBFC",
    "HDFCLIFE": "Financial Services & NBFC",
    "SBILIFE": "Financial Services & NBFC",
    "ICICIPRULI": "Financial Services & NBFC",
    "ICICIGI": "Financial Services & NBFC",
    "PFC": "Financial Services & NBFC",
    "RECLTD": "Financial Services & NBFC",
    "L&TFH": "Financial Services & NBFC",
    "M&MFIN": "Financial Services & NBFC",
    "POONAWALLA": "Financial Services & NBFC",
    "CREDITACC": "Financial Services & NBFC",
    "MANAPPURAM": "Financial Services & NBFC",
    "HDFCAMC": "Financial Services & NBFC",
    "NAM-INDIA": "Financial Services & NBFC",
    "UTIAMC": "Financial Services & NBFC",
    "BSE": "Financial Services & NBFC",
    "MCX": "Financial Services & NBFC",
    "CDSL": "Financial Services & NBFC",
    "CAMS": "Financial Services & NBFC",
    "KFINTECH": "Financial Services & NBFC",
    "ANGELONE": "Financial Services & NBFC",
    "MOTILALOFS": "Financial Services & NBFC",
    "ISEC": "Financial Services & NBFC",
    "LICHSGFIN": "Financial Services & NBFC",
    "PNBHOUSING": "Financial Services & NBFC",
    "CANFINHOME": "Financial Services & NBFC",
    "AAVAS": "Financial Services & NBFC",
    "HOMEFIRST": "Financial Services & NBFC",
    "HUDCO": "Financial Services & NBFC",
    "IREDA": "Financial Services & NBFC",
    "LICI": "Financial Services & NBFC",

    # Information Technology
    "TCS": "Information Technology",
    "INFY": "Information Technology",
    "HCLTECH": "Information Technology",
    "WIPRO": "Information Technology",
    "LTIM": "Information Technology",
    "TECHM": "Information Technology",
    "PERSISTENT": "Information Technology",
    "COFORGE": "Information Technology",
    "MPHASIS": "Information Technology",
    "KPITTECH": "Information Technology",
    "TATAELXSI": "Information Technology",
    "LTTS": "Information Technology",
    "CYIENT": "Information Technology",
    "SONATSOFTW": "Information Technology",
    "BSOFT": "Information Technology",
    "ZENSARTECH": "Information Technology",
    "HAPPSTMNDS": "Information Technology",
    "MASTEK": "Information Technology",
    "INTELLECT": "Information Technology",
    "NEWGEN": "Information Technology",
    "LATENTVIEW": "Information Technology",
    "MAPMYINDIA": "Information Technology",
    "RATEGAIN": "Information Technology",
    "TANLA": "Information Technology",
    "ROUTE": "Information Technology",
    "OFSS": "Information Technology",
    "FSL": "Information Technology",
    "ECLERX": "Information Technology",
    "SASKEN": "Information Technology",
    "AFFLE": "Information Technology",

    # Pharmaceuticals & Healthcare
    "SUNPHARMA": "Pharmaceuticals & Healthcare",
    "CIPLA": "Pharmaceuticals & Healthcare",
    "DRREDDY": "Pharmaceuticals & Healthcare",
    "DIVISLAB": "Pharmaceuticals & Healthcare",
    "APOLLOHOSP": "Pharmaceuticals & Healthcare",
    "LUPIN": "Pharmaceuticals & Healthcare",
    "TORNTPHARM": "Pharmaceuticals & Healthcare",
    "ZYDUSLIFE": "Pharmaceuticals & Healthcare",
    "MANKIND": "Pharmaceuticals & Healthcare",
    "MAXHEALTH": "Pharmaceuticals & Healthcare",
    "AUROPHARMA": "Pharmaceuticals & Healthcare",
    "ALKEM": "Pharmaceuticals & Healthcare",
    "FORTIS": "Pharmaceuticals & Healthcare",
    "BIOCON": "Pharmaceuticals & Healthcare",
    "GLENMARK": "Pharmaceuticals & Healthcare",
    "AJANTPHARM": "Pharmaceuticals & Healthcare",
    "IPCALAB": "Pharmaceuticals & Healthcare",
    "LAURUSLABS": "Pharmaceuticals & Healthcare",
    "SYNGENE": "Pharmaceuticals & Healthcare",
    "MEDANTA": "Pharmaceuticals & Healthcare",
    "GLAXO": "Pharmaceuticals & Healthcare",
    "SANOFI": "Pharmaceuticals & Healthcare",
    "ABBOTINDIA": "Pharmaceuticals & Healthcare",
    "PFIZER": "Pharmaceuticals & Healthcare",
    "KIMS": "Pharmaceuticals & Healthcare",
    "RAINBOW": "Pharmaceuticals & Healthcare",
    "METROPOLIS": "Pharmaceuticals & Healthcare",
    "LALPATHLAB": "Pharmaceuticals & Healthcare",
    "VIJAYA": "Pharmaceuticals & Healthcare",
    "JBCHEPHARM": "Pharmaceuticals & Healthcare",
    "NATCOPHARM": "Pharmaceuticals & Healthcare",
    "ERIS": "Pharmaceuticals & Healthcare",
    "GRANULES": "Pharmaceuticals & Healthcare",
    "SUVENPHAR": "Pharmaceuticals & Healthcare",
    "NEULANDLAB": "Pharmaceuticals & Healthcare",
    "MARKSANS": "Pharmaceuticals & Healthcare",

    # Fast Moving Consumer Goods (FMCG)
    "HINDUNILVR": "Fast Moving Consumer Goods (FMCG)",
    "ITC": "Fast Moving Consumer Goods (FMCG)",
    "NESTLEIND": "Fast Moving Consumer Goods (FMCG)",
    "BRITANNIA": "Fast Moving Consumer Goods (FMCG)",
    "TATACONSUM": "Fast Moving Consumer Goods (FMCG)",
    "VBL": "Fast Moving Consumer Goods (FMCG)",
    "GODREJCP": "Fast Moving Consumer Goods (FMCG)",
    "DABUR": "Fast Moving Consumer Goods (FMCG)",
    "MARICO": "Fast Moving Consumer Goods (FMCG)",
    "COLPAL": "Fast Moving Consumer Goods (FMCG)",
    "PGHH": "Fast Moving Consumer Goods (FMCG)",
    "EMAMILTD": "Fast Moving Consumer Goods (FMCG)",
    "RADICO": "Fast Moving Consumer Goods (FMCG)",
    "UBL": "Fast Moving Consumer Goods (FMCG)",
    "MCDOWELL-N": "Fast Moving Consumer Goods (FMCG)",
    "PATANJALI": "Fast Moving Consumer Goods (FMCG)",
    "BIKAJI": "Fast Moving Consumer Goods (FMCG)",
    "HONASA": "Fast Moving Consumer Goods (FMCG)",
    "JYOTHYLAB": "Fast Moving Consumer Goods (FMCG)",
    "CCL": "Fast Moving Consumer Goods (FMCG)",
    "BALRAMCHIN": "Fast Moving Consumer Goods (FMCG)",
    "EIDPARRY": "Fast Moving Consumer Goods (FMCG)",
    "RENUKA": "Fast Moving Consumer Goods (FMCG)",

    # Consumer Durables & Electronics
    "TITAN": "Consumer Durables & Electronics",
    "HAVELLS": "Consumer Durables & Electronics",
    "DIXON": "Consumer Durables & Electronics",
    "VOLTAS": "Consumer Durables & Electronics",
    "BLUESTARCO": "Consumer Durables & Electronics",
    "CROMPTON": "Consumer Durables & Electronics",
    "WHIRLPOOL": "Consumer Durables & Electronics",
    "AMBER": "Consumer Durables & Electronics",
    "KAJARIACER": "Consumer Durables & Electronics",
    "CERA": "Consumer Durables & Electronics",
    "SYRMA": "Consumer Durables & Electronics",
    "KAYNES": "Consumer Durables & Electronics",
    "CENTURYPLY": "Consumer Durables & Electronics",
    "GREENPANEL": "Consumer Durables & Electronics",
    "ORIENTELEC": "Consumer Durables & Electronics",
    "BAJAJELEC": "Consumer Durables & Electronics",
    "VGUARD": "Consumer Durables & Electronics",
    "TTKPRESTIG": "Consumer Durables & Electronics",
    "RAJESHEXPO": "Consumer Durables & Electronics",
    "KALYANKJIL": "Consumer Durables & Electronics",
    "SENCO": "Consumer Durables & Electronics",

    # Metals & Mining
    "TATASTEEL": "Metals & Mining",
    "JSWSTEEL": "Metals & Mining",
    "HINDALCO": "Metals & Mining",
    "VEDL": "Metals & Mining",
    "COALINDIA": "Metals & Mining",
    "JINDALSTEL": "Metals & Mining",
    "NMDC": "Metals & Mining",
    "SAIL": "Metals & Mining",
    "NATIONALUM": "Metals & Mining",
    "HINDZINC": "Metals & Mining",
    "APLAPOLLO": "Metals & Mining",
    "RATNAMANI": "Metals & Mining",
    "WELCORP": "Metals & Mining",
    "JSL": "Metals & Mining",
    "SHYAMMETL": "Metals & Mining",
    "LLOYDSENGG": "Metals & Mining",
    "MOIL": "Metals & Mining",
    "GPIL": "Metals & Mining",
    "KIOCL": "Metals & Mining",

    # Oil, Gas & Petroleum
    "RELIANCE": "Oil, Gas & Petroleum",
    "ONGC": "Oil, Gas & Petroleum",
    "BPCL": "Oil, Gas & Petroleum",
    "IOC": "Oil, Gas & Petroleum",
    "HPCL": "Oil, Gas & Petroleum",
    "GAIL": "Oil, Gas & Petroleum",
    "OIL": "Oil, Gas & Petroleum",
    "PETRONET": "Oil, Gas & Petroleum",
    "GUJGASLTD": "Oil, Gas & Petroleum",
    "IGL": "Oil, Gas & Petroleum",
    "MGL": "Oil, Gas & Petroleum",
    "ATGL": "Oil, Gas & Petroleum",
    "AEGISLOG": "Oil, Gas & Petroleum",
    "DEEPINDS": "Oil, Gas & Petroleum",
    "CASTROLIND": "Oil, Gas & Petroleum",
    "MRPL": "Oil, Gas & Petroleum",
    "CHENNPETRO": "Oil, Gas & Petroleum",

    # Power & Renewable Energy
    "NTPC": "Power & Renewable Energy",
    "POWERGRID": "Power & Renewable Energy",
    "ADANIPOWER": "Power & Renewable Energy",
    "ADANIGREEN": "Power & Renewable Energy",
    "ADANIENSOL": "Power & Renewable Energy",
    "TATAPOWER": "Power & Renewable Energy",
    "JSWENERGY": "Power & Renewable Energy",
    "NHPC": "Power & Renewable Energy",
    "SJVN": "Power & Renewable Energy",
    "TORNTPOWER": "Power & Renewable Energy",
    "SUZLON": "Power & Renewable Energy",
    "INOXWIND": "Power & Renewable Energy",
    "CESC": "Power & Renewable Energy",
    "NLCINDIA": "Power & Renewable Energy",
    "IEX": "Power & Renewable Energy",
    "JPPOWER": "Power & Renewable Energy",
    "RPOWER": "Power & Renewable Energy",
    "RTNPOWER": "Power & Renewable Energy",
    "KPIGREEN": "Power & Renewable Energy",
    "WAAREE": "Power & Renewable Energy",

    # Chemicals & Petrochemicals
    "PIDILITIND": "Chemicals & Petrochemicals",
    "SRF": "Chemicals & Petrochemicals",
    "GUJFLUORO": "Chemicals & Petrochemicals",
    "DEEPAKNTR": "Chemicals & Petrochemicals",
    "TATACHEM": "Chemicals & Petrochemicals",
    "AARTIIND": "Chemicals & Petrochemicals",
    "NAVINFLUOR": "Chemicals & Petrochemicals",
    "ATUL": "Chemicals & Petrochemicals",
    "VINATIORGA": "Chemicals & Petrochemicals",
    "FINEORG": "Chemicals & Petrochemicals",
    "CLEAN": "Chemicals & Petrochemicals",
    "ALKYLAMINE": "Chemicals & Petrochemicals",
    "BALAMINES": "Chemicals & Petrochemicals",
    "FLUOROCHEM": "Chemicals & Petrochemicals",
    "ANUPAM": "Chemicals & Petrochemicals",
    "ROSSARI": "Chemicals & Petrochemicals",
    "AETHER": "Chemicals & Petrochemicals",
    "AMIORG": "Chemicals & Petrochemicals",
    "NEOGEN": "Chemicals & Petrochemicals",
    "SUMICHEM": "Chemicals & Petrochemicals",
    "GHCL": "Chemicals & Petrochemicals",

    # Real Estate & Realty
    "DLF": "Real Estate & Realty",
    "GODREJPROP": "Real Estate & Realty",
    "LODHA": "Real Estate & Realty",
    "MACROTECH": "Real Estate & Realty",
    "OBEROIRLTY": "Real Estate & Realty",
    "PRESTIGE": "Real Estate & Realty",
    "PHOENIXLTD": "Real Estate & Realty",
    "BRIGADE": "Real Estate & Realty",
    "SOBHA": "Real Estate & Realty",
    "SIGNATURE": "Real Estate & Realty",
    "MAHLIFE": "Real Estate & Realty",
    "SUNTECK": "Real Estate & Realty",
    "IBREALEST": "Real Estate & Realty",
    "KOLTEPATIL": "Real Estate & Realty",
    "PURVA": "Real Estate & Realty",
    "AJMERA": "Real Estate & Realty",
    "ASHIANA": "Real Estate & Realty",
    "KEYFINSERV": "Real Estate & Realty",

    # Capital Goods & Industrial Engineering
    "LT": "Capital Goods & Industrial Engineering",
    "SIEMENS": "Capital Goods & Industrial Engineering",
    "ABB": "Capital Goods & Industrial Engineering",
    "CUMMINSIND": "Capital Goods & Industrial Engineering",
    "BHEL": "Capital Goods & Industrial Engineering",
    "AIAENG": "Capital Goods & Industrial Engineering",
    "THERMAX": "Capital Goods & Industrial Engineering",
    "CARBORUNIV": "Capital Goods & Industrial Engineering",
    "TIMKEN": "Capital Goods & Industrial Engineering",
    "SKFINDIA": "Capital Goods & Industrial Engineering",
    "SCHAEFFLER": "Capital Goods & Industrial Engineering",
    "KEC": "Capital Goods & Industrial Engineering",
    "KALPATPOWR": "Capital Goods & Industrial Engineering",
    "KPIL": "Capital Goods & Industrial Engineering",
    "PRAJIND": "Capital Goods & Industrial Engineering",
    "ELECON": "Capital Goods & Industrial Engineering",
    "TRITURBINE": "Capital Goods & Industrial Engineering",
    "KIRLOSENG": "Capital Goods & Industrial Engineering",
    "INGERRAND": "Capital Goods & Industrial Engineering",
    "ELGIEQUIP": "Capital Goods & Industrial Engineering",
    "TITAGARH": "Capital Goods & Industrial Engineering",
    "JWL": "Capital Goods & Industrial Engineering",
    "TEXRAIL": "Capital Goods & Industrial Engineering",
    "RAILTEL": "Capital Goods & Industrial Engineering",
    "RVNL": "Capital Goods & Industrial Engineering",
    "IRFC": "Capital Goods & Industrial Engineering",
    "RITES": "Capital Goods & Industrial Engineering",
    "IRCON": "Capital Goods & Industrial Engineering",

    # Cement & Building Materials
    "ULTRACEMCO": "Cement & Building Materials",
    "AMBUJACEM": "Cement & Building Materials",
    "ACC": "Cement & Building Materials",
    "SHREECEM": "Cement & Building Materials",
    "DALBHARAT": "Cement & Building Materials",
    "JKCEMENT": "Cement & Building Materials",
    "RAMCOCEM": "Cement & Building Materials",
    "BIRLACORPN": "Cement & Building Materials",
    "HEIDELBERG": "Cement & Building Materials",
    "PRSMJOHNSN": "Cement & Building Materials",
    "STARCEMENT": "Cement & Building Materials",
    "SAGCEM": "Cement & Building Materials",
    "ASTRAL": "Cement & Building Materials",
    "SUPREMEIND": "Cement & Building Materials",
    "FINPIPE": "Cement & Building Materials",
    "PRINCEPIPE": "Cement & Building Materials",
    "ASIANPAINT": "Cement & Building Materials",
    "BERGEPAINT": "Cement & Building Materials",
    "KANSAINER": "Cement & Building Materials",
    "AKZOINDIA": "Cement & Building Materials",
    "INDIGOPNTS": "Cement & Building Materials",

    # Media, Telecom & Entertainment
    "BHARTIARTL": "Media, Telecom & Entertainment",
    "INDUSTOWER": "Media, Telecom & Entertainment",
    "IDEA": "Media, Telecom & Entertainment",
    "TATACOMM": "Media, Telecom & Entertainment",
    "TTML": "Media, Telecom & Entertainment",
    "HFCL": "Media, Telecom & Entertainment",
    "STLTECH": "Media, Telecom & Entertainment",
    "TEJASNET": "Media, Telecom & Entertainment",
    "OPTCL": "Media, Telecom & Entertainment",
    "SUNTV": "Media, Telecom & Entertainment",
    "ZEEL": "Media, Telecom & Entertainment",
    "PVRINOX": "Media, Telecom & Entertainment",
    "NETWORK18": "Media, Telecom & Entertainment",
    "TV18BRDCST": "Media, Telecom & Entertainment",
    "SAREGAMA": "Media, Telecom & Entertainment",
    "TIPSINDLTD": "Media, Telecom & Entertainment",
    "NAZARA": "Media, Telecom & Entertainment",
    "DBCORP": "Media, Telecom & Entertainment",
    "JAGRAN": "Media, Telecom & Entertainment",

    # Textiles & Apparel
    "PAGEIND": "Textiles & Apparel",
    "TRENT": "Textiles & Apparel",
    "ABFRL": "Textiles & Apparel",
    "RAYMOND": "Textiles & Apparel",
    "KPRMILL": "Textiles & Apparel",
    "TRIDENT": "Textiles & Apparel",
    "ALOKINDS": "Textiles & Apparel",
    "GOCLCORP": "Textiles & Apparel",
    "LUXIND": "Textiles & Apparel",
    "RUPA": "Textiles & Apparel",
    "DOLLAR": "Textiles & Apparel",
    "WELSPUNLIV": "Textiles & Apparel",
    "GARFIBRES": "Textiles & Apparel",
    "VTL": "Textiles & Apparel",
    "ARVIND": "Textiles & Apparel",
    "SPAL": "Textiles & Apparel",
    "GOKEX": "Textiles & Apparel",

    # Logistics, Shipping & Aviation
    "ADANIPORTS": "Logistics, Shipping & Aviation",
    "INDIGO": "Logistics, Shipping & Aviation",
    "DELHIVERY": "Logistics, Shipping & Aviation",
    "CONCOR": "Logistics, Shipping & Aviation",
    "BLUEDART": "Logistics, Shipping & Aviation",
    "MAHLOG": "Logistics, Shipping & Aviation",
    "TCIEXP": "Logistics, Shipping & Aviation",
    "TCI": "Logistics, Shipping & Aviation",
    "ALLCARGO": "Logistics, Shipping & Aviation",
    "VRLLOG": "Logistics, Shipping & Aviation",
    "SCI": "Logistics, Shipping & Aviation",
    "GESHIP": "Logistics, Shipping & Aviation",
    "SPICEJET": "Logistics, Shipping & Aviation",
    "GATEWAY": "Logistics, Shipping & Aviation",

    # Defense & Aerospace
    "HAL": "Defense & Aerospace",
    "BEL": "Defense & Aerospace",
    "MAZDOCK": "Defense & Aerospace",
    "COCHINSHIP": "Defense & Aerospace",
    "BDL": "Defense & Aerospace",
    "DATAPATTNS": "Defense & Aerospace",
    "PARAS": "Defense & Aerospace",
    "ASTRAMICRO": "Defense & Aerospace",
    "MTARTECH": "Defense & Aerospace",
    "ZEN": "Defense & Aerospace",
    "GRSE": "Defense & Aerospace",
    "DYNAMATECH": "Defense & Aerospace",
    "APOLLO": "Defense & Aerospace",

    # Hotels, Tourism & Leisure
    "INDHOTEL": "Hotels, Tourism & Leisure",
    "EIHOTEL": "Hotels, Tourism & Leisure",
    "LEMONTREE": "Hotels, Tourism & Leisure",
    "CHALET": "Hotels, Tourism & Leisure",
    "JUNIPER": "Hotels, Tourism & Leisure",
    "PARK": "Hotels, Tourism & Leisure",
    "SAMHI": "Hotels, Tourism & Leisure",
    "TAJGVK": "Hotels, Tourism & Leisure",
    "MAHINDCIE": "Hotels, Tourism & Leisure",
    "EASEMYTRIP": "Hotels, Tourism & Leisure",
    "YATRA": "Hotels, Tourism & Leisure",
    "BLS": "Hotels, Tourism & Leisure",
    "WONDERLA": "Hotels, Tourism & Leisure",
    "DELTA CORP": "Hotels, Tourism & Leisure",
    "DELTACORP": "Hotels, Tourism & Leisure",

    # Agriculture, Fertilizers & Sugar
    "COROMANDEL": "Agriculture, Fertilizers & Sugar",
    "PIIND": "Agriculture, Fertilizers & Sugar",
    "UPL": "Agriculture, Fertilizers & Sugar",
    "CHAMBLFERT": "Agriculture, Fertilizers & Sugar",
    "GNFC": "Agriculture, Fertilizers & Sugar",
    "GSFC": "Agriculture, Fertilizers & Sugar",
    "DEEPAKFERT": "Agriculture, Fertilizers & Sugar",
    "RCF": "Agriculture, Fertilizers & Sugar",
    "NFL": "Agriculture, Fertilizers & Sugar",
    "FACT": "Agriculture, Fertilizers & Sugar",
    "BAYERCROP": "Agriculture, Fertilizers & Sugar",
    "SUMICHEM": "Agriculture, Fertilizers & Sugar",
    "DHANUKA": "Agriculture, Fertilizers & Sugar",
    "RALLIS": "Agriculture, Fertilizers & Sugar",
    "SHARDACROP": "Agriculture, Fertilizers & Sugar",
    "KAVERI": "Agriculture, Fertilizers & Sugar",
    "AVANTIFEED": "Agriculture, Fertilizers & Sugar",
    "GODREJAGRO": "Agriculture, Fertilizers & Sugar",

    # Retail & E-Commerce
    "DMART": "Retail & E-Commerce",
    "AVENUE": "Retail & E-Commerce",
    "ZOMATO": "Retail & E-Commerce",
    "SWIGGY": "Retail & E-Commerce",
    "NYKAA": "Retail & E-Commerce",
    "PAYTM": "Retail & E-Commerce",
    "POLICYBZR": "Retail & E-Commerce",
    "MEDPLUS": "Retail & E-Commerce",
    "ETHOSLTD": "Retail & E-Commerce",
    "VEDANTFASH": "Retail & E-Commerce",
    "SHOPERSTOP": "Retail & E-Commerce",
    "VMART": "Retail & E-Commerce",

    # Conglomerates & Diversified
    "ADANIENT": "Conglomerates & Diversified",
    "GRASIM": "Conglomerates & Diversified",
    "ITC": "Fast Moving Consumer Goods (FMCG)",
    "3MINDIA": "Conglomerates & Diversified",
    "DCMSRIRAM": "Conglomerates & Diversified",
}

# Nifty Sectoral Indices Mapping
NIFTY_SECTORAL_INDICES = {
    "NIFTY AUTO": [
        "MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "EICHERMOT", "HEROMOTOCO",
        "TVSMOTOR", "BHARATFORG", "BOSCHLTD", "MOTHERSON", "MRF", "BALKRISIND",
        "APOLLOTYRE", "SONACOMS", "EXIDEIND", "ASHOKLEY", "TIINDIA"
    ],
    "NIFTY BANK": [
        "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "INDUSINDBK",
        "BANKBARODA", "PNB", "FEDERALBNK", "IDFCFIRSTB", "AUBANK", "BANDHANBNK"
    ],
    "NIFTY FINANCIAL SERVICES": [
        "HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "SBIN", "BAJFINANCE",
        "BAJAJFINSV", "SHRIRAMFIN", "CHOLAFIN", "HDFCLIFE", "SBILIFE", "ICICIPRULI",
        "ICICIGI", "PFC", "RECLTD", "MUTHOOTFIN", "JIOFIN"
    ],
    "NIFTY FMCG": [
        "ITC", "HINDUNILVR", "NESTLEIND", "BRITANNIA", "TATACONSUM", "VBL",
        "GODREJCP", "DABUR", "MARICO", "COLPAL", "PGHH", "RADICO", "UBL",
        "UNITEDSPR", "EMAMILTD", "BALRAMCHIN"
    ],
    "NIFTY IT": [
        "TCS", "INFY", "HCLTECH", "WIPRO", "LTIM", "TECHM", "PERSISTENT",
        "COFORGE", "MPHASIS", "LTTS"
    ],
    "NIFTY MEDIA": [
        "SUNTV", "ZEEL", "PVRINOX", "NETWORK18", "TV18BRDCST", "NAZARA",
        "DBCORP", "JAGRAN", "HATHWAY", "DEN", "DISHTV"
    ],
    "NIFTY METAL": [
        "TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "COALINDIA", "JINDALSTEL",
        "NMDC", "SAIL", "NATIONALUM", "HINDZINC", "APLAPOLLO", "RATNAMANI", "WELCORP"
    ],
    "NIFTY PHARMA": [
        "SUNPHARMA", "CIPLA", "DRREDDY", "DIVISLAB", "LUPIN", "TORNTPHARM",
        "ZYDUSLIFE", "MANKIND", "AUROPHARMA", "ALKEM", "BIOCON", "GLENMARK",
        "IPCALAB", "LAURUSLABS", "ABBOTINDIA"
    ],
    "NIFTY PSU BANK": [
        "SBIN", "BANKBARODA", "PNB", "CANBK", "UNIONBANK", "INDIANB", "IOB",
        "BANKINDIA", "CENTRALBK", "MAHABANK", "UCOBANK", "PSB"
    ],
    "NIFTY REALTY": [
        "DLF", "GODREJPROP", "LODHA", "MACROTECH", "OBEROIRLTY", "PRESTIGE",
        "PHOENIXLTD", "BRIGADE", "SOBHA", "SIGNATURE", "MAHLIFE"
    ],
    "NIFTY HEALTHCARE": [
        "SUNPHARMA", "CIPLA", "DRREDDY", "DIVISLAB", "APOLLOHOSP", "LUPIN",
        "MAXHEALTH", "TORNTPHARM", "ZYDUSLIFE", "FORTIS", "MEDANTA", "LALPATHLAB"
    ],
    "NIFTY CONSUMER DURABLES": [
        "TITAN", "HAVELLS", "DIXON", "VOLTAS", "BLUESTARCO", "CROMPTON",
        "WHIRLPOOL", "AMBER", "KAJARIACER", "CERA", "CENTURYPLY", "VGUARD", "KALYANKJIL"
    ],
    "NIFTY OIL & GAS": [
        "RELIANCE", "ONGC", "BPCL", "IOC", "HPCL", "GAIL", "OIL", "PETRONET",
        "GUJGASLTD", "IGL", "MGL", "ATGL", "AEGISLOG", "CASTROLIND"
    ]
}


def infer_sector_from_name(symbol: str, company_name: str = "") -> str:
    """Intelligently classifies any unmapped NSE company based on name clues."""
    txt = f"{symbol} {company_name}".upper()

    if any(k in txt for k in ["BANK", "FINANCE", "CAPITAL", "INVEST", "HOLDINGS", "SECURITIES", "FINSERV", "WEALTH"]):
        if any(k in txt for k in ["STATE BANK", "PUNJAB", "BARODA", "CANARA", "UNION BANK", "INDIAN BANK", "UCO", "CENTRAL BANK", "MAHARASHTRA"]):
            return "Banking - PSU"
        elif "BANK" in txt:
            return "Banking - Private"
        return "Financial Services & NBFC"

    if any(k in txt for k in ["TECH", "SOFT", "INFO", "DIGITAL", "SYSTEM", "CYBER", "SOLUTIONS", "DATA", "TELECOM"]):
        return "Information Technology"

    if any(k in txt for k in ["PHARMA", "HEALTH", "DRUG", "BIO", "LAB", "HOSPITAL", "CLINIC", "MEDIC"]):
        return "Pharmaceuticals & Healthcare"

    if any(k in txt for k in ["AUTO", "MOTOR", "TYRE", "WHEEL", "BEARING", "BRAKE", "ENGINE", "AUTOMOTIVE"]):
        return "Automobiles & Auto Ancillaries"

    if any(k in txt for k in ["POWER", "ENERGY", "SOLAR", "WIND", "ELECTRIC", "GRID", "RENEWABLE", "HYDRO"]):
        return "Power & Renewable Energy"

    if any(k in txt for k in ["OIL", "GAS", "PETRO", "REFINER", "HYDROCARBON", "FUELS"]):
        return "Oil, Gas & Petroleum"

    if any(k in txt for k in ["STEEL", "IRON", "METAL", "MINING", "ALUMIN", "COPPER", "ZINC", "MINERAL", "ALLOY"]):
        return "Metals & Mining"

    if any(k in txt for k in ["CHEM", "FERT", "POLYMER", "ACID", "CARBON", "PESTICIDE"]):
        return "Chemicals & Petrochemicals"

    if any(k in txt for k in ["REALTY", "ESTATE", "PROP", "INFRA", "BUILD", "HOUSING", "DEVELOPER"]):
        return "Real Estate & Realty"

    if any(k in txt for k in ["CEMENT", "PIPE", "CERAMIC", "PAINT", "TILE", "GLASS", "PLY"]):
        return "Cement & Building Materials"

    if any(k in txt for k in ["FOOD", "TEA", "COFFEE", "SUGAR", "CONSUMER", "DAIRY", "BREWER", "BEVERAGE", "EDIBLE", "TOBACCO"]):
        return "Fast Moving Consumer Goods (FMCG)"

    if any(k in txt for k in ["TEXTILE", "SPINNING", "YARN", "FABRIC", "GARMENT", "COTTON", "DENIM", "APPAREL", "FASHION"]):
        return "Textiles & Apparel"

    if any(k in txt for k in ["LOGISTIC", "SHIPPING", "PORT", "CARRIER", "TRANSPORT", "FREIGHT", "EXPRESS"]):
        return "Logistics, Shipping & Aviation"

    if any(k in txt for k in ["HOTEL", "RESORT", "TRAVEL", "TOURISM", "HOLIDAY", "HOSPITALITY"]):
        return "Hotels, Tourism & Leisure"

    if any(k in txt for k in ["DEFENCE", "AEROSPACE", "SHIPYARD", "MISSILE", "RADAR"]):
        return "Defense & Aerospace"

    if any(k in txt for k in ["RETAIL", "MART", "STORE", "MALL", "COMMERCE"]):
        return "Retail & E-Commerce"

    if any(k in txt for k in ["ENGG", "ENGINEER", "ELECTRICAL", "MACHIN", "TOOLS", "HEAVY"]):
        return "Capital Goods & Industrial Engineering"

    return "Conglomerates & Diversified"


def get_symbol_sector(symbol: str, company_name: str = "") -> str:
    """Returns the Definedge sector for a given symbol."""
    sym_clean = symbol.upper().replace("-EQ", "").replace(".NS", "").strip()
    if sym_clean in SYMBOL_TO_DEFINEDGE_SECTOR:
        return SYMBOL_TO_DEFINEDGE_SECTOR[sym_clean]
    return infer_sector_from_name(sym_clean, company_name)


# ---------------------------------------------------------------------------
# Vectorized Multi-Timeframe Return Engine
# ---------------------------------------------------------------------------

def calculate_market_heatmap_data() -> pd.DataFrame:
    """
    Computes Daily (1D), Weekly (1W), Monthly (1M), and Yearly (1Y)
    performance for all NSE symbols from SQLite daily candles.
    Uses ROW_NUMBER() window partitioning for robust handling of every stock's
    latest available candle.
    """
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
    try:
        # Fetch company names from instruments master
        inst_df = pd.read_sql_query(
            "SELECT trading_symbol, name, instrument_type, exchange FROM instruments WHERE exchange = 'NSE_EQ';",
            conn
        )
        name_map = dict(zip(inst_df["trading_symbol"], inst_df["name"]))

        # Window query to get latest candle (rn=1), 1 day ago (rn=2), 1 week ago (rn=6), 1 month ago (rn=22), 1 year ago (rn=251)
        q = """
            WITH ranked AS (
                SELECT trading_symbol, date, open, high, low, close, volume,
                       ROW_NUMBER() OVER (PARTITION BY trading_symbol ORDER BY date DESC) as rn
                FROM daily_candles
                WHERE date >= date((SELECT max(date) FROM daily_candles), '-400 days')
            )
            SELECT trading_symbol, date, open, high, low, close, volume, rn
            FROM ranked
            WHERE rn IN (1, 2, 6, 22, 251);
        """
        df_candles = pd.read_sql_query(q, conn)

        if df_candles.empty:
            return pd.DataFrame()

        piv_close = df_candles.pivot(index="trading_symbol", columns="rn", values="close")
        rn1 = df_candles[df_candles["rn"] == 1].set_index("trading_symbol")

        results = pd.DataFrame(index=piv_close.index)
        results["Symbol"] = piv_close.index
        results["Company_Name"] = results["Symbol"].map(lambda s: name_map.get(s, s))
        
        # Sector Mapping
        results["Sector"] = results.apply(
            lambda r: get_symbol_sector(r["Symbol"], r["Company_Name"]), axis=1
        )

        results["LTP"] = piv_close[1].round(2)
        results["Open"] = rn1["open"].round(2) if "open" in rn1 else results["LTP"]
        results["High"] = rn1["high"].round(2) if "high" in rn1 else results["LTP"]
        results["Low"] = rn1["low"].round(2) if "low" in rn1 else results["LTP"]
        results["Volume"] = rn1["volume"].fillna(0).astype(int) if "volume" in rn1 else 0
        results["Date"] = rn1["date"] if "date" in rn1 else ""

        c_prev = piv_close[2] if 2 in piv_close else results["LTP"]
        c_week = piv_close[6].fillna(c_prev) if 6 in piv_close else c_prev
        c_month = piv_close[22].fillna(c_week) if 22 in piv_close else c_week
        c_year = piv_close[251].fillna(c_month) if 251 in piv_close else c_month

        # Returns
        results["Return_1D"] = (((results["LTP"] - c_prev) / c_prev) * 100.0).round(2).fillna(0.0)
        results["Return_1W"] = (((results["LTP"] - c_week) / c_week) * 100.0).round(2).fillna(0.0)
        results["Return_1M"] = (((results["LTP"] - c_month) / c_month) * 100.0).round(2).fillna(0.0)
        results["Return_1Y"] = (((results["LTP"] - c_year) / c_year) * 100.0).round(2).fillna(0.0)

        results["Turnover"] = (results["LTP"] * results["Volume"]).round(0)

        results.dropna(subset=["LTP"], inplace=True)
        results = results[results["LTP"] > 0]
        results.reset_index(drop=True, inplace=True)

        return results
    finally:
        conn.close()


def generate_cmc_heatmap_html(
    df: pd.DataFrame,
    active_tf: str = "Daily (1D)",
    theme: str = "dark",
    height: int = 740,
    view_mode: str = "dominance"
) -> str:
    """
    Renders a CoinMarketCap / Finviz-style squarified market heatmap using D3.js.
    Features:
    - Large bold typography inside tiles (Symbol, Price, Change % with arrows, Dominance %).
    - Dynamic squarified treemap tiling matching CoinMarketCap / Finviz.
    - Solid emerald green (#00B36B) for gainers, crimson red (#C61F2E) for losers, slate (#6B7280) for flat.
    - In-canvas Timeframe Selector: [1D] [1W] [1M] [1Y].
    - View Mode Switcher: Market Dominance vs Sector-Wise Grouping.
    - Quick Search filter and Native Fullscreen toggle.
    """
    import json

    if df.empty:
        return f"""<div style="height:{height}px; display:flex; align-items:center; justify-content:center; background:#131722; color:#94A3B8; font-family:sans-serif; border-radius:8px;">No heatmap data available.</div>"""

    is_light = (str(theme).lower() == "light")

    # Prepare records
    records = []
    tot_to = df["Turnover"].sum() if "Turnover" in df and df["Turnover"].sum() > 0 else 1.0

    for _, row in df.iterrows():
        sym = str(row["Symbol"]).upper()
        name = str(row.get("Company_Name", sym))
        sec = str(row.get("Sector", "Conglomerates & Diversified"))
        ltp = float(row.get("LTP", 0))
        c1d = float(row.get("Return_1D", 0))
        c1w = float(row.get("Return_1W", 0))
        c1m = float(row.get("Return_1M", 0))
        c1y = float(row.get("Return_1Y", 0))
        to = float(row.get("Turnover", 1000))
        dom = round((to / tot_to) * 100, 2)

        records.append({
            "symbol": sym,
            "name": name,
            "sector": sec,
            "price": round(ltp, 2),
            "chg_1d": round(c1d, 2),
            "chg_1w": round(c1w, 2),
            "chg_1m": round(c1m, 2),
            "chg_1y": round(c1y, 2),
            "turnover": round(to, 0),
            "dominance": dom
        })

    data_json = json.dumps(records)

    tf_initial = "1D"
    if "1W" in active_tf:
        tf_initial = "1W"
    elif "1M" in active_tf:
        tf_initial = "1M"
    elif "1Y" in active_tf:
        tf_initial = "1Y"

    html_code = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Market Heatmap - CoinMarketCap Style</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        html, body {{
            width: 100%;
            height: 100%;
            overflow: hidden;
            background-color: {'#ffffff' if is_light else '#0e1117'};
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            color: {'#131722' if is_light else '#ffffff'};
            user-select: none;
        }}
        .heatmap-container {{
            width: 100%;
            height: {height}px;
            display: flex;
            flex-direction: column;
            background-color: {'#ffffff' if is_light else '#0e1117'};
            border: 1px solid {'#e2e8f0' if is_light else '#1e222d'};
            border-radius: 8px;
            overflow: hidden;
            position: relative;
        }}
        .heatmap-container.css-fullscreen {{
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            width: 100vw !important;
            height: 100vh !important;
            z-index: 999999 !important;
            border-radius: 0 !important;
            border: none !important;
        }}
        /* Header Toolbar */
        .hm-toolbar {{
            height: 44px;
            padding: 0 12px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            background-color: {'#f8fafc' if is_light else '#151922'};
            border-bottom: 1px solid {'#e2e8f0' if is_light else '#222734'};
            flex-shrink: 0;
            z-index: 10;
        }}
        .hm-toolbar-left {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .hm-title {{
            font-size: 13px;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 6px;
            color: {'#0f172a' if is_light else '#f8fafc'};
        }}
        .pill-group {{
            display: flex;
            align-items: center;
            background: {'#e2e8f0' if is_light else '#1e2330'};
            padding: 2px;
            border-radius: 6px;
            gap: 2px;
        }}
        .pill-btn, .pill-btn-univ {{
            background: transparent;
            border: none;
            color: {'#64748b' if is_light else '#94a3b8'};
            padding: 3px 10px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s;
        }}
        .pill-btn:hover, .pill-btn-univ:hover {{
            color: {'#0f172a' if is_light else '#ffffff'};
        }}
        .pill-btn.active, .pill-btn-univ.active {{
            background: #2563EB;
            color: #ffffff;
        }}
        .hm-toolbar-right {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .search-input {{
            background: {'#ffffff' if is_light else '#1e2330'};
            border: 1px solid {'#cbd5e1' if is_light else '#2e3648'};
            color: {'#0f172a' if is_light else '#ffffff'};
            padding: 4px 8px;
            border-radius: 5px;
            font-size: 11px;
            outline: none;
            width: 140px;
        }}
        .action-btn {{
            background: {'#ffffff' if is_light else '#1e2330'};
            border: 1px solid {'#cbd5e1' if is_light else '#2e3648'};
            color: {'#334155' if is_light else '#cbd5e1'};
            padding: 4px 10px;
            border-radius: 5px;
            font-size: 11px;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 4px;
            transition: all 0.15s;
        }}
        .action-btn:hover {{
            background: {'#f1f5f9' if is_light else '#2a3142'};
            color: {'#0f172a' if is_light else '#ffffff'};
        }}
        /* Treemap Canvas Area */
        .treemap-area {{
            flex: 1;
            width: 100%;
            height: calc(100% - 44px);
            position: relative;
            overflow: hidden;
            background-color: {'#f1f5f9' if is_light else '#0a0d14'};
        }}
        /* Treemap Node (Tile) */
        .tm-tile {{
            position: absolute;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            border-radius: 4px;
            cursor: pointer;
            box-shadow: inset 0 0 0 1px rgba(0,0,0,0.25);
            transition: filter 0.15s ease, transform 0.15s ease;
            overflow: hidden;
            padding: 4px;
        }}
        .tm-tile:hover {{
            filter: brightness(1.18);
            z-index: 50 !important;
            box-shadow: 0 4px 14px rgba(0,0,0,0.5), inset 0 0 0 1.5px rgba(255,255,255,0.4);
        }}
        .tm-sym {{
            font-weight: 800;
            color: #FFFFFF;
            line-height: 1.1;
            letter-spacing: -0.5px;
            text-shadow: 0 1px 3px rgba(0,0,0,0.5);
        }}
        .tm-price {{
            font-weight: 700;
            color: rgba(255, 255, 255, 0.95);
            line-height: 1.15;
            margin-top: 2px;
            text-shadow: 0 1px 2px rgba(0,0,0,0.4);
        }}
        .tm-chg {{
            font-weight: 700;
            color: #FFFFFF;
            display: flex;
            align-items: center;
            gap: 2px;
            margin-top: 2px;
            text-shadow: 0 1px 2px rgba(0,0,0,0.5);
        }}
        .tm-dom {{
            color: rgba(255, 255, 255, 0.75);
            font-weight: 500;
            margin-top: 4px;
            font-size: 10px;
            letter-spacing: 0.2px;
            text-shadow: 0 1px 2px rgba(0,0,0,0.5);
        }}
        /* Sector Header Border (When in sector view) */
        .tm-sector-box {{
            position: absolute;
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 5px;
            pointer-events: none;
        }}
        .tm-sector-label {{
            position: absolute;
            top: 2px;
            left: 6px;
            font-size: 10px;
            font-weight: 700;
            color: rgba(255, 255, 255, 0.7);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            pointer-events: none;
            z-index: 5;
        }}
        /* Floating Detail Card / Tooltip */
        .hm-tooltip {{
            position: absolute;
            display: none;
            background: rgba(15, 23, 42, 0.96);
            border: 1px solid #38BDF8;
            border-radius: 6px;
            padding: 8px 12px;
            font-size: 11px;
            color: #F8FAFC;
            box-shadow: 0 8px 24px rgba(0,0,0,0.6);
            pointer-events: none;
            z-index: 99999;
            backdrop-filter: blur(6px);
            width: 210px;
            line-height: 1.4;
        }}
        .hm-tip-sym {{
            font-weight: 800;
            font-size: 13px;
            color: #38BDF8;
            margin-bottom: 3px;
            padding-bottom: 2px;
            border-bottom: 1px solid #334155;
        }}
        .hm-tip-row {{
            display: flex;
            justify-content: space-between;
            margin: 2px 0;
        }}
        .hm-tip-key {{ color: #94A3B8; }}
        .hm-tip-val {{ font-weight: 600; font-family: monospace; }}
    </style>
</head>
<body>
    <div id="hm_container" class="heatmap-container">
        <!-- Top Toolbar -->
        <div class="hm-toolbar">
            <div class="hm-toolbar-left">
                <div class="hm-title">
                    <span>🪙 Market Heatmap</span>
                </div>
                <!-- Timeframe Pills -->
                <div class="pill-group">
                    <button class="pill-btn {'active' if tf_initial == '1D' else ''}" data-tf="1D">1D</button>
                    <button class="pill-btn {'active' if tf_initial == '1W' else ''}" data-tf="1W">1W</button>
                    <button class="pill-btn {'active' if tf_initial == '1M' else ''}" data-tf="1M">1M</button>
                    <button class="pill-btn {'active' if tf_initial == '1Y' else ''}" data-tf="1Y">1Y</button>
                </div>
                <!-- Universe Pills -->
                <div class="pill-group">
                    <button class="pill-btn-univ" data-univ="50">Top 50</button>
                    <button class="pill-btn-univ active" data-univ="100">Top 100</button>
                    <button class="pill-btn-univ" data-univ="250">Top 250</button>
                    <button class="pill-btn-univ" data-univ="500">Top 500</button>
                    <button class="pill-btn-univ" data-univ="0">All</button>
                </div>
                <!-- Grouping Toggle -->
                <div class="pill-group">
                    <button id="view_dom_btn" class="pill-btn active" title="Single Squarified Market Heatmap (CoinMarketCap Style)">Market Dominance</button>
                    <button id="view_sec_btn" class="pill-btn" title="Group Stocks by Definedge Sectors">By Sector</button>
                </div>
            </div>
            <div class="hm-toolbar-right">
                <input type="text" id="sym_search" class="search-input" placeholder="🔍 Search symbol..." />
                <button id="fs_toggle" class="action-btn">⛶ Fullscreen</button>
            </div>
        </div>

        <!-- Treemap Area -->
        <div id="treemap_area" class="treemap-area"></div>

        <!-- Tooltip -->
        <div id="hm_tooltip" class="hm-tooltip"></div>
    </div>

    <script>
        (function() {{
            const rawData = {data_json};
            const container = document.getElementById("hm_container");
            const area = document.getElementById("treemap_area");
            const tooltip = document.getElementById("hm_tooltip");
            const fsBtn = document.getElementById("fs_toggle");
            const searchInput = document.getElementById("sym_search");

            let currentTf = "{tf_initial}";
            let currentMode = "{view_mode}"; // "dominance" or "sector"
            let currentUniverse = 100; // 50, 100, 250, 500, or 0 (all)

            function getReturn(d) {{
                if (currentTf === "1W") return d.chg_1w;
                if (currentTf === "1M") return d.chg_1m;
                if (currentTf === "1Y") return d.chg_1y;
                return d.chg_1d;
            }}

            // Exact CoinMarketCap Palette (Matching User Reference Image):
            // Positive (Gainers): Solid Emerald Green (#00B36B)
            // Negative (Losers): Solid Crimson Red (#C61F2E)
            // Flat / Unchanged: Solid Slate Grey (#717988)
            function getNodeColor(val) {{
                if (Math.abs(val) < 0.01) return "#717988";
                return val > 0 ? "#00B36B" : "#C61F2E";
            }}

            function render() {{
                area.innerHTML = "";
                const width = area.clientWidth || 1000;
                const height = area.clientHeight || 650;

                if (!rawData || rawData.length === 0) return;

                // Sort by turnover / dominance
                const sorted = [...rawData].sort((a, b) => b.turnover - a.turnover);

                // Filter by Universe if selected
                const displayed = (currentUniverse > 0 && sorted.length > currentUniverse)
                    ? sorted.slice(0, currentUniverse)
                    : sorted;

                // Recalculate dominance relative to displayed universe
                const totalDisplayTo = displayed.reduce((acc, x) => acc + (x.turnover || 1000), 0);
                displayed.forEach(d => {{
                    d.dominance = (((d.turnover || 1000) / totalDisplayTo) * 100).toFixed(2);
                }});

                let hierarchyData = null;
                if (currentMode === "sector") {{
                    const bySec = d3.group(displayed, d => d.sector);
                    const children = Array.from(bySec, ([sec, items]) => ({{
                        name: sec,
                        children: items
                    }}));
                    hierarchyData = {{ name: "root", children: children }};
                }} else {{
                    hierarchyData = {{ name: "root", children: displayed }};
                }}

                const root = d3.hierarchy(hierarchyData)
                    .sum(d => d.turnover || 1000)
                    .sort((a, b) => b.value - a.value);

                const treemap = d3.treemap()
                    .size([width, height])
                    .paddingOuter(2)
                    .paddingInner(2)
                    .tile(d3.treemapSquarify.ratio(1.618));

                treemap(root);

                const leaves = root.leaves();

                leaves.forEach(leaf => {{
                    const d = leaf.data;
                    const w = leaf.x1 - leaf.x0;
                    const h = leaf.y1 - leaf.y0;
                    if (w <= 2 || h <= 2) return;

                    const ret = getReturn(d);
                    const bg = getNodeColor(ret);

                    const tile = document.createElement("div");
                    tile.className = "tm-tile";
                    tile.style.left = `${{leaf.x0}}px`;
                    tile.style.top = `${{leaf.y0}}px`;
                    tile.style.width = `${{w}}px`;
                    tile.style.height = `${{h}}px`;
                    tile.style.backgroundColor = bg;
                    tile.setAttribute("data-sym", d.symbol);

                    // Dynamic font scaling according to tile area and dimensions
                    const minDim = Math.min(w, h);
                    let symSize = Math.max(9, Math.min(68, Math.floor(minDim * 0.22)));
                    let priceSize = Math.max(8, Math.min(42, Math.floor(symSize * 0.68)));
                    let chgSize = Math.max(8, Math.min(26, Math.floor(symSize * 0.50)));
                    let domSize = Math.max(7, Math.min(16, Math.floor(symSize * 0.32)));

                    const arrow = ret > 0 ? "▴" : (ret < 0 ? "▾" : "▾");
                    const sign = ret > 0 ? "+" : "";

                    // Exact CoinMarketCap Layout & Typography
                    if (w >= 90 && h >= 65) {{
                        tile.innerHTML = `
                            <div style="flex: 1; display: flex; flex-direction: column; justify-content: center; align-items: center; width: 100%;">
                                <div class="tm-sym" style="font-size: ${{symSize}}px; font-weight: 800; letter-spacing: -0.5px;">${{d.symbol}}</div>
                                <div class="tm-price" style="font-size: ${{priceSize}}px; font-weight: 700; margin-top: 3px;">₹${{d.price.toLocaleString('en-IN', {{minimumFractionDigits: 2, maximumFractionDigits: 2}})}}</div>
                                <div class="tm-chg" style="font-size: ${{chgSize}}px; font-weight: 600; margin-top: 3px;">${{arrow}} ${{Math.abs(ret).toFixed(2)}}%</div>
                            </div>
                            <div class="tm-dom" style="font-size: ${{domSize}}px; font-weight: 500; margin-bottom: 3px; opacity: 0.9;">Dominance : ${{d.dominance}}%</div>
                        `;
                    }} else if (w >= 60 && h >= 42) {{
                        tile.innerHTML = `
                            <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; width: 100%; height: 100%;">
                                <div class="tm-sym" style="font-size: ${{Math.max(10, Math.floor(symSize * 0.85))}}px; font-weight: 800;">${{d.symbol}}</div>
                                <div class="tm-price" style="font-size: ${{Math.max(9, Math.floor(priceSize * 0.85))}}px; font-weight: 600; margin-top: 1px;">₹${{d.price.toLocaleString('en-IN')}}</div>
                                <div class="tm-chg" style="font-size: ${{Math.max(8, Math.floor(chgSize * 0.85))}}px; font-weight: 600; margin-top: 1px;">${{arrow}} ${{Math.abs(ret).toFixed(2)}}%</div>
                            </div>
                        `;
                    }} else if (w >= 36 && h >= 24) {{
                        tile.innerHTML = `
                            <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; width: 100%; height: 100%;">
                                <div class="tm-sym" style="font-size: ${{Math.max(9, Math.floor(symSize * 0.72))}}px; font-weight: 700;">${{d.symbol}}</div>
                                <div class="tm-chg" style="font-size: ${{Math.max(8, Math.floor(chgSize * 0.72))}}px; font-weight: 600;">${{arrow}}${{Math.abs(ret).toFixed(1)}}%</div>
                            </div>
                        `;
                    }} else if (w >= 18 && h >= 12) {{
                        tile.innerHTML = `<div class="tm-sym" style="font-size: 8px; font-weight: 700; text-align: center;">${{d.symbol}}</div>`;
                    }}

                    // Hover Tooltip
                    tile.addEventListener("mouseenter", (e) => {{
                        tooltip.innerHTML = `
                            <div class="hm-tip-sym">${{d.symbol}}</div>
                            <div class="hm-tip-row"><span class="hm-tip-key">Name</span><span class="hm-tip-val">${{d.name}}</span></div>
                            <div class="hm-tip-row"><span class="hm-tip-key">Sector</span><span class="hm-tip-val">${{d.sector}}</span></div>
                            <div class="hm-tip-row"><span class="hm-tip-key">LTP</span><span class="hm-tip-val">₹${{d.price.toLocaleString('en-IN')}}</span></div>
                            <div class="hm-tip-row"><span class="hm-tip-key">${{currentTf}} Return</span><span class="hm-tip-val" style="color:${{ret>=0?'#00C076':'#EF4444'}}">${{sign}}${{ret.toFixed(2)}}%</span></div>
                            <div class="hm-tip-row"><span class="hm-tip-key">Dominance</span><span class="hm-tip-val">${{d.dominance}}%</span></div>
                        `;
                        tooltip.style.display = "block";
                    }});

                    tile.addEventListener("mousemove", (e) => {{
                        const rect = container.getBoundingClientRect();
                        const x = Math.min(e.clientX - rect.left + 15, rect.width - 230);
                        const y = Math.min(e.clientY - rect.top + 15, rect.height - 150);
                        tooltip.style.left = `${{x}}px`;
                        tooltip.style.top = `${{y}}px`;
                    }});

                    tile.addEventListener("mouseleave", () => {{
                        tooltip.style.display = "none";
                    }});

                    area.appendChild(tile);
                }});
            }}

            // Timeframe Selector Buttons
            document.querySelectorAll(".pill-btn[data-tf]").forEach(btn => {{
                btn.addEventListener("click", () => {{
                    document.querySelectorAll(".pill-btn[data-tf]").forEach(b => b.classList.remove("active"));
                    btn.classList.add("active");
                    currentTf = btn.getAttribute("data-tf");
                    render();
                }});
            }});

            // Universe Selector Buttons
            document.querySelectorAll(".pill-btn-univ").forEach(btn => {{
                btn.addEventListener("click", () => {{
                    document.querySelectorAll(".pill-btn-univ").forEach(b => b.classList.remove("active"));
                    btn.classList.add("active");
                    currentUniverse = parseInt(btn.getAttribute("data-univ") || "100", 10);
                    render();
                }});
            }});

            // Mode Selector Buttons
            const btnDom = document.getElementById("view_dom_btn");
            const btnSec = document.getElementById("view_sec_btn");

            if (btnDom && btnSec) {{
                btnDom.addEventListener("click", () => {{
                    btnDom.classList.add("active");
                    btnSec.classList.remove("active");
                    currentMode = "dominance";
                    render();
                }});
                btnSec.addEventListener("click", () => {{
                    btnSec.classList.add("active");
                    btnDom.classList.remove("active");
                    currentMode = "sector";
                    render();
                }});
            }}

            // Search Filter
            if (searchInput) {{
                searchInput.addEventListener("input", (e) => {{
                    const query = e.target.value.trim().toUpperCase();
                    const tiles = area.querySelectorAll(".tm-tile");
                    tiles.forEach(t => {{
                        const sym = t.getAttribute("data-sym") || "";
                        if (!query) {{
                            t.style.opacity = "1";
                            t.style.transform = "scale(1)";
                        }} else if (sym.includes(query)) {{
                            t.style.opacity = "1";
                            t.style.transform = "scale(1.03)";
                            t.style.boxShadow = "0 0 12px #38BDF8";
                            t.style.zIndex = "40";
                        }} else {{
                            t.style.opacity = "0.2";
                            t.style.transform = "scale(1)";
                            t.style.boxShadow = "none";
                            t.style.zIndex = "1";
                        }}
                    }});
                }});
            }}

            // Fullscreen Toggle
            if (fsBtn) {{
                fsBtn.addEventListener("click", () => {{
                    if (!document.fullscreenElement) {{
                        if (container.requestFullscreen) container.requestFullscreen().catch(fallbackFs);
                        else fallbackFs();
                    }} else {{
                        if (document.exitFullscreen) document.exitFullscreen();
                    }}
                }});
                function fallbackFs() {{
                    container.classList.toggle("css-fullscreen");
                    render();
                }}
                document.addEventListener("fullscreenchange", () => {{
                    const isFs = !!document.fullscreenElement;
                    fsBtn.classList.toggle("active", isFs);
                    render();
                }});
            }}

            window.addEventListener("resize", () => {{
                render();
            }});

            // Initial render
            setTimeout(render, 50);
            setTimeout(render, 300);
        }})();
    </script>
</body>
</html>"""
    return html_code

