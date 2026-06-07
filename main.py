"""
API de Predicción SPI-30 — Sequía e Inundación en Zonas Agrícolas
Trabajo Integrador — Ciencia de Datos
v2 — adaptada al notebook PREDIC_1 con:
  · Objetivo: SPI-30 (en lugar de SPI-90)
  · Umbral calibrado: ±0.95 (F1-macro train = 0.607)
  · Métricas reales: RMSE 0.7234 | MAE 0.5677 | r = 0.6246 | Skill +0.170
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
from scipy import stats
from datetime import datetime, timedelta
import requests as req_lib
from io import StringIO
import os

app = FastAPI(title="SPI Agro API v2", version="2.0.0",
              description="Predicción SPI-30 — Pampa Húmeda, Argentina")

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["GET"], allow_headers=["*"])

cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=3, backoff_factor=0.2)
om = openmeteo_requests.Client(session=retry_session)

ZONAS = {
    "santafe":     {"lat":-31.6107,"lon":-60.6975,"nombre":"Santa Fe"},
    "cordoba":     {"lat":-31.4201,"lon":-64.1888,"nombre":"Córdoba"},
    "rosario":     {"lat":-32.9442,"lon":-60.6505,"nombre":"Rosario"},
    "parana":      {"lat":-31.7333,"lon":-60.5333,"nombre":"Paraná"},
    "buenosaires": {"lat":-34.6037,"lon":-58.3816,"nombre":"Buenos Aires"},
}

UMBRAL = 0.95   # calibrado sobre entrenamiento — F1 macro = 0.607

METRICAS = {
    "objetivo":"SPI-30","horizonte_dias":15,
    "RMSE":0.7234,"MAE":0.5677,"correlacion":0.6246,
    "skill_score":0.170,"accuracy":0.70,"f1_macro":0.58,
    "umbral_alerta":UMBRAL,"epocas_entrenamiento":14,"variables":8,
    "arquitectura":"LSTM bidireccional (64+32 unidades) — Huber Loss",
    "ponderacion":"extremos ×3.25 vs normales",
}

def calcular_spi(serie, ventana):
    acum = serie.rolling(ventana, min_periods=ventana).sum()
    spi  = acum.copy() * np.nan
    for m in range(1,13):
        mask = acum.index.month == m
        vals = acum[mask].dropna()
        if len(vals) < 10: continue
        mu, sg = vals.mean(), vals.std()
        if sg > 0: spi[mask] = (acum[mask] - mu) / sg
    return spi

def clasificar(v, umbral=UMBRAL):
    if np.isnan(v):       return {"categoria":"Sin datos",       "color":"#888888","nivel":0,"alerta":False}
    if v >= umbral*2.1:   return {"categoria":"Inundación severa","color":"#0B6E3F","nivel":3,"alerta":True}
    if v >= umbral:       return {"categoria":"Exceso hídrico",  "color":"#27A567","nivel":2,"alerta":True}
    if v > -umbral:       return {"categoria":"Normal",          "color":"#3B8BD4","nivel":0,"alerta":False}
    if v >= -umbral*1.58: return {"categoria":"Sequía moderada", "color":"#E24B4A","nivel":-2,"alerta":True}
    return                       {"categoria":"Sequía severa",   "color":"#9B2020","nivel":-3,"alerta":True}

def get_clima(lat, lon, dias=730):
    fin  = datetime.now().strftime("%Y-%m-%d")
    ini  = (datetime.now()-timedelta(days=dias)).strftime("%Y-%m-%d")
    p = {"latitude":lat,"longitude":lon,
         "daily":["precipitation_sum","temperature_2m_max","temperature_2m_min"],
         "start_date":ini,"end_date":fin,"timezone":"America/Argentina/Buenos_Aires"}
    r = om.weather_api("https://archive-api.open-meteo.com/v1/archive",params=p)[0]
    d = r.Daily()
    idx = pd.date_range(
        start=pd.to_datetime(d.Time(),unit="s",utc=True),
        end=pd.to_datetime(d.TimeEnd(),unit="s",utc=True),
        freq=pd.Timedelta(seconds=d.Interval()),inclusive="left").tz_localize(None)
    df = pd.DataFrame({"precip_mm":d.Variables(0).ValuesAsNumpy(),
                       "temp_max":d.Variables(1).ValuesAsNumpy(),
                       "temp_min":d.Variables(2).ValuesAsNumpy()},index=idx)
    df["precip_mm"] = df["precip_mm"].clip(lower=0).fillna(0)
    df["temp_media"] = (df["temp_max"]+df["temp_min"])/2
    return df

def get_oni():
    try:
        r = req_lib.get("https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt",timeout=10)
        oni = pd.read_csv(StringIO(r.text),sep=r'\s+',skiprows=1,
                          names=["season","year","total","anom"])
        S2M={"DJF":1,"JFM":2,"FMA":3,"MAM":4,"AMJ":5,"MJJ":6,
             "JJA":7,"JAS":8,"ASO":9,"SON":10,"OND":11,"NDJ":12}
        oni["month"]=oni["season"].map(S2M)
        oni=oni.dropna(subset=["month"])
        oni["month"]=oni["month"].astype(int); oni["year"]=oni["year"].astype(int)
        oni["date"]=pd.to_datetime(oni[["year","month"]].assign(day=1))
        oni=oni.set_index("date")[["anom"]].rename(columns={"anom":"ONI"})
        idx=pd.date_range(oni.index.min(),oni.index.max()+pd.DateOffset(months=1),freq="D")
        return oni.reindex(idx).interpolate("linear")["ONI"]
    except: return pd.Series(dtype=float)

modelo_lstm = None
def cargar_modelo():
    global modelo_lstm
    if os.path.exists("modelo_spi_lstm.keras"):
        try:
            import tensorflow as tf
            modelo_lstm = tf.keras.models.load_model("modelo_spi_lstm.keras")
            print("Modelo LSTM cargado.")
        except Exception as e:
            print(f"No se pudo cargar: {e}")
cargar_modelo()

def predecir(serie_spi30, horizonte=15):
    if modelo_lstm is not None:
        from sklearn.preprocessing import MinMaxScaler
        sc = MinMaxScaler()
        v  = sc.fit_transform(serie_spi30.dropna().values.reshape(-1,1))
        if len(v) >= 90:
            seq = v[-90:].reshape(1,90,1)
            out = []
            for _ in range(horizonte):
                p = modelo_lstm.predict(seq,verbose=0)[0,0]
                out.append(float(np.clip(sc.inverse_transform([[p]])[0,0],-3.5,3.5)))
                seq = np.roll(seq,-1,axis=1); seq[0,-1,0]=p
            return out
    s = serie_spi30.dropna()
    mu = float(s.iloc[-30:].mean())
    td = float(np.clip((s.iloc[-1]-s.iloc[-15])/15,-0.04,0.04))
    ult= float(s.iloc[-1])
    return [round(float(np.clip(ult*0.65+mu*0.35+td*i,-3.5,3.5)),3) for i in range(1,horizonte+1)]

# ── ENDPOINTS ───────────────────────────────────────────────

@app.get("/")
def root():
    return {"api":"SPI Agro v2","objetivo":"SPI-30","horizonte":"15 días",
            "endpoints":["/zonas","/spi/{zona}","/historial/{zona}","/modelo","/estado"]}

@app.get("/zonas")
def zonas():
    return {"zonas":[{"id":k,"nombre":v["nombre"],"lat":v["lat"],"lon":v["lon"]}
                     for k,v in ZONAS.items()]}

@app.get("/modelo")
def modelo():
    return {"metricas":METRICAS,"modelo_cargado":modelo_lstm is not None,
            "variables":["SPI_30","SPI_90","precip_acum_30d","temp_media",
                         "ONI","sm_7_28cm","ET0","balance_hidrico_30d"]}

@app.get("/estado")
def estado():
    return {"estado":"operativo","modelo":"LSTM" if modelo_lstm else "estadístico",
            "objetivo":"SPI-30","umbral":UMBRAL,"ts":datetime.now().isoformat()}

@app.get("/spi/{zona_id}")
def spi(zona_id:str):
    zona_id=zona_id.lower()
    if zona_id not in ZONAS: return {"error":f"Zona '{zona_id}' no encontrada."}
    z=ZONAS[zona_id]
    try: df=get_clima(z["lat"],z["lon"],730)
    except Exception as e: return {"error":str(e)}

    df["SPI_30"]=calcular_spi(df["precip_mm"],30)
    df["SPI_90"]=calcular_spi(df["precip_mm"],90)
    df["precip_acum_30d"]=df["precip_mm"].rolling(30).sum()

    oni=get_oni()
    df["ONI"]=oni.reindex(df.index).ffill().fillna(0) if not oni.empty else 0.0

    dfv=df.dropna(subset=["SPI_30"])
    if dfv.empty: return {"error":"Datos insuficientes."}

    s30=float(dfv["SPI_30"].iloc[-1])
    s90=float(dfv["SPI_90"].iloc[-1])
    p30=float(dfv["precip_acum_30d"].iloc[-1])
    tmp=float(df["temp_media"].iloc[-1])
    oni_v=float(df["ONI"].iloc[-1])

    if oni_v>=1.5: enso="El Niño fuerte"
    elif oni_v>=0.5: enso="El Niño moderado"
    elif oni_v<=-1.5: enso="La Niña fuerte"
    elif oni_v<=-0.5: enso="La Niña moderada"
    else: enso="Neutro"

    persist=float(dfv["SPI_30"].iloc[-16]) if len(dfv)>16 else s30
    preds=predecir(dfv["SPI_30"])
    pred15=preds[-1]

    tend="empeora" if pred15<s30-0.1 else ("mejora" if pred15>s30+0.1 else "estable")
    spark=[{"fecha":str(d.date()),"spi":round(float(v),3)}
           for d,v in dfv["SPI_30"].iloc[-90:].items() if not np.isnan(v)]

    return {
        "zona":z["nombre"],"zona_id":zona_id,
        "actualizado":datetime.now().isoformat(),
        "condicion_actual":{
            "spi_30":round(s30,3),"spi_90":round(s90,3),
            "precip_30d_mm":round(p30,1),"temp_media_c":round(tmp,1),
            "oni":round(oni_v,2),"fase_enso":enso,
            **clasificar(s30)},
        "prediccion_15_dias":[
            {"dia":i+1,"fecha":(datetime.now()+timedelta(days=i+1)).strftime("%Y-%m-%d"),
             "spi_predicho":round(v,3),**clasificar(v)}
            for i,v in enumerate(preds)],
        "persistencia_spi":round(persist,3),
        "tendencia":tend,"umbral_usado":UMBRAL,
        "modelo_usado":"LSTM" if modelo_lstm else "estadístico",
        "sparkline_90d":spark,
    }

@app.get("/historial/{zona_id}")
def historial(zona_id:str,dias:int=365):
    zona_id=zona_id.lower()
    if zona_id not in ZONAS: return {"error":f"Zona '{zona_id}' no encontrada."}
    z=ZONAS[zona_id]; dias=min(dias,730)
    try: df=get_clima(z["lat"],z["lon"],dias+40)
    except Exception as e: return {"error":str(e)}
    df["SPI_30"]=calcular_spi(df["precip_mm"],30)
    dfv=df.dropna(subset=["SPI_30"]).iloc[-dias:]
    return {"zona":z["nombre"],"zona_id":zona_id,"objetivo":"SPI-30","dias":len(dfv),
            "historial":[{"fecha":str(d.date()),"spi":round(float(v),3),**clasificar(v)}
                         for d,v in dfv["SPI_30"].items() if not np.isnan(v)]}
