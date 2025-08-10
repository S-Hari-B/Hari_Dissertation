#!/usr/bin/env python
# coding: utf-8


# Imports
from dotenv import load_dotenv
import os
import numpy as np
import pandas as pd
import requests
from autogluon.tabular import TabularPredictor
from dotenv import load_dotenv
import os, time, datetime as dt
import requests
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openai import OpenAI
import json
import warnings


load_dotenv()
AV_KEY = os.environ["AV_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


client = OpenAI()
BASE_URL = "https://www.alphavantage.co/query"


pd.set_option("display.max_colwidth", 120)
pd.set_option("display.expand_frame_repr", False)


ticker_to_name = {
    "TSLA": "Tesla","F": "Ford","GM": "GM","RIVN": "Rivian","LCID": "Lucid",
    "TM": "Toyota","HMC": "Honda","NIO": "NIO","XPEV": "XPeng","STLA": "Stellantis",
    "PSNY": "Polestar","LI": "Li Auto","RACE": "Ferrari","LCII": "LCI Industries",
    "ALV": "Autoliv",
}


# Agent 1 prediction

# Agent 1 (prediction-only, top-k ensemble confidence)
warnings.filterwarnings("ignore", category=UserWarning, module="fastai.learner")


# Alpha-Vantage helper
def _av_get(function, **params):
    params |= {"apikey": AV_KEY, "function": function}
    r = requests.get("https://www.alphavantage.co/query", params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def get_market_cap_now(ticker):
    v = _av_get("OVERVIEW", symbol=ticker).get("MarketCapitalization")
    return None if v in (None, "") else float(v)


def pull_quarterly_fundamentals(ticker):
    income = _av_get("INCOME_STATEMENT", symbol=ticker)
    bal = _av_get("BALANCE_SHEET", symbol=ticker)
    try:
        inc, bal = income["quarterlyReports"][0], bal["quarterlyReports"][0]
    except (KeyError, IndexError):
        return pd.DataFrame()


    f = lambda row, k: np.nan if (v:=row.get(k)) in (None,"") else float(v)
    df = pd.DataFrame([{
        "Quarter": pd.to_datetime(inc["fiscalDateEnding"]),
        "Total_Revenue": f(inc,"totalRevenue"),
        "Net_Income": f(inc,"netIncome"),
        "EPS": f(inc,"eps"),
        "Total_Debt": f(bal,"totalLiabilities"),
        "Total_Assets": f(bal,"totalAssets"),
    }])
    df["Cash_to_Debt"] = df["Total_Assets"]/df["Total_Debt"].replace(0,np.nan)
    df["NetMargin"] = df["Net_Income"]/df["Total_Revenue"]
    df["Debt_Ratio"] = df["Total_Debt"]/df["Total_Assets"]
    df["Asset_Turnover"] = df["Total_Revenue"]/df["Total_Assets"]
    df[["Revenue_pctchg","EPS_pctchg","Income_Growth"]] = 0
    return df


# Predictor and ensemble helpers
predictor = TabularPredictor.load("agent1_model")
features = predictor.feature_metadata.get_features()
unique_companies = [c[len("Company_"):] for c in features if c.startswith("Company_")]


def _top_k_models(pred, k=5):
    lb = pred.leaderboard(silent=True)
    return (lb.sort_values("score_val", ascending=False)
              .head(k)["model"].tolist())


_TOP_MODELS = _top_k_models(predictor, k=5)


def _ensemble_stats_k(feat_df):
    preds = [np.expm1(predictor.predict(feat_df, model=m).iloc[0])
             for m in _TOP_MODELS]
    preds = np.asarray(preds)
    return preds.mean(), preds.std(ddof=0)


def predict_agent1(tickers):
    out = {}
    for tk in tickers:
        feat = pull_quarterly_fundamentals(tk)
        if feat.empty: continue
        for c in unique_companies:
            feat[f"Company_{c}"] = int(ticker_to_name[tk]==c)
        for col in set(features)-set(feat.columns):
            feat[col] = 0 if col.startswith("Company_") else np.nan
        feat = feat[features]


        pred, spread = _ensemble_stats_k(feat)
        cur = get_market_cap_now(tk)
        pct = None if cur in (None,0) else (pred-cur)/cur*100
        conf = float(np.clip(1 - spread/pred, 0, 1))
        out[tk] = {"pred_cap": pred,"cur_cap": cur,"pct_diff": pct,"confidence": conf}
    return out




# Agent 2 Prediction

tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
finbert = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
finbert.eval()
labels = ["negative", "neutral", "positive"]
label_to_score = {"negative": -1, "neutral": 0, "positive": 1}


def alpha_news_window(ticker: str,
                      from_dt: dt.datetime,
                      to_dt: dt.datetime,
                      limit: int = 5) -> list[str]:
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "time_from": from_dt.strftime("%Y%m%dT%H%M"),
        "time_to": to_dt.strftime("%Y%m%dT%H%M"),
        "sort": "LATEST",
        "apikey": AV_KEY,
    }
    r = requests.get(url, params=params, timeout=10)
    if r.status_code == 503:
        raise RuntimeError("Alpha Vantage news quota exhausted")
    r.raise_for_status()
    return [item["title"] for item in r.json().get("feed", [])[:limit]]


def classify_sentiment(text: str) -> dict:
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, max_length=512)
    with torch.no_grad():
        out = finbert(**inputs)
    probs = torch.nn.functional.softmax(out.logits, dim=1).numpy()[0]
    idx = int(np.argmax(probs))
    return {"score": label_to_score[labels[idx]],
            "conf": float(probs[idx])}


def predict_agent2(tickers, lookback_days=7, top_n=5):
    res = {}
    today = dt.datetime.utcnow()
    window_start = today - dt.timedelta(days=lookback_days)


    for tk in tickers:
        heads = alpha_news_window(tk, window_start, today, limit=top_n)
        if not heads:
            continue


        scores, confs = [], []
        for h in heads:
            d = classify_sentiment(h)
            scores.append(d["score"])
            confs.append(d["conf"])
            time.sleep(0.05)


        s = float(np.average(scores, weights=np.square(confs)))
        res[tk] = {"sentiment": s, "mean_conf": float(np.mean(confs)), "n": len(heads)}
    return res


# Agent 3 prediction

def alpha_headlines(ticker: str,
                    top_n: int = 10) -> list[str]:
    """
    Return up to `top_n` latest headlines for a single ticker.
    Very short titles (<30 chars) are skipped.
    """
    params = {"function": "NEWS_SENTIMENT",
              "tickers": ticker,
              "apikey": AV_KEY}
    resp = requests.get(BASE_URL, params=params, timeout=20)
    resp.raise_for_status()
    return [item["title"]
            for item in resp.json().get("feed", [])[:top_n]
            if len(item["title"]) > 30]


def gpt_sentiment(headline: str,
                  company: str,
                  temperature: float = 0.3) -> tuple[float, str]:
    """
    Return (score, reason) where score ∈ [-1, 1].
    Falls back to (np.nan, 'fail') on any API error.
    """
    prompt = f"""
You are an investor-sentiment analyst.

Company: {company}
Text: "{headline}"

On a scale of -1 (very negative) to +1 (very positive) give a numeric score
and a one-word rationale, as JSON:
{{"score": <number>, "reason": "<word>"}}
""".strip()
    try:
        rsp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user","content": prompt}],
            temperature=temperature
        )
        data = json.loads(rsp.choices[0].message.content)
        return float(data["score"]), str(data["reason"])
    except Exception:
        return np.nan, "fail"  # ← surface the real error


def predict_agent3(tickers: list[str],
                   ticker_to_name: dict,
                   top_n: int = 8,
                   pause: float = 1.0) -> dict:
    """
    Return {ticker:
        {"sentiment": float, "confidence": float, "n": int}}
    Sentiment = mean(score); confidence = 1 – stdev(score) (clipped 0-1).
    """
    results = {}
    for tk in tickers:
        headlines = alpha_headlines(tk, top_n=top_n)
        if not headlines:
            continue


        scores = []
        for h in headlines:
            sc, _ = gpt_sentiment(h, ticker_to_name[tk])
            scores.append(sc)
            time.sleep(pause)  # 60 calls/min < AV limit


        scores = np.array([s for s in scores if not np.isnan(s)])
        if scores.size == 0:
            continue
        avg = float(scores.mean())
        stdev = float(scores.std(ddof=0))
        conf = max(0.0, 1.0 - stdev)  # simple dispersion proxy
        results[tk] = {"sentiment": avg,
                       "confidence": conf,
                       "n": int(scores.size)}
    return results


# Unified Agent

# panel builder
def build_panel(tickers, lookback=7, top_n=5):
    a1 = predict_agent1(tickers)
    a2 = predict_agent2(tickers, lookback_days=lookback, top_n=top_n)
    a3 = predict_agent3(tickers, ticker_to_name, top_n=top_n)


    rows=[]
    for tk in tickers:
        if tk not in a1 or tk not in a2 or tk not in a3: continue
        rows.append({
            "ticker": tk,
            "pred_cap": a1[tk]["pred_cap"],
            "pct_diff": a1[tk]["pct_diff"],
            "confidence": a1[tk]["confidence"],
            "news_sent": a2[tk]["sentiment"],
            "news_n": a2[tk]["n"],
            "gpt_sent": a3[tk]["sentiment"],
            "gpt_conf": a3[tk]["confidence"],
        })
    return pd.DataFrame(rows)


import json

# LLM ranking
def rank_companies(df, model="gpt-4o", temperature=0.2):
    if df.empty:
        raise RuntimeError("panel empty")


    sig_lines = [
        (f"{r.ticker}: Δcap={r.pct_diff:+.1f}% (conf={r.confidence:.2f}) "
         f"news={r.news_sent:+.2f} (n={r.news_n}) "
         f"gpt={r.gpt_sent:+.2f} (conf={r.gpt_conf:.2f})")
        for r in df.itertuples()
    ]


    prompt = (
        "You are an equity meta-analyst.\n\n"
        "Signals per ticker:\n"
        "  • Δcap  and its confidence\n"
        "  • news_sent with headline count\n"
        "  • gpt_sent with confidence\n\n"
        f"Rank ALL {len(df)} companies from best investment to worst, "
        "weighting higher-confidence signals more.\n\n"
        "Return JSON {\"results\": [{ticker,rank,rationale}…]} only.\n\n"
        "Signals:\n" + "\n".join(sig_lines)
    )


    rsp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[{"role": "system","content": "You are an equity analyst."},
                  {"role": "user","content": prompt}]
    ).choices[0].message.content


    rows = json.loads(rsp)["results"]
    for i,d in enumerate(rows,1):
        d.setdefault("rank", i)
    return sorted(rows, key=lambda d:d["rank"])


pd.set_option("display.max_colwidth", None)  # stop truncating long text










