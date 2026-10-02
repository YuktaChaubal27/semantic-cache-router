"""streamlit run dashboard/app.py"""
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings  # noqa: E402

st.set_page_config(page_title="Cache + routing", layout="wide")
st.title("Semantic cache and cost-aware routing")

df = pd.read_sql("SELECT * FROM requests", sqlite3.connect(settings()["paths"]["db"]))
if df.empty:
    st.info("No requests logged yet. Send some traffic or run scripts/benchmark.py.")
    st.stop()

df["time"] = pd.to_datetime(df["ts"], unit="s")
runs = ["(all)"] + sorted(df["run_id"].dropna().unique().tolist(), reverse=True)
run = st.sidebar.selectbox("Run", runs)
if run != "(all)":
    df = df[df["run_id"] == run]
ok = df[df["error"].isna()]

c = st.columns(6)
c[0].metric("Requests", len(df))
c[1].metric("Cost / request", f"${ok['cost_usd'].mean():.5f}")
c[2].metric("Cache hit rate", f"{ok['cache_hit'].mean():.0%}")
c[3].metric("p50 latency", f"{ok['latency_ms'].quantile(0.5):.0f} ms")
c[4].metric("p95 latency", f"{ok['latency_ms'].quantile(0.95):.0f} ms")
c[5].metric("Error rate", f"{df['error'].notna().mean():.1%}")

ok = ok.assign(path=ok.apply(
    lambda r: "cache" if r.cache_hit else (f"{r.route} (escalated)" if r.escalated else r.route), axis=1))

left, right = st.columns(2)
with left:
    st.subheader("Cost over time")
    st.line_chart(ok.set_index("time").resample("1min")["cost_usd"].sum())
    st.subheader("Route split")
    st.bar_chart(ok["path"].value_counts())
with right:
    st.subheader("Latency (p50 / p95) over time")
    lat = ok.set_index("time").resample("1min")["latency_ms"]
    st.line_chart(pd.DataFrame({"p50": lat.quantile(0.5), "p95": lat.quantile(0.95)}))
    st.subheader("Quality by path")
    st.bar_chart(ok.dropna(subset=["quality"]).groupby("path")["quality"].mean())

if {"baseline", "full"} <= set(df["mode"].dropna()):
    st.subheader("Baseline vs full")
    st.dataframe(ok.groupby("mode").agg(
        requests=("id", "count"), cost_per_request=("cost_usd", "mean"),
        p95_ms=("latency_ms", lambda s: s.quantile(0.95)), quality=("quality", "mean"),
    ))
