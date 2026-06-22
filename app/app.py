"""
Search Engine Pantai Karimunjawa
=================================
Dual Mode:
  - STANDALONE MODE (default / Streamlit Cloud): semua logika pandas + IndoBERT
    dijalankan langsung di dalam app ini.
  - API MODE (Docker): set env var API_URL=http://api:8000, maka app memanggil
    FastAPI backend untuk semua operasi data & prediksi.
"""
import os
import re

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ─── Konfigurasi Mode ────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", None)  # None → Standalone, URL string → API Mode
IS_API_MODE = API_URL is not None

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Search Engine Pantai Karimunjawa",
    page_icon="🏖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Helper: Standalone Mode ──────────────────────────────────────────────────

@st.cache_data
def load_data() -> pd.DataFrame:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "..", "data", "labelling", "label_data.csv")
    df = pd.read_csv(data_path)
    df["text_processed"] = (
        df["text"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[^\w\s]", " ", regex=True)
    )
    return df


@st.cache_resource
def load_indobert():
    """Load IndoBERT hanya di Standalone Mode (tidak di-load saat API Mode)."""
    from transformers import pipeline as hf_pipeline
    return hf_pipeline(
        "sentiment-analysis",
        model="w11wo/indonesian-roberta-base-sentiment-classifier",
        device=-1,
        truncation=True,
        max_length=512,
    )


def standalone_predict(text: str) -> dict:
    clf = load_indobert()
    result = clf(text)[0]
    label_map = {"pos": "positive", "neg": "negative", "neu": "neutral"}
    label = label_map.get(result["label"].lower(), result["label"].lower())
    return {"sentimen": label, "confidence": round(result["score"], 4)}


def standalone_filter(df: pd.DataFrame, keyword, sentimen, min_r, max_r) -> pd.DataFrame:
    filtered = df[(df["stars"] >= min_r) & (df["stars"] <= max_r)].copy()
    if sentimen and sentimen != "Semua":
        filtered = filtered[filtered["sentimen"] == sentimen.lower()]
    if keyword:
        filtered = filtered[filtered["text_processed"].str.contains(keyword.lower(), na=False)]
    return filtered


def standalone_rankings(df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    stats = df.groupby("title").agg(
        rating_mean=("stars", "mean"),
        review_count=("stars", "count"),
        positive_count=("sentimen", lambda x: (x == "positive").sum()),
    ).reset_index()
    stats["positive_percentage"] = (stats["positive_count"] / stats["review_count"] * 100).round(2)
    stats["score"] = (
        stats["rating_mean"]
        * (stats["positive_percentage"] / 100)
        * np.log1p(stats["review_count"])
    ).round(4)
    return stats.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)


# ─── Helper: API Mode ─────────────────────────────────────────────────────────

def api_predict(text: str) -> dict:
    try:
        r = requests.post(f"{API_URL}/predict", json={"text": text}, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"❌ Gagal menghubungi API: {e}")
        return {}


def api_search(keyword, sentimen, min_r, max_r) -> dict:
    params = {"min_rating": min_r, "max_rating": max_r, "limit": 200}
    if keyword:
        params["keyword"] = keyword
    if sentimen and sentimen != "Semua":
        params["sentimen"] = sentimen.lower()
    try:
        r = requests.get(f"{API_URL}/search", params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"❌ Gagal menghubungi API: {e}")
        return {"total": 0, "results": []}


def api_rankings(top_n: int = 3) -> list:
    try:
        r = requests.get(f"{API_URL}/rankings", params={"top_n": top_n}, timeout=15)
        r.raise_for_status()
        return r.json().get("rankings", [])
    except Exception as e:
        st.error(f"❌ Gagal menghubungi API: {e}")
        return []


# ─── Load Data ────────────────────────────────────────────────────────────────

if not IS_API_MODE:
    df = load_data()
    sentiment_list = ["Semua"] + sorted(df["sentimen"].dropna().unique().tolist())
else:
    # Ambil daftar sentimen statis karena ini konstan
    sentiment_list = ["Semua", "positive", "neutral", "negative"]
    df = None  # tidak dipakai di API mode

# ─── Sidebar ─────────────────────────────────────────────────────────────────

st.sidebar.title("🔍 Filter Pencarian")

st.sidebar.subheader("⭐ Filter Rating")
min_rating, max_rating = st.sidebar.slider(
    "Pilih rentang rating:",
    min_value=1, max_value=5, value=(1, 5),
)

st.sidebar.subheader("😊 Filter Sentimen")
selected_sentiment = st.sidebar.selectbox("Pilih sentimen:", sentiment_list)

st.sidebar.subheader("📝 Pencarian Kata Kunci")
search_query = st.sidebar.text_input("Masukkan kata kunci:")

# ─── Live Predict ─────────────────────────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Prediksi Sentimen")
st.sidebar.caption("Masukkan teks review untuk diprediksi langsung oleh IndoBERT.")
predict_input = st.sidebar.text_area("Teks review:", height=100, placeholder="Contoh: Pantai yang sangat bersih dan indah!")
predict_btn = st.sidebar.button("🔍 Prediksi", use_container_width=True)

if predict_btn and predict_input.strip():
    with st.sidebar:
        with st.spinner("Menganalisis sentimen..."):
            if IS_API_MODE:
                pred = api_predict(predict_input)
            else:
                pred = standalone_predict(predict_input)

        if pred:
            emoji_map = {"positive": "😊", "neutral": "😐", "negative": "😞"}
            color_map = {"positive": "green", "neutral": "orange", "negative": "red"}
            sent = pred.get("sentimen", "unknown")
            conf = pred.get("confidence", 0)
            st.markdown(
                f"**Hasil:** :{color_map.get(sent, 'gray')}[{emoji_map.get(sent, '🤔')} {sent.capitalize()}]  \n"
                f"**Keyakinan:** `{conf:.1%}`"
            )
elif predict_btn:
    st.sidebar.warning("Masukkan teks terlebih dahulu.")

# ─── Main Content ─────────────────────────────────────────────────────────────

st.title("🏖️ Search Engine Pantai Karimunjawa")
st.markdown("Cari informasi pantai di Karimunjawa berdasarkan rating, sentimen, dan kata kunci.")

# Ambil data berdasarkan mode
if IS_API_MODE:
    search_result = api_search(search_query, selected_sentiment, min_rating, max_rating)
    total_found = search_result.get("total", 0)
    results_raw = search_result.get("results", [])
    filtered_df = pd.DataFrame(results_raw) if results_raw else pd.DataFrame(
        columns=["title", "stars", "text", "sentimen"]
    )
else:
    filtered_df = standalone_filter(df, search_query, selected_sentiment, min_rating, max_rating)
    total_found = len(filtered_df)

# ─── Statistik ────────────────────────────────────────────────────────────────

st.subheader(f"📊 Hasil Pencarian: {total_found} review ditemukan")

col1, col2, col3 = st.columns(3)
with col1:
    avg_rating = filtered_df["stars"].mean() if not filtered_df.empty else 0
    st.metric("Rating Rata-rata", f"{avg_rating:.2f} ⭐")
with col2:
    pos_count = len(filtered_df[filtered_df["sentimen"] == "positive"]) if not filtered_df.empty else 0
    st.metric("Review Positif", pos_count)
with col3:
    neg_count = len(filtered_df[filtered_df["sentimen"] == "negative"]) if not filtered_df.empty else 0
    st.metric("Review Negatif", neg_count)

# ─── TOP 3 Pantai Terbaik ────────────────────────────────────────────────────

st.markdown("---")
st.subheader("🏆 TOP 3 Pantai Terbaik")

if not filtered_df.empty:
    if IS_API_MODE:
        top_3_data = api_rankings(top_n=3)
        # Konversi ke format yang sesuai untuk tampilan
        top_3_display = [
            {
                "name": item["name"],
                "rating_mean": item["rating_mean"],
                "positive_percentage": item["positive_percentage"],
                "review_count": item["review_count"],
            }
            for item in top_3_data
        ]
    else:
        top_3_df = standalone_rankings(filtered_df, top_n=3)
        top_3_display = top_3_df.rename(columns={"title": "name"}).to_dict("records")

    cols = st.columns(3)
    colors = ["#FFD700", "#C0C0C0", "#CD7F32"]
    rank_icons = ["🥇", "🥈", "🥉"]

    for idx, item in enumerate(top_3_display[:3]):
        with cols[idx]:
            st.markdown(f"<h2 style='text-align:center'>{rank_icons[idx]}</h2>", unsafe_allow_html=True)
            st.markdown(
                f"<h3 style='text-align:center;color:{colors[idx]}'>{item['name']}</h3>",
                unsafe_allow_html=True,
            )
            st.metric("⭐ Rating", f"{item['rating_mean']:.2f}")
            st.metric("😊 Positif", f"{item['positive_percentage']:.1f}%")
            st.metric("📝 Review", int(item["review_count"]))

# ─── 5 Pantai Rekomendasi per Sentimen ───────────────────────────────────────

st.markdown("---")
st.subheader("👍 5 Pantai Rekomendasi Berdasarkan Sentimen")

if "show_reviews_for" not in st.session_state:
    st.session_state.show_reviews_for = None

if not filtered_df.empty:
    all_sentiments = ["positive", "neutral", "negative"]
    tabs = st.tabs(
        [f"{s.capitalize()} ({len(filtered_df[filtered_df['sentimen'] == s])})" for s in all_sentiments]
    )

    for tab, sentiment in zip(tabs, all_sentiments):
        with tab:
            sentiment_df = filtered_df[filtered_df["sentimen"] == sentiment]

            if not sentiment_df.empty:
                # Ranking per sentimen
                s_stats = sentiment_df.groupby("title").agg(
                    rating_mean=("stars", "mean"),
                    review_count=("stars", "count"),
                ).reset_index()
                s_stats["score"] = s_stats["rating_mean"] * np.log1p(s_stats["review_count"])
                top_5 = s_stats.sort_values("score", ascending=False).head(5)

                cols = st.columns(5)
                for idx, (_, row) in enumerate(top_5.iterrows()):
                    with cols[idx]:
                        with st.container():
                            st.markdown(
                                f"<div style='text-align:center;height:60px;display:flex;"
                                f"align-items:center;justify-content:center'>"
                                f"<strong>{row['title']}</strong></div>",
                                unsafe_allow_html=True,
                            )
                            stars_str = "⭐" * int(round(row["rating_mean"]))
                            st.markdown(
                                f"<div style='text-align:center;margin:10px 0'>"
                                f"{stars_str}<br><small>({row['rating_mean']:.1f})</small></div>",
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                f"<div style='text-align:center;margin:10px 0'>"
                                f"📝 {int(row['review_count'])} review</div>",
                                unsafe_allow_html=True,
                            )
                            c1, c2, c3 = st.columns([1, 2, 1])
                            with c2:
                                btn_key = f"{sentiment}_{row['title']}_{idx}"
                                if st.button("⬇️", key=btn_key, use_container_width=True,
                                             help=f"Lihat review {row['title']}"):
                                    st.session_state.show_reviews_for = (row["title"], sentiment)

                # Tampilkan review detail
                if st.session_state.show_reviews_for:
                    sel_beach, sel_sent = st.session_state.show_reviews_for
                    if sel_sent == sentiment and sel_beach in top_5["title"].values:
                        st.markdown("---")
                        h1, h2 = st.columns([0.9, 0.1])
                        with h1:
                            st.subheader(f"📋 Review untuk {sel_beach} ({sel_sent})")
                        with h2:
                            if st.button("✖️", key=f"close_{sel_beach}_{sentiment}"):
                                st.session_state.show_reviews_for = None
                                st.rerun()

                        reviews_df = filtered_df[
                            (filtered_df["title"] == sel_beach)
                            & (filtered_df["sentimen"] == sel_sent)
                        ][["stars", "text"]].head(10)

                        if not reviews_df.empty:
                            st.dataframe(
                                reviews_df,
                                column_config={
                                    "stars": st.column_config.NumberColumn("⭐ Rating", format="%d ⭐"),
                                    "text": st.column_config.TextColumn("📝 Review", width="large"),
                                },
                                use_container_width=True,
                                hide_index=True,
                            )
                            total_rev = len(
                                filtered_df[
                                    (filtered_df["title"] == sel_beach)
                                    & (filtered_df["sentimen"] == sel_sent)
                                ]
                            )
                            st.caption(f"Menampilkan {len(reviews_df)} dari {total_rev} review")
            else:
                st.info(f"Tidak ada data untuk sentimen {sentiment}.")
else:
    st.warning("🚫 Tidak ada data yang sesuai dengan filter. Coba ubah kriteria filter Anda.")

# ─── Footer ───────────────────────────────────────────────────────────────────

st.markdown("---")
mode_label = f"🔗 API Mode (`{API_URL}`)" if IS_API_MODE else "💻 Standalone Mode"
st.markdown(
    f"**Tips:** Gunakan filter di sidebar untuk mempersempit hasil pencarian.  \n"
    f"*Mode aktif: {mode_label}*"
)

# ─── CSS ──────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
div.stButton > button {
    background: transparent !important;
    border: 1px solid #444 !important;
    border-radius: 50% !important;
    width: 40px !important;
    height: 40px !important;
    min-width: auto !important;
    margin: 5px auto !important;
    padding: 0 !important;
}
div.stButton > button:hover {
    background: #1e1e2e !important;
    border-color: #1E90FF !important;
    transform: scale(1.1);
    transition: all 0.2s ease;
}
</style>
""", unsafe_allow_html=True)