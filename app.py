"""
Emotion / Sentiment Analysis - Streamlit UI
-------------------------------------------
Uses the SAME pipeline as your notebook:
lowercase -> remove punctuation -> remove numbers -> remove emojis
-> remove stop words -> TF-IDF -> Logistic Regression

Run:  streamlit run app.py
Keep train.txt in the same folder as this file.
"""

import string
from pathlib import Path

import nltk
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Emotion Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_FILE = Path(__file__).parent / "train.txt"

# Emoji + colour for each emotion in your dataset
EMOTIONS = {
    "joy":      {"emoji": "😄", "color": "#FFD166", "msg": "Positive and happy vibes!"},
    "love":     {"emoji": "😍", "color": "#FF6FB5", "msg": "Warm, caring and affectionate."},
    "surprise": {"emoji": "😲", "color": "#4CC9F0", "msg": "Something unexpected happened!"},
    "sadness":  {"emoji": "😢", "color": "#6C8EF5", "msg": "Feeling low or hurt."},
    "anger":    {"emoji": "😡", "color": "#FF5C5C", "msg": "Frustration or annoyance detected."},
    "fear":     {"emoji": "😨", "color": "#B388FF", "msg": "Worry, anxiety or fear."},
}

EXAMPLES = {
    "😄 Happy":    "i am so happy today because i passed my exam with great marks",
    "😢 Sad":      "i feel so lonely and hopeless after everything that happened",
    "😡 Angry":    "i am furious that they ignored my complaint again",
    "😨 Scared":   "i am terrified about the results coming out tomorrow",
    "😍 Loving":   "i feel so much love and care for my family",
    "😲 Surprised": "i was shocked and amazed when i saw the surprise party",
}

# ----------------------------------------------------------------------------
# Custom CSS  (this makes the UI beautiful)
# ----------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

.stApp {
    background: radial-gradient(circle at 15% 10%, #2b1a5e 0%, #12122b 45%, #0b0b1a 100%);
    color: #EDEDFF;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 1150px; }

/* Hero */
.hero {
    text-align: center; padding: 2.2rem 1rem 1.2rem 1rem;
}
.hero h1 {
    font-size: 3rem; font-weight: 800; margin: 0;
    background: linear-gradient(90deg, #FFD166, #FF6FB5, #8E7CFF, #4CC9F0);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p { color: #B9B9DD; font-size: 1.05rem; margin-top: .4rem; }

/* Glass cards */
.card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 20px; padding: 1.4rem 1.6rem;
    backdrop-filter: blur(10px);
    box-shadow: 0 10px 35px rgba(0,0,0,.35);
}

/* Result card */
.result {
    text-align: center; border-radius: 22px; padding: 1.8rem 1rem;
    border: 2px solid var(--c); background: rgba(255,255,255,0.05);
    box-shadow: 0 0 40px -8px var(--c);
    animation: pop .5s ease;
}
.result .emoji { font-size: 5rem; line-height: 1.1; }
.result .label {
    font-size: 2rem; font-weight: 800; color: var(--c);
    text-transform: capitalize; margin-top: .3rem;
}
.result .conf { font-size: 1rem; color: #D5D5F2; }
.result .msg  { color: #A9A9D0; margin-top: .4rem; font-size: .95rem; }
@keyframes pop { from {transform: scale(.9); opacity: 0;} to {transform: scale(1); opacity: 1;} }

/* Probability bars */
.bar-row { display: flex; align-items: center; gap: .7rem; margin: .55rem 0; }
.bar-name { width: 110px; font-weight: 600; text-transform: capitalize; }
.bar-track { flex: 1; height: 14px; background: rgba(255,255,255,.1); border-radius: 99px; overflow: hidden; }
.bar-fill  { height: 100%; border-radius: 99px; }
.bar-pct   { width: 60px; text-align: right; font-weight: 600; color: #D5D5F2; }

/* Cleaned text chip */
.chip {
    display: inline-block; background: rgba(255,255,255,.08);
    border-radius: 10px; padding: .45rem .8rem; color: #CFCFF5; font-size: .9rem;
}

/* Text area + buttons */
.stTextArea textarea {
    background: rgba(255,255,255,0.07) !important; color: #fff !important;
    border-radius: 14px !important; border: 1px solid rgba(255,255,255,.18) !important;
    font-size: 1.05rem !important;
}
.stButton > button, .stDownloadButton > button {
    border-radius: 12px; border: none; font-weight: 600; color: white;
    background: linear-gradient(90deg, #8E7CFF, #FF6FB5);
    padding: .55rem 1.2rem; transition: transform .15s ease, box-shadow .15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-2px); box-shadow: 0 8px 22px rgba(142,124,255,.45); color: white;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,.06); border-radius: 12px; padding: .5rem 1.1rem;
}
.stTabs [aria-selected="true"] { background: rgba(142,124,255,.35) !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(12,12,32,.92); border-right: 1px solid rgba(255,255,255,.08);
}
.stat {
    background: rgba(255,255,255,.06); border-radius: 14px; padding: .8rem 1rem; margin-bottom: .6rem;
}
.stat b { font-size: 1.4rem; color: #FFD166; }
.stat span { display: block; color: #AEAED6; font-size: .8rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Text cleaning (same steps as your notebook)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_stopwords():
    try:
        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))


def clean_text(text: str) -> str:
    stop_words = get_stopwords()
    text = str(text).lower()                                              # 1. lowercase
    text = text.translate(str.maketrans("", "", string.punctuation))      # 2. remove punctuation
    text = "".join(ch for ch in text if not ch.isdigit())                 # 3. remove numbers
    text = "".join(ch for ch in text if ch.isascii())                     # 4. remove emojis
    words = [w for w in text.split() if w not in stop_words]              # 5. remove stop words
    return " ".join(words)


# ----------------------------------------------------------------------------
# Train the model (runs only once, then cached)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training the model... (only the first time)")
def train_model():
    df = pd.read_csv(DATA_FILE, sep=";", names=["text", "emotions"])
    df["clean"] = df["text"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["emotions"], test_size=0.20, random_state=42
    )

    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    acc = accuracy_score(y_test, model.predict(X_test_tfidf))
    counts = df["emotions"].value_counts()
    return model, vectorizer, acc, len(df), counts


def predict(text: str):
    """Return (label, confidence, {emotion: probability}, cleaned_text)."""
    model, vectorizer, *_ = train_model()
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    probs = model.predict_proba(vec)[0]
    prob_dict = dict(zip(model.classes_, probs))
    label = max(prob_dict, key=prob_dict.get)
    return label, prob_dict[label], prob_dict, cleaned


# ----------------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------------
def result_card(label, conf):
    info = EMOTIONS.get(label, {"emoji": "🙂", "color": "#8E7CFF", "msg": ""})
    st.markdown(
        f"""
        <div class="result" style="--c:{info['color']}">
            <div class="emoji">{info['emoji']}</div>
            <div class="label">{label}</div>
            <div class="conf">Confidence: <b>{conf*100:.1f}%</b></div>
            <div class="msg">{info['msg']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def prob_bars(prob_dict):
    rows = ""
    for emo, p in sorted(prob_dict.items(), key=lambda kv: kv[1], reverse=True):
        info = EMOTIONS.get(emo, {"emoji": "", "color": "#8E7CFF"})
        rows += f"""
        <div class="bar-row">
            <div class="bar-name">{info['emoji']} {emo}</div>
            <div class="bar-track"><div class="bar-fill" style="width:{p*100:.1f}%;background:{info['color']}"></div></div>
            <div class="bar-pct">{p*100:.1f}%</div>
        </div>"""
    st.markdown(f'<div class="card"><h4 style="margin-top:0">📊 Emotion probabilities</h4>{rows}</div>',
                unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Load model + sidebar
# ----------------------------------------------------------------------------
if not DATA_FILE.exists():
    st.error("`train.txt` not found. Put it in the same folder as `app.py` and refresh.")
    st.stop()

model, vectorizer, accuracy, n_rows, counts = train_model()

with st.sidebar:
    st.markdown("## 🧠 About the model")
    st.markdown(
        f"""
        <div class="stat"><b>{accuracy*100:.1f}%</b><span>Test accuracy</span></div>
        <div class="stat"><b>{n_rows:,}</b><span>Training sentences</span></div>
        <div class="stat"><b>{len(model.classes_)}</b><span>Emotions detected</span></div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("**Pipeline**")
    st.markdown(
        "1. Lowercase\n2. Remove punctuation, numbers, emojis\n"
        "3. Remove stop words\n4. TF-IDF vectorizer\n5. Logistic Regression"
    )
    st.markdown("---")
    st.markdown("**Try an example**")
    for name, sentence in EXAMPLES.items():
        if st.button(name, width="stretch", key=f"ex_{name}"):
            st.session_state["user_text"] = sentence

# ----------------------------------------------------------------------------
# Main page
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🧠 Emotion Detector</h1>
        <p>Type any sentence and let the NLP model tell you how it feels.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["✍️ Analyze text", "📁 Batch analysis", "📈 Dataset insights"])

# ---------------------------- TAB 1: single text ----------------------------
with tab1:
    left, right = st.columns([1.1, 1], gap="large")

    with left:
        st.markdown("#### Your sentence")
        user_text = st.text_area(
            "Enter text",
            key="user_text",
            height=170,
            placeholder="e.g. i feel really excited about my new job...",
            label_visibility="collapsed",
        )
        analyze = st.button("✨ Analyze emotion", width="stretch")

    with right:
        if analyze or user_text.strip():
            if not user_text.strip():
                st.warning("Please type something first.")
            else:
                label, conf, probs, cleaned = predict(user_text)
                result_card(label, conf)
        else:
            st.markdown(
                '<div class="card" style="text-align:center;padding:3rem 1rem">'
                '<div style="font-size:3.5rem">💬</div>'
                '<p style="color:#B9B9DD">Your result will appear here.<br>'
                'Type a sentence or pick an example from the sidebar.</p></div>',
                unsafe_allow_html=True,
            )

    if user_text.strip():
        st.write("")
        c1, c2 = st.columns([1.1, 1], gap="large")
        with c1:
            prob_bars(probs)
        with c2:
            st.markdown(
                f'<div class="card"><h4 style="margin-top:0">🧹 After cleaning</h4>'
                f'<p style="color:#B9B9DD;font-size:.9rem">What the model actually sees '
                f'(stop words, punctuation and numbers removed):</p>'
                f'<span class="chip">{cleaned if cleaned else "(nothing left after cleaning)"}</span></div>',
                unsafe_allow_html=True,
            )

# ---------------------------- TAB 2: batch analysis -------------------------
with tab2:
    st.markdown("#### Analyze many sentences at once")
    st.caption("Upload a CSV with a text column, or paste one sentence per line.")

    mode = st.radio("Input method", ["Paste text", "Upload CSV"], horizontal=True)
    texts = []

    if mode == "Paste text":
        raw = st.text_area("One sentence per line", height=160, key="batch_text",
                           placeholder="i am so happy\ni feel scared about tomorrow\n...")
        texts = [t.strip() for t in raw.split("\n") if t.strip()]
    else:
        file = st.file_uploader("Upload CSV", type=["csv"])
        if file is not None:
            up = pd.read_csv(file)
            col = st.selectbox("Which column has the text?", up.columns)
            texts = up[col].dropna().astype(str).tolist()

    if st.button("🚀 Run batch analysis") and texts:
        rows = []
        for t in texts:
            lab, cf, _, _ = predict(t)
            rows.append({"text": t, "emotion": f"{EMOTIONS[lab]['emoji']} {lab}", "confidence %": round(cf * 100, 1)})
        out = pd.DataFrame(rows)

        st.dataframe(out, width="stretch", hide_index=True)
        st.bar_chart(out["emotion"].value_counts())
        st.download_button("⬇️ Download results (CSV)", out.to_csv(index=False).encode("utf-8"),
                           "emotion_results.csv", "text/csv")
    elif texts == [] and st.session_state.get("batch_text"):
        st.info("Nothing to analyze yet.")

# ---------------------------- TAB 3: dataset insights -----------------------
with tab3:
    st.markdown("#### What the model learned from")
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        st.markdown("**Sentences per emotion**")
        st.bar_chart(counts)
    with c2:
        st.markdown("**Emotion guide**")
        for emo, info in EMOTIONS.items():
            n = int(counts.get(emo, 0))
            st.markdown(
                f"<div class='stat'><b>{info['emoji']} {emo.capitalize()}</b>"
                f"<span>{n:,} sentences - {info['msg']}</span></div>",
                unsafe_allow_html=True,
            )
    st.markdown("**Sample of training data**")
    sample = pd.read_csv(DATA_FILE, sep=";", names=["text", "emotion"]).sample(8, random_state=1)
    st.dataframe(sample, width="stretch", hide_index=True)

st.markdown(
    "<p style='text-align:center;color:#7d7da8;margin-top:2.5rem;font-size:.85rem'>"
    "Built with Streamlit · scikit-learn · NLTK</p>",
    unsafe_allow_html=True,
)
