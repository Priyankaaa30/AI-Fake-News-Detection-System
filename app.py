import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import re
import string
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_curve, auc
)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem; font-weight: 700;
        background: linear-gradient(90deg, #1a73e8, #e8453c);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle { color: #666; font-size: 1rem; margin-bottom: 2rem; }
    .metric-card {
        background: #f8f9fa; border-radius: 12px;
        padding: 1.2rem 1.5rem; border-left: 4px solid #1a73e8;
        margin-bottom: 1rem;
    }
    .result-real {
        background: #e8f5e9; border-radius: 12px;
        padding: 1.5rem; border-left: 5px solid #2e7d32;
        font-size: 1.1rem; font-weight: 600; color: #1b5e20;
    }
    .result-fake {
        background: #ffebee; border-radius: 12px;
        padding: 1.5rem; border-left: 5px solid #c62828;
        font-size: 1.1rem; font-weight: 600; color: #b71c1c;
    }
    .stProgress > div > div { background-color: #1a73e8; }
    .section-header {
        font-size: 1.15rem; font-weight: 600;
        color: #1a73e8; margin: 1.5rem 0 0.8rem;
        border-bottom: 2px solid #e8f0fe; padding-bottom: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)


# ─── Text cleaning ───────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ─── Data generation (synthetic dataset for demo) ───────────────────────────
@st.cache_data(show_spinner=False)
def generate_dataset(n_samples: int = 2000):
    """
    Generates a balanced synthetic dataset of REAL and FAKE news snippets.
    Replace this with real CSV data (e.g. Kaggle Fake News dataset) for
    production use — just load with pd.read_csv() instead.
    """
    np.random.seed(42)

    real_templates = [
        "Scientists at {org} published a peer-reviewed study in {journal} confirming {finding}.",
        "According to official data from {agency}, {statistic} as of {month} {year}.",
        "{official}, speaking at a press conference, confirmed that {policy} has been approved.",
        "A new report by {org} highlights the impact of {topic} on {outcome}, citing {n} studies.",
        "Researchers at {uni} developed a new {tech} that improves {metric} by {pct}%.",
        "The {country} government announced a {policy} initiative backed by {amount} in funding.",
        "Health authorities confirmed {cases} new cases of {disease}, urging residents to {action}.",
        "The unemployment rate fell to {pct}% according to {agency}, driven by gains in {sector}.",
        "{company} reported quarterly earnings of {amount}, beating analyst expectations by {pct}%.",
        "A {year} census showed that {fact}, with experts calling it a {adj} demographic shift.",
    ]

    fake_templates = [
        "BREAKING: {person} SECRETLY {action} — whistleblower EXPOSES shocking TRUTH!!!",
        "ALERT: Government HIDING {topic} from public — share before it gets DELETED!!!",
        "Doctors DON'T want you to know about this {remedy} that CURES {disease} overnight!",
        "EXPOSED: {person} caught {action} — mainstream media REFUSES to cover it up!!",
        "The REAL reason {topic} is being covered up by {org} will SHOCK you!!!",
        "Scientists BAFFLED as {miracle} DESTROYS {problem} in just {n} days — Big {org} FURIOUS!",
        "100% PROOF that {conspiracy} — share this before {platform} BANS it FOREVER!!!",
        "This ONE weird trick {action} and {authority} doesn't want you to know WHY!",
        "URGENT: {product} recalled after it was found to cause {problem} in 9 out of 10 users!!",
        "BOMBSHELL: Secret documents REVEAL {person} has been {action} for YEARS — SPREAD THE WORD",
    ]

    orgs     = ["NASA","WHO","MIT","Stanford University","Oxford University","Harvard Medical School"]
    journals = ["Nature","The Lancet","Science","NEJM","Cell"]
    agencies = ["Bureau of Labor Statistics","CDC","OECD","IMF","World Bank"]
    people   = ["a top senator","the CEO","a former official","an unnamed source","the president"]
    actions  = ["planning a cover-up","manipulating data","hiding evidence","running a secret program"]
    topics   = ["5G towers","tap water chemicals","vaccine microchips","climate data","food additives"]
    months   = ["January","February","March","April","May","June","July","August","September"]

    rows = []
    for _ in range(n_samples // 2):
        t = np.random.choice(real_templates)
        text = t.format(
            org=np.random.choice(orgs), journal=np.random.choice(journals),
            finding="a significant correlation between diet and longevity",
            agency=np.random.choice(agencies),
            statistic=f"GDP growth reached {np.random.uniform(1,5):.1f}%",
            month=np.random.choice(months), year=np.random.randint(2020, 2025),
            official="The health minister", policy="universal screening",
            topic="climate change", outcome="public health", n=np.random.randint(20,200),
            uni=np.random.choice(orgs), tech="diagnostic tool",
            metric="detection accuracy", pct=np.random.randint(10,60),
            country="The federal", amount=f"${np.random.randint(10,500)}M",
            cases=np.random.randint(100,5000), disease="respiratory illness",
            action="wear masks indoors", sector="technology",
            company="A major tech firm", year=np.random.randint(2018,2025),
            fact=f"{np.random.randint(40,80)}% of households now own a smartphone",
            adj="significant",
        )
        rows.append({"text": text, "label": 1})   # 1 = REAL

    for _ in range(n_samples // 2):
        t = np.random.choice(fake_templates)
        text = t.format(
            person=np.random.choice(people), action=np.random.choice(actions),
            topic=np.random.choice(topics), org=np.random.choice(["Pharma","Government","Media"]),
            remedy="herbal extract", disease="cancer",
            miracle="one simple fruit", problem="inflammation",
            n=np.random.randint(3, 14), conspiracy="elections are rigged",
            platform=np.random.choice(["Facebook","YouTube","Twitter"]),
            authority="the government", product="a popular supplement",
        )
        rows.append({"text": text, "label": 0})   # 0 = FAKE

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    df["clean_text"] = df["text"].apply(clean_text)
    return df


# ─── Model training ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_model():
    df = generate_dataset()
    X = df["clean_text"]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_df=0.7, ngram_range=(1, 2))),
        ("clf",   LogisticRegression(max_iter=1000, C=1.0)),
    ])
    pipeline.fit(X_train, y_train)
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "report":    classification_report(y_test, y_pred, target_names=["Fake", "Real"], output_dict=True),
        "cm":        confusion_matrix(y_test, y_pred),
        "roc":       roc_curve(y_test, y_proba),
        "auc":       auc(*roc_curve(y_test, y_proba)[:2]),
        "X_test":    X_test,
        "y_test":    y_test,
        "y_pred":    y_pred,
    }
    return pipeline, metrics, df


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 Navigation")
    page = st.radio("Go to", ["Home & Detector", "Model Performance", "Dataset Explorer", "About"])
    st.markdown("---")
    st.markdown("### ⚙️ Model Settings")
    show_confidence = st.checkbox("Show confidence score", value=True)
    show_top_words  = st.checkbox("Show top feature words", value=True)
    st.markdown("---")
    st.markdown("### 📌 Project Info")
    st.markdown("""
- **Model:** Logistic Regression  
- **Features:** TF-IDF (unigrams + bigrams)  
- **Developer:** Priyanka Rajput and team
- **College:** Manav Rachna IIRS  
    """)


# ─── Load model ──────────────────────────────────────────────────────────────
with st.spinner("Loading model… this takes a few seconds on first run."):
    pipeline, metrics, df = train_model()


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Home & Detector
# ════════════════════════════════════════════════════════════════════════════
if page == "Home & Detector":
    st.markdown('<h1 class="main-title">📰 AI-Powered Fake News Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Developed by Priyanka Rajput, Somye, Manas, Om · Manav Rachna IIRS · Microsoft AI Internship</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div style="font-size:0.8rem;color:#666">Model Accuracy</div>
            <div style="font-size:2rem;font-weight:700;color:#1a73e8">{metrics['accuracy']*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div style="font-size:0.8rem;color:#666">ROC-AUC Score</div>
            <div style="font-size:2rem;font-weight:700;color:#1a73e8">{metrics['auc']:.3f}</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div style="font-size:0.8rem;color:#666">Training Samples</div>
            <div style="font-size:2rem;font-weight:700;color:#1a73e8">{len(df):,}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">🧪 Try It — Paste a News Article</div>', unsafe_allow_html=True)

    examples = {
        "Select an example…": "",
        "✅ Credible (Scientific)": "Scientists at NASA confirm the James Webb Space Telescope has captured detailed images of a newly discovered exoplanet with a potential water vapor atmosphere, according to a peer-reviewed paper published in Nature journal.",
        "✅ Credible (Economic)": "The unemployment rate fell to 3.8% last month according to the Bureau of Labor Statistics, the lowest level in six months, driven by gains in the hospitality and healthcare sectors.",
        "❌ Fake (Clickbait)": "BREAKING: Government secretly adding mind control chemicals to tap water — whistleblower EXPOSES shocking TRUTH that mainstream media refuses to cover!! Share before it gets deleted!!!",
        "❌ Fake (Health Misinformation)": "Doctors DON'T want you to know about this simple herbal remedy that CURES cancer overnight! Big Pharma is FURIOUS that millions are healing themselves — share this NOW before it's banned!",
    }

    chosen = st.selectbox("Load an example:", list(examples.keys()))
    default_text = examples[chosen]

    user_input = st.text_area(
        "Enter news article text:",
        value=default_text,
        height=160,
        placeholder="Paste a news headline or article here…",
    )

    col_btn1, col_btn2, _ = st.columns([1, 1, 4])
    with col_btn1:
        analyze_clicked = st.button("🔍 Analyze", type="primary", use_container_width=True)
    with col_btn2:
        clear_clicked = st.button("🗑️ Clear", use_container_width=True)

    if clear_clicked:
        st.rerun()

    if analyze_clicked and user_input.strip():
        with st.spinner("Analyzing…"):
            time.sleep(0.4)
            cleaned  = clean_text(user_input)
            pred     = pipeline.predict([cleaned])[0]
            proba    = pipeline.predict_proba([cleaned])[0]
            conf     = proba[pred] * 100
            label    = "REAL" if pred == 1 else "FAKE"
            css_cls  = "result-real" if pred == 1 else "result-fake"
            icon     = "✅" if pred == 1 else "❌"

        st.markdown(f"""
        <div class="{css_cls}">
            {icon} This article is classified as <strong>{label}</strong>
        </div>""", unsafe_allow_html=True)

        if show_confidence:
            st.markdown("**Confidence scores:**")
            col_r, col_f = st.columns(2)
            with col_r:
                st.metric("Real probability",  f"{proba[1]*100:.1f}%")
                st.progress(float(proba[1]))
            with col_f:
                st.metric("Fake probability",  f"{proba[0]*100:.1f}%")
                st.progress(float(proba[0]))

        if show_top_words:
            st.markdown('<div class="section-header">🔑 Key Words That Influenced This Prediction</div>', unsafe_allow_html=True)
            vectorizer = pipeline.named_steps["tfidf"]
            clf        = pipeline.named_steps["clf"]
            tokens     = vectorizer.transform([cleaned])
            feature_names = vectorizer.get_feature_names_out()
            nonzero_idx   = tokens.nonzero()[1]
            if len(nonzero_idx) > 0:
                coefs  = clf.coef_[0][nonzero_idx]
                words  = feature_names[nonzero_idx]
                top_n  = min(10, len(words))
                sorted_idx = np.argsort(np.abs(coefs))[::-1][:top_n]
                word_df = pd.DataFrame({
                    "Word / Phrase": words[sorted_idx],
                    "Influence":     coefs[sorted_idx],
                    "Direction":     ["→ Real" if c > 0 else "→ Fake" for c in coefs[sorted_idx]],
                })
                st.dataframe(word_df, use_container_width=True, hide_index=True)

    elif analyze_clicked:
        st.warning("Please enter some text before analyzing.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Model Performance
# ════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.markdown("## 📊 Model Performance")
    st.markdown("Detailed evaluation metrics for the Logistic Regression classifier.")

    # ── Classification report ────────────────────────────────────────────
    st.markdown('<div class="section-header">Classification Report</div>', unsafe_allow_html=True)
    report = metrics["report"]
    report_df = pd.DataFrame({
        "Class":     ["Fake", "Real", "Macro Avg", "Weighted Avg"],
        "Precision": [report["Fake"]["precision"], report["Real"]["precision"],
                      report["macro avg"]["precision"], report["weighted avg"]["precision"]],
        "Recall":    [report["Fake"]["recall"],    report["Real"]["recall"],
                      report["macro avg"]["recall"],    report["weighted avg"]["recall"]],
        "F1-Score":  [report["Fake"]["f1-score"],  report["Real"]["f1-score"],
                      report["macro avg"]["f1-score"],  report["weighted avg"]["f1-score"]],
        "Support":   [int(report["Fake"]["support"]), int(report["Real"]["support"]),
                      int(report["macro avg"]["support"]), int(report["weighted avg"]["support"])],
    })
    report_df[["Precision","Recall","F1-Score"]] = report_df[["Precision","Recall","F1-Score"]].applymap(lambda x: f"{x:.3f}")
    st.dataframe(report_df, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)

    # ── Confusion Matrix ─────────────────────────────────────────────────
    with col1:
        st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        cm = metrics["cm"]
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"],
            linewidths=0.5, ax=ax, annot_kws={"size": 14}
        )
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)
        ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── ROC Curve ────────────────────────────────────────────────────────
    with col2:
        st.markdown('<div class="section-header">ROC Curve</div>', unsafe_allow_html=True)
        fpr, tpr, _ = metrics["roc"]
        roc_auc     = metrics["auc"]
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.plot(fpr, tpr, color="#1a73e8", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
        ax2.plot([0, 1], [0, 1], color="#999", lw=1, linestyle="--", label="Random classifier")
        ax2.fill_between(fpr, tpr, alpha=0.08, color="#1a73e8")
        ax2.set_xlim([0, 1]); ax2.set_ylim([0, 1.02])
        ax2.set_xlabel("False Positive Rate", fontsize=11)
        ax2.set_ylabel("True Positive Rate", fontsize=11)
        ax2.set_title("ROC Curve", fontsize=13, fontweight="bold")
        ax2.legend(loc="lower right", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # ── Top TF-IDF Features ──────────────────────────────────────────────
    st.markdown('<div class="section-header">Top 20 Most Informative Words</div>', unsafe_allow_html=True)
    vectorizer   = pipeline.named_steps["tfidf"]
    clf          = pipeline.named_steps["clf"]
    feature_names = vectorizer.get_feature_names_out()
    coefs         = clf.coef_[0]
    top_pos_idx   = np.argsort(coefs)[-10:][::-1]
    top_neg_idx   = np.argsort(coefs)[:10]

    words  = list(feature_names[top_pos_idx]) + list(feature_names[top_neg_idx])
    values = list(coefs[top_pos_idx])         + list(coefs[top_neg_idx])
    colors = ["#2e7d32"] * 10 + ["#c62828"] * 10

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    bars = ax3.barh(words, values, color=colors, edgecolor="white", height=0.7)
    ax3.axvline(0, color="#333", linewidth=0.8)
    ax3.set_xlabel("TF-IDF Coefficient (positive → Real, negative → Fake)", fontsize=10)
    ax3.set_title("Most Informative Features", fontsize=13, fontweight="bold")
    green_patch = mpatches.Patch(color="#2e7d32", label="Signals REAL news")
    red_patch   = mpatches.Patch(color="#c62828", label="Signals FAKE news")
    ax3.legend(handles=[green_patch, red_patch], fontsize=10)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Dataset Explorer
# ════════════════════════════════════════════════════════════════════════════
elif page == "Dataset Explorer":
    st.markdown("## 🗂️ Dataset Explorer")

    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Total Samples",   len(df))
    with col2: st.metric("Real Articles",   int((df["label"] == 1).sum()))
    with col3: st.metric("Fake Articles",   int((df["label"] == 0).sum()))

    st.markdown('<div class="section-header">Class Distribution</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(5, 3))
    counts = df["label"].value_counts()
    ax.bar(["Fake", "Real"], [counts[0], counts[1]], color=["#c62828", "#2e7d32"], width=0.5)
    ax.set_ylabel("Number of articles"); ax.set_title("Class Balance")
    for i, v in enumerate([counts[0], counts[1]]):
        ax.text(i, v + 10, str(v), ha="center", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown('<div class="section-header">Sample Articles</div>', unsafe_allow_html=True)
    filter_label = st.radio("Filter by:", ["All", "Real only", "Fake only"], horizontal=True)
    if filter_label == "Real only":
        show_df = df[df["label"] == 1]
    elif filter_label == "Fake only":
        show_df = df[df["label"] == 0]
    else:
        show_df = df
    display = show_df[["text", "label"]].copy()
    display["label"] = display["label"].map({1: "✅ Real", 0: "❌ Fake"})
    st.dataframe(display.head(20).reset_index(drop=True), use_container_width=True)

    st.markdown('<div class="section-header">Article Length Distribution</div>', unsafe_allow_html=True)
    df["text_len"] = df["text"].str.split().str.len()
    fig2, ax2 = plt.subplots(figsize=(8, 3))
    ax2.hist(df[df["label"]==1]["text_len"], bins=30, alpha=0.6, color="#2e7d32", label="Real")
    ax2.hist(df[df["label"]==0]["text_len"], bins=30, alpha=0.6, color="#c62828", label="Fake")
    ax2.set_xlabel("Word count"); ax2.set_ylabel("Frequency")
    ax2.set_title("Article Length by Class"); ax2.legend()
    plt.tight_layout(); st.pyplot(fig2); plt.close()


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — About
# ════════════════════════════════════════════════════════════════════════════
elif page == "About":
    st.markdown("## ℹ️ About This Project")
    st.markdown("""
### 📰 AI-Powered Fake News Detection System

**Developer:** Priyanka Rajput and team  
**College:** Manav Rachna International Institute of Research and Studies  
**Department:** B.Tech Computer Science and Engineering  
**Internship:** Microsoft AI Internship (GitHub Copilot)

---

### 🎯 Project Objective
Build an AI-powered system that can automatically classify news articles as **real** or **fake**
using Natural Language Processing (NLP) and Machine Learning techniques.

---

### 🧰 Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.x |
| Web Framework | Streamlit |
| ML Model | Logistic Regression |
| Feature Extraction | TF-IDF Vectorizer |
| Data Processing | pandas, numpy |
| Visualization | matplotlib, seaborn |
| ML Library | scikit-learn |

---

### 🔬 How It Works

1. **Text Preprocessing** — Lowercasing, removing URLs, punctuation, numbers, and extra whitespace  
2. **Feature Extraction** — TF-IDF converts text to numerical vectors (unigrams + bigrams)  
3. **Classification** — Logistic Regression predicts Real (1) or Fake (0)  
4. **Evaluation** — Accuracy, Confusion Matrix, ROC-AUC, Classification Report  

---

### 📊 Model Performance Summary
    """)
    st.metric("Accuracy",  f"{metrics['accuracy']*100:.1f}%")
    st.metric("ROC-AUC",   f"{metrics['auc']:.3f}")
    st.markdown("""
---
### 📚 References
- Scikit-learn Documentation: https://scikit-learn.org  
- Streamlit Documentation: https://docs.streamlit.io  
- TF-IDF Paper: Salton & Buckley (1988)  
- Fake News Detection Survey: Shu et al. (2017)
    """)
