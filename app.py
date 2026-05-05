import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import json, warnings
warnings.filterwarnings("ignore")

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
*, *::before, *::after { font-family: 'Inter', sans-serif; box-sizing: border-box; }

.stApp {
    background: linear-gradient(-45deg, #0d0d1a, #12122a, #1a0d2e, #0d1a2e);
    background-size: 400% 400%;
    animation: bgShift 18s ease infinite;
    min-height: 100vh;
}
@keyframes bgShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Hide branding but keep sidebar toggle arrow */
#MainMenu, footer, [data-testid="stDecoration"] {
    visibility: hidden !important; display: none !important;
}
[data-testid="stToolbar"] { display: none !important; }

/* Keep the header just for the sidebar collapse button */
header[data-testid="stHeader"] {
    background: transparent !important;
    height: auto !important;
}

/* Style the sidebar toggle arrow button so it looks good */
[data-testid="collapsedControl"] {
    background: rgba(108, 92, 231, 0.25) !important;
    border: 1px solid rgba(162, 155, 254, 0.35) !important;
    border-radius: 0 12px 12px 0 !important;
    color: #a29bfe !important;
    transition: background 0.25s, transform 0.25s !important;
    top: 1rem !important;
}
[data-testid="collapsedControl"]:hover {
    background: rgba(108, 92, 231, 0.5) !important;
    transform: scale(1.08) !important;
}

.main .block-container {
    max-width: 940px !important;
    padding: 2rem 2rem 4rem !important;
    margin: 0 auto !important;
}

/* Hero */
.hero-wrap { text-align:center; padding:2.2rem 0 0.3rem; animation: fadeDown 0.9s both; }
@keyframes fadeDown {
    from { opacity:0; transform:translateY(-24px); }
    to   { opacity:1; transform:translateY(0); }
}
.hero-icon { font-size:3.4rem; display:block; margin-bottom:0.4rem; animation: heartBeat 1.8s ease-in-out infinite; }
@keyframes heartBeat {
    0%,100%{transform:scale(1)} 14%{transform:scale(1.14)} 28%{transform:scale(1)} 42%{transform:scale(1.08)} 70%{transform:scale(1)}
}
.hero-title {
    font-size:clamp(1.7rem,4vw,2.7rem); font-weight:800; margin:0 0 0.35rem;
    background:linear-gradient(110deg,#ff6b6b 0%,#feca57 35%,#a29bfe 70%,#74b9ff 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    letter-spacing:-0.5px;
}
.hero-sub { font-size:0.93rem; color:rgba(255,255,255,0.42); margin:0 0 1.8rem; }

/* Stat pills */
.stat-row { display:flex; justify-content:center; gap:10px; flex-wrap:wrap; margin-bottom:2.2rem; animation:fadeUp 0.8s 0.2s both; }
@keyframes fadeUp { from{opacity:0;transform:translateY(18px)} to{opacity:1;transform:translateY(0)} }
.stat-pill {
    background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.11);
    border-radius:50px; padding:5px 16px; font-size:0.78rem; color:rgba(255,255,255,0.6);
    backdrop-filter:blur(8px);
}
.stat-pill b { color:#feca57; }

/* Glass card */
.glass {
    background:rgba(255,255,255,0.055);
    backdrop-filter:blur(20px) saturate(150%);
    -webkit-backdrop-filter:blur(20px) saturate(150%);
    border:1px solid rgba(255,255,255,0.11);
    border-radius:22px; padding:26px 28px; margin-bottom:18px;
    box-shadow:0 10px 40px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.07);
    animation:fadeUp 0.7s both;
    transition:transform 0.3s ease, box-shadow 0.3s ease;
}
.glass:hover { transform:translateY(-3px); box-shadow:0 16px 50px rgba(0,0,0,0.45); }

.sec-label {
    font-size:0.7rem; font-weight:700; letter-spacing:1.8px; text-transform:uppercase;
    color:rgba(255,255,255,0.3); margin-bottom:14px;
}

/* Tooltip */
.tip-wrap { display:inline-flex; align-items:center; gap:6px; margin-bottom:5px; }
.tip-label { font-size:0.84rem; color:rgba(255,255,255,0.8); font-weight:500; }
.tip-q {
    display:inline-flex; align-items:center; justify-content:center;
    width:17px; height:17px;
    background:rgba(254,202,87,0.15); border:1px solid rgba(254,202,87,0.4);
    border-radius:50%; font-size:10px; font-weight:700; color:#feca57;
    cursor:help; position:relative; flex-shrink:0; transition:background 0.2s;
}
.tip-q:hover { background:rgba(254,202,87,0.32); }
.tip-q .tip-box {
    visibility:hidden; opacity:0;
    position:absolute; bottom:130%; left:50%; transform:translateX(-50%);
    background:rgba(6,6,18,0.97); border:1px solid rgba(254,202,87,0.28);
    border-radius:12px; padding:10px 13px; width:205px;
    font-size:0.75rem; color:rgba(255,255,255,0.85); line-height:1.6;
    box-shadow:0 8px 30px rgba(0,0,0,0.6); z-index:9999;
    transition:opacity 0.2s; pointer-events:none;
}
.tip-q:hover .tip-box { visibility:visible; opacity:1; }

/* Widget overrides */
label { color:rgba(255,255,255,0.72) !important; font-size:0.83rem !important; }
.stSelectbox > div > div {
    background:rgba(255,255,255,0.06) !important;
    border:1px solid rgba(255,255,255,0.12) !important;
    border-radius:12px !important; color:white !important;
}
.stNumberInput > div > div > input {
    background:rgba(255,255,255,0.06) !important;
    border:1px solid rgba(255,255,255,0.12) !important;
    border-radius:12px !important; color:white !important;
}
div[data-testid="stNumberInput"] button { color:white !important; background:rgba(255,255,255,0.08) !important; }
.stTextInput > div > div > input {
    background:rgba(255,255,255,0.06) !important;
    border:1px solid rgba(255,255,255,0.12) !important;
    border-radius:12px !important; color:white !important;
}
.stTextArea > div > div > textarea {
    background:rgba(255,255,255,0.06) !important;
    border:1px solid rgba(255,255,255,0.12) !important;
    border-radius:12px !important; color:rgba(255,255,255,0.85) !important;
    font-size:0.87rem !important;
}

/* Buttons */
.stButton > button {
    background:linear-gradient(135deg,#6c5ce7,#a29bfe) !important;
    color:white !important; border:none !important;
    border-radius:50px !important; padding:11px 32px !important;
    font-weight:600 !important; font-size:0.92rem !important;
    width:100% !important; letter-spacing:0.2px !important;
    transition:all 0.3s ease !important;
    box-shadow:0 4px 20px rgba(108,92,231,0.38) !important;
}
.stButton > button:hover {
    transform:translateY(-2px) !important;
    box-shadow:0 8px 30px rgba(108,92,231,0.58) !important;
}
.predict-btn .stButton > button {
    background:linear-gradient(135deg,#ff6b6b,#ee5a24) !important;
    box-shadow:0 4px 22px rgba(255,107,107,0.42) !important;
    font-size:1rem !important; padding:13px 40px !important;
}
.predict-btn .stButton > button:hover {
    box-shadow:0 10px 34px rgba(255,107,107,0.62) !important;
    background:linear-gradient(135deg,#ff7675,#fd79a8) !important;
}
.ai-btn .stButton > button {
    background:linear-gradient(135deg,#00b894,#00cec9) !important;
    box-shadow:0 4px 20px rgba(0,184,148,0.38) !important;
}
.ai-btn .stButton > button:hover {
    box-shadow:0 8px 30px rgba(0,184,148,0.58) !important;
}

/* Result cards */
.result-danger {
    background:linear-gradient(135deg,rgba(255,71,87,0.18),rgba(214,48,49,0.10));
    border:1px solid rgba(255,71,87,0.38); border-radius:20px; padding:30px 26px; text-align:center;
    box-shadow:0 0 50px rgba(255,71,87,0.1), inset 0 1px 0 rgba(255,150,150,0.1);
    animation:resultIn 0.6s both, dangerGlow 3s 0.6s ease-in-out infinite;
    backdrop-filter:blur(12px);
}
.result-safe {
    background:linear-gradient(135deg,rgba(0,210,150,0.18),rgba(0,184,212,0.10));
    border:1px solid rgba(0,210,150,0.38); border-radius:20px; padding:30px 26px; text-align:center;
    box-shadow:0 0 50px rgba(0,210,150,0.1), inset 0 1px 0 rgba(100,255,200,0.1);
    animation:resultIn 0.6s both, safeGlow 3s 0.6s ease-in-out infinite;
    backdrop-filter:blur(12px);
}
@keyframes resultIn { from{opacity:0;transform:scale(0.88) translateY(20px)} to{opacity:1;transform:scale(1) translateY(0)} }
@keyframes dangerGlow { 0%,100%{box-shadow:0 0 50px rgba(255,71,87,0.1)} 50%{box-shadow:0 0 70px rgba(255,71,87,0.22)} }
@keyframes safeGlow   { 0%,100%{box-shadow:0 0 50px rgba(0,210,150,0.1)} 50%{box-shadow:0 0 70px rgba(0,210,150,0.22)} }
.result-emoji { font-size:3.2rem; display:block; margin-bottom:12px; }
.result-title { font-size:1.65rem; font-weight:800; color:white; margin:0 0 8px; }
.result-sub   { font-size:0.9rem; color:rgba(255,255,255,0.62); margin:0; }

/* Prob bars */
.prob-row { display:flex; align-items:center; gap:12px; margin-bottom:10px; }
.prob-label { font-size:0.8rem; color:rgba(255,255,255,0.58); width:100px; flex-shrink:0; }
.prob-bar-bg { flex:1; height:7px; background:rgba(255,255,255,0.08); border-radius:99px; overflow:hidden; }
.prob-bar-fill { height:100%; border-radius:99px; }
.prob-pct { font-size:0.8rem; font-weight:700; width:40px; text-align:right; }

/* Metric chips */
.chip-row { display:flex; gap:10px; flex-wrap:wrap; margin-top:16px; }
.chip {
    background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.1);
    border-radius:12px; padding:10px 16px; text-align:center; flex:1; min-width:80px;
}
.chip-val { font-size:1.25rem; font-weight:700; color:#a29bfe; }
.chip-lbl { font-size:0.68rem; color:rgba(255,255,255,0.38); margin-top:2px; }

/* AI panel */
.ai-header { display:flex; align-items:center; gap:10px; margin-bottom:6px; }
.ai-title  { font-size:0.95rem; font-weight:700; color:#00d2ff; }
.ai-desc   { font-size:0.82rem; color:rgba(255,255,255,0.45); margin-bottom:14px; line-height:1.6; }

/* About */
.about-name {
    display:inline-block; background:rgba(162,155,254,0.14); border:1px solid rgba(162,155,254,0.25);
    border-radius:8px; padding:2px 12px; font-size:0.8rem; color:#a29bfe; margin:3px 4px;
}
.div-line { height:1px; background:linear-gradient(90deg,transparent,rgba(255,255,255,0.09),transparent); margin:18px 0; }

::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:rgba(255,255,255,0.03); }
::-webkit-scrollbar-thumb { background:rgba(162,155,254,0.35); border-radius:3px; }

.stAlert { background:rgba(255,255,255,0.05) !important; border-radius:14px !important; border:1px solid rgba(255,255,255,0.1) !important; }
h3 { color:rgba(255,255,255,0.9) !important; }
p, li { color:rgba(255,255,255,0.7) !important; }
</style>
""", unsafe_allow_html=True)


# ── Tooltip helper ────────────────────────────────────────────────────────────
def tip(label, hint):
    return (f"<div class='tip-wrap'>"
            f"<span class='tip-label'>{label}</span>"
            f"<span class='tip-q'>?<span class='tip-box'>{hint}</span></span>"
            f"</div>")


# ── Load & prepare ────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("heart.csv")
    except FileNotFoundError:
        st.error("❌  heart.csv not found — make sure it's in the same folder as app.py")
        st.stop()
    df_enc = df.copy()
    for col in ['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope']:
        df_enc[col] = LabelEncoder().fit_transform(df_enc[col])
    X = df_enc.drop("HeartDisease", axis=1)
    y = df_enc["HeartDisease"]
    scaler = StandardScaler()
    X_s = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X_s, y, scaler


# ── Auto-pick best model ──────────────────────────────────────────────────────
@st.cache_resource
def get_best_model():
    X, y, scaler = load_data()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    candidates = {
        "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42),
        "Logistic Regression": LogisticRegression(C=1, max_iter=1000, random_state=42),
        "SVM":                 SVC(C=10, kernel="rbf", probability=True, random_state=42),
        "KNN (k=15)":          KNeighborsClassifier(n_neighbors=15),
    }
    best_name, best_cv, best_m = "", 0, None
    for name, m in candidates.items():
        cv = cross_val_score(m, X, y, cv=5, scoring="accuracy").mean()
        if cv > best_cv:
            best_cv, best_name, best_m = cv, name, m
    best_m.fit(Xtr, ytr)
    tacc = accuracy_score(yte, best_m.predict(Xte))
    cvs  = cross_val_score(best_m, X, y, cv=5, scoring="accuracy")
    rep  = classification_report(yte, best_m.predict(Xte), output_dict=True)
    return best_m, scaler, best_name, tacc, cvs.mean(), rep


# ── Encode user row ───────────────────────────────────────────────────────────
def encode_row(row, scaler):
    sex_m = {"Male":"M","Female":"F"}
    cp_m  = {"Asymptomatic":"ASY","Typical Angina":"TA","Atypical Angina":"ATA","Non-Anginal Pain":"NAP"}
    ecg_m = {"Normal":"Normal","ST-T Abnormality":"ST","Left Ventricular Hypertrophy":"LVH"}
    ea_m  = {"Yes":"Y","No":"N"}
    sl_m  = {"Upsloping":"Up","Flat":"Flat","Downsloping":"Down"}

    le_sex = LabelEncoder(); le_sex.fit(["F","M"])
    le_cp  = LabelEncoder(); le_cp.fit(["ASY","ATA","NAP","TA"])
    le_ecg = LabelEncoder(); le_ecg.fit(["LVH","Normal","ST"])
    le_ea  = LabelEncoder(); le_ea.fit(["N","Y"])
    le_sl  = LabelEncoder(); le_sl.fit(["Down","Flat","Up"])

    vals = [
        row["Age"],
        le_sex.transform([sex_m[row["Sex"]]])[0],
        le_cp.transform([cp_m[row["ChestPainType"]]])[0],
        row["RestingBP"], row["Cholesterol"],
        1 if row["FastingBS"] == "Yes" else 0,
        le_ecg.transform([ecg_m[row["RestingECG"]]])[0],
        row["MaxHR"],
        le_ea.transform([ea_m[row["ExerciseAngina"]]])[0],
        row["Oldpeak"],
        le_sl.transform([sl_m[row["ST_Slope"]]])[0],
    ]
    cols = ["Age","Sex","ChestPainType","RestingBP","Cholesterol",
            "FastingBS","RestingECG","MaxHR","ExerciseAngina","Oldpeak","ST_Slope"]
    return scaler.transform(pd.DataFrame([vals], columns=cols))


# ════════════════════════════════════════════════════════════════════════════
#  BOOT — train silently
# ════════════════════════════════════════════════════════════════════════════
with st.spinner("Starting up..."):
    try:
        model, scaler, model_name, test_acc, cv_mean, report = get_best_model()
        prec = report["weighted avg"]["precision"]
        model_ready = True
    except Exception as e:
        st.error(f"Startup error: {e}")
        model_ready = False


# ════════════════════════════════════════════════════════════════════════════
#  HERO
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='hero-wrap'>
  <span class='hero-icon'>🫀</span>
  <h1 class='hero-title'>Heart Disease Prediction System</h1>
  <p class='hero-sub'>Fill in your details below — the model will analyse your risk instantly</p>
</div>
""", unsafe_allow_html=True)

if model_ready:
    st.markdown(f"""
    <div class='stat-row'>
      <div class='stat-pill'>🧠 Auto-selected: <b>{model_name}</b></div>
      <div class='stat-pill'>🎯 Accuracy: <b>{test_acc*100:.1f}%</b></div>
      <div class='stat-pill'>📊 CV Score: <b>{cv_mean*100:.1f}%</b></div>
      <div class='stat-pill'>👥 Trained on <b>918 patients</b></div>
      <div class='stat-pill'>🔬 Precision: <b>{prec:.2f}</b></div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  AI ASSISTANT
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.markdown("""
<div class='ai-header'>
  <span style='font-size:1.3rem;'>🤖</span>
  <span class='ai-title'>AI Assistant — Auto-fill from your description</span>
</div>
<div class='ai-desc'>
  Don't want to fill everything manually? Just describe your health situation in plain words
  and the AI will read it and fill the form for you automatically.
</div>
""", unsafe_allow_html=True)

d_col, k_col = st.columns([3, 2])
with d_col:
    user_desc = st.text_area(
        "desc", height=85, label_visibility="collapsed",
        placeholder='e.g. "52 year old male, BP around 138, cholesterol 260, gets chest pain on exercise, max HR was 120..."'
    )
with k_col:
    api_key = st.text_input(
        "Anthropic API Key", type="password",
        placeholder="sk-ant-api03-...",
        help="Get a free key at console.anthropic.com"
    )

a1, a2, _ = st.columns([1.2, 1.2, 2.6])
with a1:
    st.markdown('<div class="ai-btn">', unsafe_allow_html=True)
    ai_clicked = st.button("✨ Fill with AI", key="ai_btn")
    st.markdown('</div>', unsafe_allow_html=True)
with a2:
    if st.button("🔄 Clear", key="clear_btn"):
        for k in ["ai_age","ai_sex","ai_cp","ai_rbp","ai_chol",
                  "ai_fbs","ai_ecg","ai_hr","ai_ea","ai_op","ai_sl"]:
            st.session_state.pop(k, None)
        st.rerun()

if ai_clicked:
    if not user_desc.strip():
        st.warning("Please type your health description first.")
    elif not api_key.strip():
        st.warning("Enter your Anthropic API key to use AI fill.")
    else:
        with st.spinner("AI is reading your description..."):
            try:
                import urllib.request
                payload = json.dumps({
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 500,
                    "system": (
                        "You are a medical data extractor. Extract values from the user's description and return "
                        "ONLY a valid JSON object with these exact keys: "
                        "Age (int 18-100), Sex (Male or Female), "
                        "ChestPainType (Asymptomatic / Typical Angina / Atypical Angina / Non-Anginal Pain), "
                        "RestingBP (int 80-220), Cholesterol (int 0-650), FastingBS (Yes or No), "
                        "RestingECG (Normal / ST-T Abnormality / Left Ventricular Hypertrophy), "
                        "MaxHR (int 50-220), ExerciseAngina (Yes or No), Oldpeak (float -3 to 7), "
                        "ST_Slope (Upsloping / Flat / Downsloping). "
                        "Use healthy baseline values for anything not mentioned. Return ONLY JSON, no other text."
                    ),
                    "messages": [{"role":"user","content": user_desc}]
                }).encode()
                req = urllib.request.Request(
                    "https://api.anthropic.com/v1/messages", data=payload,
                    headers={"Content-Type":"application/json",
                             "x-api-key": api_key.strip(),
                             "anthropic-version":"2023-06-01"}
                )
                with urllib.request.urlopen(req, timeout=20) as resp:
                    data = json.loads(resp.read().decode())
                raw = data["content"][0]["text"].strip()
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"): raw = raw[4:]
                parsed = json.loads(raw.strip())

                st.session_state["ai_age"]  = int(parsed.get("Age", 45))
                st.session_state["ai_sex"]  = parsed.get("Sex", "Male")
                st.session_state["ai_cp"]   = parsed.get("ChestPainType", "Asymptomatic")
                st.session_state["ai_rbp"]  = int(parsed.get("RestingBP", 120))
                st.session_state["ai_chol"] = int(parsed.get("Cholesterol", 200))
                st.session_state["ai_fbs"]  = parsed.get("FastingBS", "No")
                st.session_state["ai_ecg"]  = parsed.get("RestingECG", "Normal")
                st.session_state["ai_hr"]   = int(parsed.get("MaxHR", 150))
                st.session_state["ai_ea"]   = parsed.get("ExerciseAngina", "No")
                st.session_state["ai_op"]   = float(parsed.get("Oldpeak", 0.0))
                st.session_state["ai_sl"]   = parsed.get("ST_Slope", "Upsloping")
                st.success("✅ Form filled! Review below and hit Predict.")
                st.rerun()
            except urllib.error.HTTPError as e:
                body = e.read().decode()
                if e.code == 401: st.error("❌ Invalid API key. Check console.anthropic.com")
                else: st.error(f"❌ API error {e.code}: {body[:200]}")
            except Exception as e:
                st.error(f"❌ Something went wrong: {e}")

st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  INPUT FORM
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.markdown("<div class='sec-label'>Patient Health Details</div>", unsafe_allow_html=True)
st.markdown(
    "Hover over the <b style='color:#feca57;font-size:0.82rem;'>?</b> next to any field if you're unsure what to enter.",
    unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True)

r1c1, r1c2, r1c3 = st.columns(3)
with r1c1:
    st.markdown(tip("Age", "Your current age in years.<br><b>Dataset range:</b> 28 – 77."), unsafe_allow_html=True)
    age = st.number_input("_age", min_value=18, max_value=100,
                          value=st.session_state.get("ai_age", 45),
                          label_visibility="collapsed", key="f_age")
with r1c2:
    st.markdown(tip("Sex", "Biological sex at birth.<br><b>Options:</b> Male · Female"), unsafe_allow_html=True)
    sex_opts = ["Male","Female"]
    sex = st.selectbox("_sex", sex_opts,
                       index=sex_opts.index(st.session_state.get("ai_sex","Male")),
                       label_visibility="collapsed", key="f_sex")
with r1c3:
    st.markdown(tip("Chest Pain Type",
                    "<b>Asymptomatic</b> — no chest pain (most common in heart disease).<br>"
                    "<b>Typical Angina</b> — classic pressure during activity.<br>"
                    "<b>Atypical Angina</b> — unusual pattern discomfort.<br>"
                    "<b>Non-Anginal</b> — mild, not heart-related."), unsafe_allow_html=True)
    cp_opts = ["Asymptomatic","Typical Angina","Atypical Angina","Non-Anginal Pain"]
    cp = st.selectbox("_cp", cp_opts,
                      index=cp_opts.index(st.session_state.get("ai_cp","Asymptomatic")),
                      label_visibility="collapsed", key="f_cp")

r2c1, r2c2, r2c3 = st.columns(3)
with r2c1:
    st.markdown(tip("Resting Blood Pressure",
                    "Your BP at rest (mmHg).<br><b>Normal:</b> ~120 &nbsp;·&nbsp; <b>High:</b> 140+"), unsafe_allow_html=True)
    rbp = st.number_input("_rbp", min_value=80, max_value=220,
                           value=st.session_state.get("ai_rbp", 120),
                           label_visibility="collapsed", key="f_rbp")
with r2c2:
    st.markdown(tip("Cholesterol (mg/dL)",
                    "Total blood cholesterol.<br><b>Good:</b> &lt;200 &nbsp;·&nbsp; <b>Borderline:</b> 200–239<br><b>High:</b> 240+"), unsafe_allow_html=True)
    chol = st.number_input("_chol", min_value=0, max_value=650,
                            value=st.session_state.get("ai_chol", 200),
                            label_visibility="collapsed", key="f_chol")
with r2c3:
    st.markdown(tip("Fasting Blood Sugar",
                    "Blood sugar after 8 hrs of fasting.<br><b>Yes</b> = above 120 mg/dL (diabetes risk).<br><b>No</b> = 120 or below."), unsafe_allow_html=True)
    fbs_opts = ["No","Yes"]
    fbs = st.selectbox("_fbs", fbs_opts,
                       index=fbs_opts.index(st.session_state.get("ai_fbs","No")),
                       label_visibility="collapsed", key="f_fbs")

r3c1, r3c2, r3c3 = st.columns(3)
with r3c1:
    st.markdown(tip("Resting ECG",
                    "<b>Normal</b> — heart rhythm looks fine.<br>"
                    "<b>ST-T Abnormality</b> — minor ECG irregularity.<br>"
                    "<b>LVH</b> — left ventricle appears enlarged."), unsafe_allow_html=True)
    ecg_opts = ["Normal","ST-T Abnormality","Left Ventricular Hypertrophy"]
    ecg = st.selectbox("_ecg", ecg_opts,
                       index=ecg_opts.index(st.session_state.get("ai_ecg","Normal")),
                       label_visibility="collapsed", key="f_ecg")
with r3c2:
    st.markdown(tip("Max Heart Rate",
                    "Highest HR during exercise test (bpm).<br><b>Lower than expected</b> for your age can be a warning sign.<br>Typical: 60 – 202 bpm"), unsafe_allow_html=True)
    mhr = st.number_input("_mhr", min_value=50, max_value=220,
                           value=st.session_state.get("ai_hr", 150),
                           label_visibility="collapsed", key="f_mhr")
with r3c3:
    st.markdown(tip("Exercise-Induced Angina",
                    "Did you get chest pain during physical activity?<br><b>Yes</b> — pain during exercise.<br><b>No</b> — no discomfort."), unsafe_allow_html=True)
    ea_opts = ["No","Yes"]
    ea = st.selectbox("_ea", ea_opts,
                      index=ea_opts.index(st.session_state.get("ai_ea","No")),
                      label_visibility="collapsed", key="f_ea")

r4c1, r4c2 = st.columns(2)
with r4c1:
    st.markdown(tip("Oldpeak — ST Depression",
                    "ECG change during exercise vs rest.<br><b>0</b> = healthy &nbsp;·&nbsp; <b>1–2</b> = mild<br><b>3+</b> = worth checking out."), unsafe_allow_html=True)
    op = st.slider("_op", min_value=-3.0, max_value=7.0,
                   value=float(st.session_state.get("ai_op", 1.0)),
                   step=0.1, label_visibility="collapsed", key="f_op")
with r4c2:
    st.markdown(tip("ST Slope",
                    "<b>Upsloping</b> — healthy response ✅<br><b>Flat</b> — neutral, monitor.<br><b>Downsloping</b> — may indicate reduced blood flow ⚠️"), unsafe_allow_html=True)
    sl_opts = ["Upsloping","Flat","Downsloping"]
    sl = st.selectbox("_sl", sl_opts,
                      index=sl_opts.index(st.session_state.get("ai_sl","Upsloping")),
                      label_visibility="collapsed", key="f_sl")

st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  PREDICT BUTTON
# ════════════════════════════════════════════════════════════════════════════
_, pb_col, _ = st.columns([1.6, 2, 1.4])
with pb_col:
    st.markdown('<div class="predict-btn">', unsafe_allow_html=True)
    predict = st.button("🫀  Predict Now", key="predict_btn")
    st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  RESULT
# ════════════════════════════════════════════════════════════════════════════
if predict:
    errs = []
    if rbp < 80 or rbp > 220:   errs.append("Resting BP must be 80 – 220 mmHg.")
    if chol < 0 or chol > 650:  errs.append("Cholesterol must be 0 – 650 mg/dL.")
    if mhr < 50 or mhr > 220:   errs.append("Max Heart Rate must be 50 – 220 bpm.")

    if errs:
        for e in errs: st.error(f"❌ {e}")
    elif not model_ready:
        st.error("❌ Model not ready — please refresh the page.")
    else:
        try:
            row = {"Age":age,"Sex":sex,"ChestPainType":cp,"RestingBP":rbp,
                   "Cholesterol":chol,"FastingBS":fbs,"RestingECG":ecg,
                   "MaxHR":mhr,"ExerciseAngina":ea,"Oldpeak":op,"ST_Slope":sl}
            x_in   = encode_row(row, scaler)
            pred   = model.predict(x_in)[0]
            proba  = model.predict_proba(x_in)[0]
            conf   = proba[pred] * 100
            p0, p1 = proba[0]*100, proba[1]*100

            st.markdown("<br>", unsafe_allow_html=True)

            if pred == 1:
                st.markdown(f"""
                <div class='result-danger'>
                  <span class='result-emoji'>⚠️</span>
                  <p class='result-title'>Heart Disease Risk Detected</p>
                  <p class='result-sub'>Confidence: <b style='color:#ff6b6b;font-size:1.05rem;'>{conf:.1f}%</b>
                  &nbsp;·&nbsp; Please speak with a cardiologist for a full evaluation.</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='result-safe'>
                  <span class='result-emoji'>✅</span>
                  <p class='result-title'>No Heart Disease Detected</p>
                  <p class='result-sub'>Confidence: <b style='color:#00d296;font-size:1.05rem;'>{conf:.1f}%</b>
                  &nbsp;·&nbsp; Looking good — keep up those healthy habits!</p>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            st.markdown("<div class='sec-label'>Probability Breakdown</div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class='prob-row'>
              <div class='prob-label'>No Disease</div>
              <div class='prob-bar-bg'>
                <div class='prob-bar-fill' style='width:{p0:.1f}%;background:linear-gradient(90deg,#00d296,#00cec9);'></div>
              </div>
              <div class='prob-pct' style='color:#00d296;'>{p0:.1f}%</div>
            </div>
            <div class='prob-row'>
              <div class='prob-label'>Heart Disease</div>
              <div class='prob-bar-bg'>
                <div class='prob-bar-fill' style='width:{p1:.1f}%;background:linear-gradient(90deg,#ff6b6b,#fd79a8);'></div>
              </div>
              <div class='prob-pct' style='color:#ff6b6b;'>{p1:.1f}%</div>
            </div>
            <div class='chip-row'>
              <div class='chip'><div class='chip-val'>{test_acc*100:.1f}%</div><div class='chip-lbl'>Model Accuracy</div></div>
              <div class='chip'><div class='chip-val'>{cv_mean*100:.1f}%</div><div class='chip-lbl'>CV Score</div></div>
              <div class='chip'><div class='chip-val'>918</div><div class='chip-lbl'>Training Patients</div></div>
              <div class='chip'><div class='chip-val'>11</div><div class='chip-lbl'>Features</div></div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")


# ════════════════════════════════════════════════════════════════════════════
#  ABOUT
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="div-line"></div>', unsafe_allow_html=True)
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.markdown("<div class='sec-label'>About This Project</div>", unsafe_allow_html=True)
st.markdown("""
<p style='font-size:0.88rem;color:rgba(255,255,255,0.62);line-height:1.8;'>
  This system was built by
  <span class='about-name'>Abdul Hannan</span>
  <span class='about-name'>Dawood Rizwan</span>
  <span class='about-name'>Muhammad Zaid</span>
  — a team that wanted to make heart health screening more accessible through machine learning.
</p>
<p style='font-size:0.84rem;color:rgba(255,255,255,0.45);line-height:1.8;margin-top:6px;'>
  The model was trained on <b style='color:#a29bfe;'>918 real patient records</b> using 11 clinical features
  collected from cardiac evaluations — ECG readings, blood pressure, cholesterol, stress test outcomes, and more.
  We ran five different models (Random Forest, Gradient Boosting, Logistic Regression, SVM, and KNN) and the system
  <b style='color:#a29bfe;'>automatically picks the best one</b> using 5-fold cross-validation every time it starts —
  so you're always getting the sharpest prediction available.
</p>
<p style='font-size:0.8rem;color:rgba(255,255,255,0.28);margin-top:8px;'>
  Built with Streamlit · scikit-learn · Python &nbsp;|&nbsp; Dataset: Heart Failure Prediction (Kaggle, 918 patients)
</p>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
