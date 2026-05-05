import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS (Glassmorphism + Transitions) ────────────────────────────────
st.markdown("""
<style>
/* Google Font */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

* { font-family: 'Poppins', sans-serif; }

/* Animated gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    background-size: 400% 400%;
    animation: gradientShift 12s ease infinite;
    min-height: 100vh;
}

@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ── Main title ── */
.main-title {
    text-align: center;
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(90deg, #ff6b6b, #feca57, #48dbfb, #ff9ff3);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
    animation: titleFade 1.2s ease-in;
}

@keyframes titleFade {
    from { opacity: 0; transform: translateY(-20px); }
    to   { opacity: 1; transform: translateY(0); }
}

.subtitle {
    text-align: center;
    color: rgba(255,255,255,0.6);
    font-size: 1rem;
    margin-bottom: 1.8rem;
    animation: titleFade 1.4s ease-in;
}

/* ── Glass card ── */
.glass-card {
    background: rgba(255, 255, 255, 0.07);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.15);
    padding: 28px 32px;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    animation: cardSlide 0.6s ease-out;
}

@keyframes cardSlide {
    from { opacity: 0; transform: translateY(30px); }
    to   { opacity: 1; transform: translateY(0); }
}

.glass-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 14px 40px rgba(0, 0, 0, 0.4);
}

/* ── Section headings ── */
.section-title {
    color: #feca57;
    font-size: 1.2rem;
    font-weight: 600;
    margin-bottom: 1rem;
    border-left: 4px solid #ff6b6b;
    padding-left: 12px;
}

/* ── Result cards ── */
.result-danger {
    background: linear-gradient(135deg, rgba(255,71,87,0.25), rgba(255,71,87,0.10));
    border: 1px solid rgba(255,71,87,0.5);
    border-radius: 18px;
    padding: 28px;
    text-align: center;
    animation: pulse 2s ease-in-out infinite;
    backdrop-filter: blur(10px);
}

.result-safe {
    background: linear-gradient(135deg, rgba(72,219,251,0.25), rgba(72,219,251,0.10));
    border: 1px solid rgba(72,219,251,0.5);
    border-radius: 18px;
    padding: 28px;
    text-align: center;
    animation: pulse 2s ease-in-out infinite;
    backdrop-filter: blur(10px);
}

@keyframes pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(255,107,107,0.4); }
    50%       { box-shadow: 0 0 0 14px rgba(255,107,107,0); }
}

.result-text {
    font-size: 1.8rem;
    font-weight: 700;
    color: white;
    margin: 0;
}

.result-emoji { font-size: 3rem; display: block; margin-bottom: 10px; }

/* ── Metric badges ── */
.metric-badge {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 12px;
    padding: 14px 20px;
    text-align: center;
    transition: transform 0.2s;
}
.metric-badge:hover { transform: scale(1.04); }
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #48dbfb;
}
.metric-label {
    font-size: 0.8rem;
    color: rgba(255,255,255,0.55);
    margin-top: 2px;
}

/* ── Tooltip wrapper ── */
.tooltip-wrap {
    display: inline-block;
    position: relative;
    cursor: help;
}
.tooltip-wrap .tooltip-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 18px;
    height: 18px;
    background: rgba(254, 202, 87, 0.3);
    border: 1px solid #feca57;
    border-radius: 50%;
    font-size: 11px;
    color: #feca57;
    font-weight: 700;
    margin-left: 6px;
    vertical-align: middle;
    transition: background 0.2s;
}
.tooltip-wrap:hover .tooltip-icon {
    background: rgba(254, 202, 87, 0.6);
}
.tooltip-wrap .tooltip-text {
    visibility: hidden;
    opacity: 0;
    background: rgba(15, 12, 41, 0.95);
    border: 1px solid rgba(254, 202, 87, 0.4);
    color: #fff;
    font-size: 0.78rem;
    border-radius: 10px;
    padding: 10px 14px;
    position: absolute;
    z-index: 999;
    bottom: 130%;
    left: 50%;
    transform: translateX(-50%);
    width: 220px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.5);
    transition: opacity 0.25s;
    line-height: 1.5;
}
.tooltip-wrap:hover .tooltip-text {
    visibility: visible;
    opacity: 1;
}

/* ── Streamlit element tweaks ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stSlider { color: white !important; }

div[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.85) !important;
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255,255,255,0.1);
    display: block !important;
    visibility: visible !important;
}

div[data-testid="stSidebar"] * { color: rgba(255,255,255,0.85) !important; }

/* Keep sidebar collapse/expand button visible */
button[data-testid="collapsedControl"],
div[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    color: white !important;
}

/* Ensure sidebar nav elements are visible */
section[data-testid="stSidebar"] {
    display: block !important;
    visibility: visible !important;
}

.stButton > button {
    background: linear-gradient(90deg, #ff6b6b, #ff9ff3) !important;
    color: white !important;
    border: none !important;
    border-radius: 30px !important;
    padding: 12px 40px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(255,107,107,0.4) !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(255,107,107,0.6) !important;
}

label, .stSlider label { color: rgba(255,255,255,0.85) !important; }

/* Progress / spinner color */
.stProgress > div > div { background: linear-gradient(90deg, #ff6b6b, #feca57) !important; }

/* hide streamlit branding but keep sidebar toggle */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
/* Restore the sidebar collapse button that lives inside the header */
header button[data-testid="collapsedControl"] { visibility: visible !important; }

/* scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.05); }
::-webkit-scrollbar-thumb { background: #ff6b6b; border-radius: 3px; }

/* info / warning override */
.stAlert {
    background: rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: white !important;
}

h1, h2, h3 { color: white !important; }
p, li { color: rgba(255,255,255,0.8) !important; }

.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    margin: 1.5rem 0;
}

/* step badges */
.step-badge {
    display: inline-block;
    background: linear-gradient(135deg, #ff6b6b, #ff9ff3);
    color: white;
    border-radius: 50%;
    width: 28px;
    height: 28px;
    text-align: center;
    line-height: 28px;
    font-weight: 700;
    font-size: 0.85rem;
    margin-right: 10px;
}
</style>
""", unsafe_allow_html=True)


# ─── Helper: tooltip HTML ────────────────────────────────────────────────────
def tip(label: str, hint: str) -> str:
    return f"""
    <div class='tooltip-wrap'>
        <span style='color:rgba(255,255,255,0.85);font-size:0.9rem;'>{label}</span>
        <span class='tooltip-icon'>?</span>
        <span class='tooltip-text'>{hint}</span>
    </div>
    """


# ─── Load & Prepare Data ─────────────────────────────────────────────────────
@st.cache_data
def load_and_prepare_data():
    try:
        df = pd.read_csv("heart.csv")
    except FileNotFoundError:
        st.error("❌ heart.csv not found! Make sure the file is in the same folder as app.py")
        st.stop()

    df_enc = df.copy()

    cat_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()          # fresh encoder per column — fixes reference bug
        df_enc[col] = le.fit_transform(df_enc[col])
        encoders[col] = le

    X = df_enc.drop("HeartDisease", axis=1)
    y = df_enc["HeartDisease"]

    return X, y, encoders, df


# ─── Train Model ────────────────────────────────────────────────────────────
@st.cache_data   # cache_data is correct for serialisable return values
def train_model(model_name: str, do_tuning: bool):
    X, y, encoders, _ = load_and_prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fit scaler on training data only — prevents data leakage
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled  = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns)

    models_config = {
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5]
            }
        },
        "Gradient Boosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {
                "n_estimators": [50, 100],
                "learning_rate": [0.05, 0.1, 0.2],
                "max_depth": [3, 5]
            }
        },
        "Logistic Regression": {
            "model": LogisticRegression(random_state=42, max_iter=1000),
            "params": {
                "C": [0.01, 0.1, 1, 10],
                "solver": ["lbfgs", "liblinear"]
            }
        },
        "SVM": {
            "model": SVC(probability=True, random_state=42),
            "params": {
                "C": [0.1, 1, 10],
                "kernel": ["rbf", "linear"],
                "gamma": ["scale", "auto"]
            }
        },
        "KNN": {
            "model": KNeighborsClassifier(),
            "params": {
                "n_neighbors": [3, 5, 7, 9],
                "weights": ["uniform", "distance"],
                "metric": ["minkowski", "euclidean", "manhattan"]
            }
        }
    }

    cfg = models_config[model_name]

    if do_tuning:
        grid = GridSearchCV(
            cfg["model"], cfg["params"],
            cv=5, scoring="accuracy", n_jobs=-1
        )
        grid.fit(X_train_scaled, y_train)
        best_model = grid.best_estimator_
        best_params = grid.best_params_
    else:
        best_model = cfg["model"]
        best_model.fit(X_train_scaled, y_train)
        best_params = {}

    y_pred  = best_model.predict(X_test_scaled)
    y_prob  = best_model.predict_proba(X_test_scaled)[:, 1]
    acc     = accuracy_score(y_test, y_pred)
    auc     = roc_auc_score(y_test, y_prob)

    # CV on the full (unscaled) feature set — note: if tuning was on,
    # these scores are optimistically biased because the best_model was
    # already selected on the same data.
    X_full_scaled = pd.DataFrame(
        StandardScaler().fit_transform(X), columns=X.columns
    )
    cv_scores = cross_val_score(best_model, X_full_scaled, y, cv=5, scoring="accuracy")
    cv_biased = do_tuning   # flag exposed to UI

    report = classification_report(y_test, y_pred, output_dict=True)
    cm     = confusion_matrix(y_test, y_pred)

    return best_model, scaler, encoders, acc, auc, cv_scores, cv_biased, best_params, report, cm, X_train_scaled.shape[0]


# ─── Encode Single Input ────────────────────────────────────────────────────
def encode_input(row: dict, scaler, encoders):
    mapping_sex           = {"Male": "M",   "Female": "F"}
    mapping_chest         = {
        "Typical Angina (TA)": "TA",
        "Atypical Angina (ATA)": "ATA",
        "Non-Anginal Pain (NAP)": "NAP",
        "Asymptomatic (ASY)": "ASY"
    }
    mapping_ecg           = {"Normal": "Normal", "ST-T Abnormality (ST)": "ST", "Left Ventricular Hypertrophy (LVH)": "LVH"}
    mapping_angina        = {"Yes": "Y", "No": "N"}
    mapping_slope         = {"Upsloping (Up)": "Up", "Flat": "Flat", "Downsloping (Down)": "Down"}

    raw = {
        "Age":            row["Age"],
        "Sex":            mapping_sex[row["Sex"]],
        "ChestPainType":  mapping_chest[row["ChestPainType"]],
        "RestingBP":      row["RestingBP"],
        "Cholesterol":    row["Cholesterol"],
        "FastingBS":      1 if row["FastingBS"] == "Yes (> 120 mg/dl)" else 0,
        "RestingECG":     mapping_ecg[row["RestingECG"]],
        "MaxHR":          row["MaxHR"],
        "ExerciseAngina": mapping_angina[row["ExerciseAngina"]],
        "Oldpeak":        row["Oldpeak"],
        "ST_Slope":       mapping_slope[row["ST_Slope"]],
    }

    encoded = [
        raw["Age"],
        encoders["Sex"].transform([raw["Sex"]])[0],
        encoders["ChestPainType"].transform([raw["ChestPainType"]])[0],
        raw["RestingBP"],
        raw["Cholesterol"],
        raw["FastingBS"],
        encoders["RestingECG"].transform([raw["RestingECG"]])[0],
        raw["MaxHR"],
        encoders["ExerciseAngina"].transform([raw["ExerciseAngina"]])[0],
        raw["Oldpeak"],
        encoders["ST_Slope"].transform([raw["ST_Slope"]])[0],
    ]

    arr = np.array(encoded).reshape(1, -1)
    cols = ["Age","Sex","ChestPainType","RestingBP","Cholesterol",
            "FastingBS","RestingECG","MaxHR","ExerciseAngina","Oldpeak","ST_Slope"]
    df_input = pd.DataFrame(arr, columns=cols)
    df_scaled = scaler.transform(df_input)
    return df_scaled


# ════════════════════════════════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<h1 class="main-title">🫀 Heart Disease Prediction System</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Tell us a little about yourself and we\'ll do the rest — powered by Machine Learning</p>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — Model Settings
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Model Settings")
    st.markdown("---")

    model_choice = st.selectbox(
        "Choose a Machine Learning Model",
        ["Random Forest", "Gradient Boosting", "Logistic Regression", "SVM", "KNN"],
        help="Each model learns patterns differently. Random Forest is usually the most reliable to start with!"
    )

    do_tuning = st.toggle(
        "🔧 Enable Hyperparameter Tuning",
        value=False,
        help="This tries many different settings to find the best ones. Takes a bit longer but worth it!"
    )

    if do_tuning:
        st.info("⏳ Tuning is ON — training might take 30–60 seconds. Grab a coffee! ☕")

    st.markdown("---")
    st.markdown("### 📌 What is this app?")
    st.markdown("""
This tool uses real patient data to predict whether someone might have heart disease.
Just fill in your health details on the right and hit **Predict**!

> ⚠️ *This is a student project for educational purposes — not a medical diagnosis.*
    """)

    st.markdown("---")
    st.markdown("### 🧠 Models Available")
    st.markdown("""
- 🌲 **Random Forest** — votes from many trees
- 📈 **Gradient Boosting** — learns from mistakes step-by-step  
- 📊 **Logistic Regression** — simple, fast, reliable  
- 🔵 **SVM** — draws a boundary between healthy & at-risk  
- 👣 **KNN** — predicts using similar past examples
    """)


# ════════════════════════════════════════════════════════════════════════════
#  TRAIN MODEL SECTION
# ════════════════════════════════════════════════════════════════════════════
col_btn, col_status = st.columns([1, 3])

with col_btn:
    train_clicked = st.button("🚀 Train Model")

if "model_ready" not in st.session_state:
    st.session_state.model_ready = False

if train_clicked:
    with st.spinner(f"Training {model_choice}... hold tight! 🧠"):
        try:
            (model, scaler, encoders, acc, auc, cv_scores,
             cv_biased, best_params, report, cm, n_train) = train_model(model_choice, do_tuning)

            st.session_state.model       = model
            st.session_state.scaler      = scaler
            st.session_state.encoders    = encoders
            st.session_state.acc         = acc
            st.session_state.auc         = auc
            st.session_state.cv_scores   = cv_scores
            st.session_state.cv_biased   = cv_biased
            st.session_state.best_params = best_params
            st.session_state.report      = report
            st.session_state.cm          = cm
            st.session_state.n_train     = n_train
            st.session_state.model_name  = model_choice
            st.session_state.model_ready = True
            st.success("✅ Model trained successfully!")
        except Exception as e:
            st.error(f"❌ Something went wrong while training: {e}")

# ─── Model Performance Cards ─────────────────────────────────────────────────
if st.session_state.model_ready:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(f'<p class="section-title">📊 {st.session_state.model_name} — Performance Report</p>', unsafe_allow_html=True)

    cv_mean  = st.session_state.cv_scores.mean()
    cv_std   = st.session_state.cv_scores.std()
    precision = st.session_state.report["weighted avg"]["precision"]
    recall    = st.session_state.report["weighted avg"]["recall"]
    f1        = st.session_state.report["weighted avg"]["f1-score"]

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown(f"""
        <div class='metric-badge'>
            <div class='metric-value'>{st.session_state.acc*100:.1f}%</div>
            <div class='metric-label'>Test Accuracy</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        cv_label = "CV Accuracy*" if st.session_state.cv_biased else f"CV Accuracy (±{cv_std*100:.1f}%)"
        st.markdown(f"""
        <div class='metric-badge'>
            <div class='metric-value'>{cv_mean*100:.1f}%</div>
            <div class='metric-label'>{cv_label}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class='metric-badge'>
            <div class='metric-value'>{precision:.2f}</div>
            <div class='metric-label'>Precision</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class='metric-badge'>
            <div class='metric-value'>{recall:.2f}</div>
            <div class='metric-label'>Recall</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
        <div class='metric-badge'>
            <div class='metric-value'>{f1:.2f}</div>
            <div class='metric-label'>F1 Score</div>
        </div>""", unsafe_allow_html=True)
    with c6:
        st.markdown(f"""
        <div class='metric-badge'>
            <div class='metric-value' style='color:#ff9ff3;'>{st.session_state.auc:.3f}</div>
            <div class='metric-label'>AUC-ROC</div>
        </div>""", unsafe_allow_html=True)

    if st.session_state.cv_biased:
        st.caption("⚠️ *CV scores may be optimistically biased — the tuned model was selected using the same dataset.")

    if st.session_state.best_params:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**🔧 Best Hyperparameters Found:**")
        params_str = "  |  ".join([f"`{k}` = **{v}**" for k, v in st.session_state.best_params.items()])
        st.markdown(params_str)

    # ── Confusion Matrix ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**🔢 Confusion Matrix**")
    cm_df = pd.DataFrame(
        st.session_state.cm,
        index=["Actual: No Disease", "Actual: Disease"],
        columns=["Predicted: No Disease", "Predicted: Disease"]
    )
    st.dataframe(cm_df, use_container_width=True)

    # KNN note — single occurrence
    if st.session_state.model_name == "KNN":
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**👣 KNN Note:** KNN makes predictions by comparing you to the most similar patients in the dataset, using nearby neighbors to decide the likely outcome.")

    st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  PREDICTION FORM
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<p class="section-title">🩺 Enter Your Health Details</p>', unsafe_allow_html=True)
st.markdown("Fill in the details below as accurately as you can. Not sure about something? Hover over the **?** icon next to each field for a quick guide.")
st.markdown("<br>", unsafe_allow_html=True)

# Row 1
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(tip("Age", "Your current age in years"), unsafe_allow_html=True)
    age = st.number_input("", min_value=18, max_value=100, value=45, key="age", label_visibility="collapsed")

with col2:
    st.markdown(tip("Biological Sex", "Select the sex assigned at birth — Male or Female."), unsafe_allow_html=True)
    sex = st.selectbox("", ["Male", "Female"], key="sex", label_visibility="collapsed")

with col3:
    st.markdown(tip("Chest Pain Type", "TA = Typical chest pain during activity.<br>ATA = Chest pain that doesn't quite fit the usual pattern.<br>NAP = Mild, non-heart-related discomfort.<br>ASY = No chest pain at all (most common in heart disease!)."), unsafe_allow_html=True)
    chest_pain = st.selectbox("", [
        "Typical Angina (TA)", "Atypical Angina (ATA)",
        "Non-Anginal Pain (NAP)", "Asymptomatic (ASY)"
    ], key="cp", label_visibility="collapsed")

# Row 2
col4, col5, col6 = st.columns(3)

with col4:
    st.markdown(tip("Resting Blood Pressure (mmHg)", "Your blood pressure when at rest.<br>Normal: around 120.<br>High (hypertension): 140+."), unsafe_allow_html=True)
    resting_bp = st.number_input("", min_value=80, max_value=220, value=120, key="rbp", label_visibility="collapsed")

with col5:
    st.markdown(tip("Cholesterol (mg/dL)", "Total cholesterol level from a blood test.<br>Normal: below 200.<br>Borderline high: 200 – 239.<br>High: 240+."), unsafe_allow_html=True)
    cholesterol = st.number_input("", min_value=0, max_value=650, value=200, key="chol", label_visibility="collapsed")

with col6:
    st.markdown(tip("Fasting Blood Sugar", "Your blood sugar level after fasting (not eating for 8 hours).<br>Yes = Sugar above 120 mg/dL (possible diabetes risk).<br>No = Normal levels."), unsafe_allow_html=True)
    fasting_bs = st.selectbox("", ["No (≤ 120 mg/dl)", "Yes (> 120 mg/dl)"], key="fbs", label_visibility="collapsed")

# Row 3
col7, col8, col9 = st.columns(3)

with col7:
    st.markdown(tip("Resting ECG", "Result of your heart's electrical activity at rest.<br>Normal = All good.<br>ST = Minor abnormality in heartbeat pattern.<br>LVH = Left side of heart looks enlarged."), unsafe_allow_html=True)
    resting_ecg = st.selectbox("", [
        "Normal", "ST-T Abnormality (ST)", "Left Ventricular Hypertrophy (LVH)"
    ], key="recg", label_visibility="collapsed")

with col8:
    st.markdown(tip("Max Heart Rate Achieved", "The highest your heart rate got during an exercise test.<br>A lower max HR for your age can be a warning sign.<br>Typical range: 60 – 202 bpm."), unsafe_allow_html=True)
    max_hr = st.number_input("", min_value=50, max_value=220, value=150, key="mhr", label_visibility="collapsed")

with col9:
    st.markdown(tip("Exercise-Induced Angina", "Did you feel chest pain or tightness during physical activity?<br>Yes = Pain appeared during exercise.<br>No = No discomfort during exercise."), unsafe_allow_html=True)
    exercise_angina = st.selectbox("", ["No", "Yes"], key="ea", label_visibility="collapsed")

# Row 4
col10, col11 = st.columns(2)

with col10:
    st.markdown(tip("Oldpeak (ST Depression)", "A measurement from your ECG during exercise compared to rest.<br>0 = No change (good).<br>1 – 2 = Mild change.<br>3+ = More significant, worth checking out."), unsafe_allow_html=True)
    oldpeak = st.slider("", min_value=-3.0, max_value=7.0, value=1.0, step=0.1, key="op", label_visibility="collapsed")

with col11:
    st.markdown(tip("ST Slope", "How the ST segment on your ECG changes during peak exercise.<br>Up = Healthy response (good sign).<br>Flat = Neutral, might need attention.<br>Down = Could indicate reduced blood flow."), unsafe_allow_html=True)
    st_slope = st.selectbox("", [
        "Upsloping (Up)", "Flat", "Downsloping (Down)"
    ], key="sl", label_visibility="collapsed")

st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
#  PREDICT BUTTON
# ════════════════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
col_pred = st.columns([1, 2, 1])[1]

with col_pred:
    predict_clicked = st.button("🫀 Predict My Heart Health")

if predict_clicked:
    if not st.session_state.model_ready:
        st.warning("⚠️ Please train the model first! Click **Train Model** above before predicting.")
    else:
        # Input validation
        errors = []
        if resting_bp < 80 or resting_bp > 220:
            errors.append("Resting BP should be between 80 and 220 mmHg.")
        if cholesterol < 0 or cholesterol > 650:
            errors.append("Cholesterol should be between 0 and 650 mg/dL.")
        if max_hr < 50 or max_hr > 220:
            errors.append("Max Heart Rate should be between 50 and 220 bpm.")
        if cholesterol == 0:
            st.warning("⚠️ Cholesterol is 0 — this likely means the value was missing in the source data. The prediction may be less reliable. Please enter your actual cholesterol reading if available.")

        if errors:
            for err in errors:
                st.error(f"❌ {err}")
        else:
            user_input = {
                "Age": age, "Sex": sex, "ChestPainType": chest_pain,
                "RestingBP": resting_bp, "Cholesterol": cholesterol,
                "FastingBS": fasting_bs, "RestingECG": resting_ecg,
                "MaxHR": max_hr, "ExerciseAngina": exercise_angina,
                "Oldpeak": oldpeak, "ST_Slope": st_slope
            }

            try:
                input_scaled = encode_input(user_input, st.session_state.scaler, st.session_state.encoders)
                prediction   = st.session_state.model.predict(input_scaled)[0]
                probability  = st.session_state.model.predict_proba(input_scaled)[0]
                confidence   = probability[prediction] * 100

                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

                if prediction == 1:
                    st.markdown(f"""
                    <div class='result-danger'>
                        <span class='result-emoji'>⚠️</span>
                        <p class='result-text'>Heart Disease Risk Detected</p>
                        <p style='color:rgba(255,255,255,0.75);margin-top:10px;font-size:0.95rem;'>
                            The model is <strong>{confidence:.1f}% confident</strong> based on the details you provided.
                        </p>
                        <p style='color:rgba(255,100,100,0.9);font-size:0.85rem;margin-top:8px;'>
                            This doesn't mean you definitely have heart disease — please talk to a doctor for a proper check-up! 🏥
                        </p>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='result-safe'>
                        <span class='result-emoji'>✅</span>
                        <p class='result-text'>No Heart Disease Detected</p>
                        <p style='color:rgba(255,255,255,0.75);margin-top:10px;font-size:0.95rem;'>
                            The model is <strong>{confidence:.1f}% confident</strong> — looking good based on your data!
                        </p>
                        <p style='color:rgba(72,219,251,0.9);font-size:0.85rem;margin-top:8px;'>
                            Keep up the healthy habits — regular exercise, balanced diet, and annual check-ups go a long way! 💪
                        </p>
                    </div>""", unsafe_allow_html=True)

                # Risk breakdown
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown('<p class="section-title">🔍 Probability Breakdown</p>', unsafe_allow_html=True)

                pc1, pc2 = st.columns(2)
                with pc1:
                    st.markdown(f"""
                    <div class='metric-badge'>
                        <div class='metric-value' style='color:#48dbfb;'>{probability[0]*100:.1f}%</div>
                        <div class='metric-label'>Probability: No Disease</div>
                    </div>""", unsafe_allow_html=True)
                with pc2:
                    st.markdown(f"""
                    <div class='metric-badge'>
                        <div class='metric-value' style='color:#ff6b6b;'>{probability[1]*100:.1f}%</div>
                        <div class='metric-label'>Probability: Heart Disease</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.progress(int(probability[1] * 100))
                st.caption(f"⬆️ Risk meter — {probability[1]*100:.1f}% toward heart disease risk")
                st.markdown('</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Prediction failed: {e}. Please check your inputs and try again.")


# ════════════════════════════════════════════════════════════════════════════
#  ABOUT / DISCLAIMER
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<p class="section-title">📋 About This Project</p>', unsafe_allow_html=True)
st.markdown("""
Hey there! 👋 This project was built using a heart disease dataset from **Kaggle** 
containing 918 real patient records. We trained several classification models and tuned them to 
give the most accurate prediction possible.

The features used are standard ones you'd get from a basic cardiac check-up — things like blood 
pressure, cholesterol, ECG readings, and exercise stress test results.

**Models we tested:** Random Forest · Gradient Boosting · Logistic Regression · SVM · KNN 
**Dataset:** 918 rows × 12 features (Heart Failure Prediction Dataset""") 


st.markdown('</div>', unsafe_allow_html=True)
