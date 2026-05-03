# Projet : Analyse et Recommandation de Cultures Agricoles
# Encadrant : Pr. Mohammed KAICER


import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from scipy.stats import weibull_min, kstest
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, label_binarize
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    silhouette_score, davies_bouldin_score, adjusted_rand_score,
    f1_score, recall_score, log_loss, roc_auc_score, roc_curve,
    cohen_kappa_score, matthews_corrcoef, ndcg_score
)

# SSIM manuel (sans scikit-image)
def structural_similarity(img1, img2, data_range=1.0):
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    mu1, mu2 = img1.mean(), img2.mean()
    sigma1_sq = img1.var()
    sigma2_sq = img2.var()
    sigma12   = np.mean((img1 - mu1) * (img2 - mu2))
    num = (2*mu1*mu2 + C1) * (2*sigma12 + C2)
    den = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
    return float(num / den)


# PAGE CONFIG & CSS
st.set_page_config(
    page_title="🌾 Crop Recommendation Analytics",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── CSS Variables ── */
:root {
    --green-dark:   #064e3b;
    --green-mid:    #059669;
    --green-light:  #6ee7b7;
    --gold:         #f59e0b;
    --navy:         #0f172a;
    --slate:        #1e293b;
    --slate-mid:    #334155;
    --slate-light:  #64748b;
    --cream:        #fefce8;
    --white:        #ffffff;
    --card-bg:      #f8fafc;
    --border:       #e2e8f0;
    --red:          #ef4444;
    --purple:       #8b5cf6;
    --blue:         #3b82f6;
}

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    color: var(--navy) !important;
}

/* ── Hide Streamlit default elements ── */
#MainMenu { visibility:  visible; }
footer { visibility: hidden; }

/* IMPORTANT : garder le header visible */
header { visibility: visible !important; }
.block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; }

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, #064e3b 0%, #065f46 40%, #047857 70%, #059669 100%);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(6,78,59,0.35);
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: rgba(255,255,255,0.05);
    border-radius: 50%;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -80px; left: 30%;
    width: 250px; height: 250px;
    background: rgba(245,158,11,0.1);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Playfair Display', serif !important;
    font-size: 2.8rem !important;
    font-weight: 900 !important;
    color: #ffffff !important;
    margin: 0 !important;
    line-height: 1.1 !important;
    text-shadow: 0 2px 20px rgba(0,0,0,0.3);
}
.hero-subtitle {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1.05rem !important;
    color: #a7f3d0 !important;
    margin-top: 0.6rem !important;
    font-weight: 300 !important;
    letter-spacing: 0.3px;
}
.hero-badges {
    margin-top: 1.4rem;
    display: flex;
    gap: 0.6rem;
    flex-wrap: wrap;
}
.badge {
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    color: #ffffff !important;
    padding: 0.3rem 0.9rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 500;
    backdrop-filter: blur(4px);
}

/* ── Section Headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin: 2rem 0 1.2rem 0;
    padding-bottom: 0.7rem;
    border-bottom: 2px solid var(--border);
}
.section-icon {
    background: linear-gradient(135deg, var(--green-dark), var(--green-mid));
    color: white;
    width: 40px; height: 40px;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem;
    box-shadow: 0 4px 12px rgba(5,150,105,0.3);
}
.section-title {
    font-family: 'Playfair Display', serif !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: var(--navy) !important;
    margin: 0 !important;
}

/* ── Metric Cards ── */
.metrics-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.metric-card {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05);
    transition: transform 0.2s, box-shadow 0.2s;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--green-mid), var(--gold));
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.1);
}
.metric-label {
    font-size: 0.78rem;
    color: var(--slate-light);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.9rem !important;
    font-weight: 600 !important;
    color: var(--green-dark) !important;
    line-height: 1;
}
.metric-delta {
    font-size: 0.78rem;
    margin-top: 0.3rem;
    color: var(--slate-light);
}
.metric-delta.good { color: var(--green-mid); }
.metric-delta.bad  { color: var(--red); }

/* ── Model Cards ── */
.model-card {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05);
    margin-bottom: 1rem;
    transition: all 0.25s;
}
.model-card:hover {
    box-shadow: 0 8px 30px rgba(5,150,105,0.15);
    border-color: var(--green-light);
}
.model-card-title {
    font-family: 'Playfair Display', serif !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    color: var(--navy) !important;
    margin-bottom: 0.3rem !important;
}
.model-card-type {
    font-size: 0.75rem;
    color: var(--slate-light);
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 0.8rem;
}

/* ── Progress bars ── */
.progress-bar-wrap {
    background: #f1f5f9;
    border-radius: 999px;
    height: 8px;
    margin: 0.3rem 0;
    overflow: hidden;
}
.progress-bar-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--green-mid), var(--gold));
    transition: width 0.8s ease;
}

/* ── Info boxes ── */
.info-box {
    background: linear-gradient(135deg, #ecfdf5, #f0fdf4);
    border: 1px solid #bbf7d0;
    border-left: 4px solid var(--green-mid);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    font-size: 0.9rem;
    color: var(--navy);
}
.formula-box {
    background: var(--navy);
    border-radius: 10px;
    padding: 1rem 1.4rem;
    margin: 0.8rem 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #a7f3d0;
    border-left: 4px solid var(--gold);
}
.warning-box {
    background: linear-gradient(135deg, #fffbeb, #fef3c7);
    border: 1px solid #fde68a;
    border-left: 4px solid var(--gold);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    font-size: 0.9rem;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--green-dark) 0%, #065f46 100%) !important;
}
[data-testid="stSidebar"] * {
    color: #d1fae5 !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stRadio label {
    color: #a7f3d0 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
    font-family: 'Playfair Display', serif !important;
}
.sidebar-logo {
    text-align: center;
    padding: 1rem 0 1.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.15);
    margin-bottom: 1.5rem;
}
.sidebar-logo-icon {
    font-size: 3rem;
    display: block;
    margin-bottom: 0.5rem;
}
.sidebar-logo-text {
    font-family: 'Playfair Display', serif !important;
    font-size: 1.3rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
}
.sidebar-logo-sub {
    font-size: 0.75rem !important;
    color: #6ee7b7 !important;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.3rem;
    background: var(--card-bg);
    border-radius: 12px;
    padding: 0.3rem;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 0.5rem 1.2rem;
    font-weight: 500;
    font-size: 0.88rem;
    color: var(--slate-light) !important;
    background: transparent;
    border: none;
}
.stTabs [aria-selected="true"] {
    background: var(--white) !important;
    color: var(--green-dark) !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    font-weight: 600 !important;
}

/* ── Table styling ── */
.stDataFrame {
    border-radius: 12px !important;
    overflow: hidden;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--green-dark), var(--green-mid)) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 12px rgba(5,150,105,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(5,150,105,0.4) !important;
}

/* ── Comparison table ── */
.comparison-table {
    width: 100%;
    border-collapse: collapse;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
}
.comparison-table th {
    background: var(--green-dark);
    color: white;
    padding: 0.9rem 1rem;
    font-weight: 600;
    font-size: 0.85rem;
    text-align: center;
    letter-spacing: 0.3px;
}
.comparison-table td {
    padding: 0.75rem 1rem;
    text-align: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    border-bottom: 1px solid var(--border);
}
.comparison-table tr:nth-child(even) td { background: #f8fafc; }
.comparison-table tr:hover td { background: #ecfdf5; }
.best-value { color: var(--green-mid); font-weight: 700; }
.worst-value { color: var(--red); }

/* ── Prediction card ── */
.prediction-result {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    color: white;
    box-shadow: 0 12px 40px rgba(6,78,59,0.4);
}
.prediction-crop {
    font-family: 'Playfair Display', serif;
    font-size: 2.5rem;
    font-weight: 900;
    color: #6ee7b7;
    margin: 0.5rem 0;
}
.prediction-conf {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    color: #fde68a;
}

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 2rem;
    margin-top: 3rem;
    border-top: 1px solid var(--border);
    color: var(--slate-light);
    font-size: 0.82rem;
}
 /* ── RESPONSIVE DESIGN ── */

/* Tablettes */
@media (max-width: 1024px) {
    .metrics-row {
        grid-template-columns: repeat(2, 1fr) !important;
    }

    .hero-title {
        font-size: 2rem !important;
    }
}

/* Mobiles */
@media (max-width: 768px) {
    .metrics-row {
        grid-template-columns: 1fr !important;
    }

    .hero-banner {
        padding: 1.5rem !important;
    }

    .hero-title {
        font-size: 1.6rem !important;
    }

    .hero-subtitle {
        font-size: 0.9rem !important;
    }

    .section-title {
        font-size: 1.2rem !important;
    }

    .metric-value {
        font-size: 1.4rem !important;
    }

    .model-card {
        padding: 1rem !important;
    }

    .prediction-crop {
        font-size: 1.8rem !important;
    }
}

/* Très petits écrans */
@media (max-width: 480px) {
    .hero-title {
        font-size: 1.3rem !important;
    }

    .hero-subtitle {
        font-size: 0.8rem !important;
    }

    .badge {
        font-size: 0.65rem !important;
        padding: 0.2rem 0.6rem !important;
    }
}           
</style>
""", unsafe_allow_html=True)


# DATA & MODEL LOADING (cached)
FEATURES = ['Azote_N', 'Phosphore_P', 'Potassium_K', 'Temperature', 'Humidite', 'pH', 'Pluviometrie']
PALETTE  = ['#064e3b','#059669','#f59e0b','#ef4444','#8b5cf6','#06b6d4','#ec4899','#84cc16']

@st.cache_data
def load_data():
    df = pd.read_csv('Crop_Recommendation.csv')
    df = df.rename(columns={
        'Nitrogen':'Azote_N','Phosphorus':'Phosphore_P','Potassium':'Potassium_K',
        'Temperature':'Temperature','Humidity':'Humidite',
        'pH_Value':'pH','Rainfall':'Pluviometrie','Crop':'Culture'
    })
    return df

@st.cache_resource
def train_all_models(df):
    le = LabelEncoder()
    df['Culture_enc'] = le.fit_transform(df['Culture'])
    X_raw = df[FEATURES].values
    y     = df['Culture_enc'].values
    y_str = df['Culture'].values

    scaler = StandardScaler()
    X_std  = scaler.fit_transform(X_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X_std, y, test_size=0.25, random_state=42, stratify=y)
    X_train_raw, X_test_raw, _, _ = train_test_split(
        X_raw, y, test_size=0.25, random_state=42, stratify=y)

    n_classes = len(le.classes_)
    y_bin = label_binarize(y_test, classes=list(range(n_classes)))

    # Models
    dt  = DecisionTreeClassifier(max_depth=8, criterion='gini', min_samples_split=5, random_state=42)
    rf  = RandomForestClassifier(n_estimators=200, max_depth=12, max_features='sqrt', random_state=42, n_jobs=-1)
    nb  = GaussianNB()
    knn = KNeighborsClassifier(n_neighbors=7, metric='euclidean')

    dt.fit(X_train, y_train);  rf.fit(X_train, y_train)
    nb.fit(X_train, y_train);  knn.fit(X_train, y_train)

    pred_dt  = dt.predict(X_test);   prob_dt  = dt.predict_proba(X_test)
    pred_rf  = rf.predict(X_test);   prob_rf  = rf.predict_proba(X_test)
    pred_nb  = nb.predict(X_test);   prob_nb  = nb.predict_proba(X_test)
    pred_knn = knn.predict(X_test);  prob_knn = knn.predict_proba(X_test)

    # Markov-RF
    tm = np.zeros((n_classes, n_classes))
    for i in range(len(y)-1):
        tm[y[i], y[i+1]] += 1
    tm = tm / (tm.sum(axis=1, keepdims=True) + 1e-9)
    np.random.seed(42)
    prior = np.random.randint(0, n_classes, size=len(X_test))
    prob_mrf = 0.7*prob_rf + 0.3*tm[prior]
    prob_mrf /= prob_mrf.sum(axis=1, keepdims=True)
    pred_mrf = np.argmax(prob_mrf, axis=1)

    # K-Means
    km = KMeans(n_clusters=22, random_state=42, n_init=10)
    km.fit(X_std)
    sil_km = silhouette_score(X_std, km.labels_, sample_size=500)
    dbi_km = davies_bouldin_score(X_std, km.labels_)

    # Weibull
    weibull_params = {}
    for feat in FEATURES:
        data_pos = np.abs(X_train_raw[:, FEATURES.index(feat)]) + 0.01
        c, loc, scale = weibull_min.fit(data_pos, floc=0)
        weibull_params[feat] = (c, loc, scale)

    def weibull_features(Xr):
        W = np.zeros((len(Xr), len(FEATURES)))
        for j, feat in enumerate(FEATURES):
            c, loc, scale = weibull_params[feat]
            vals = np.abs(Xr[:, j]) + 0.01
            W[:, j] = np.clip(weibull_min.logpdf(vals, c, loc=loc, scale=scale), -100, 0)
        return W

    Xw_train = StandardScaler().fit_transform(np.hstack([X_train, weibull_features(X_train_raw)]))
    sc_w = StandardScaler().fit(np.hstack([X_train, weibull_features(X_train_raw)]))
    rf_w = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    rf_w.fit(sc_w.transform(np.hstack([X_train, weibull_features(X_train_raw)])), y_train)
    Xw_test = sc_w.transform(np.hstack([X_test, weibull_features(X_test_raw)]))
    pred_rfw = rf_w.predict(Xw_test); prob_rfw = rf_w.predict_proba(Xw_test)

    # Benford
    def benford_score(v):
        s = str(abs(float(v))).replace('0.','').replace('.','').lstrip('0')
        if not s or not s[0].isdigit() or s[0]=='0': return np.log10(1+1/5)
        return np.log10(1+1/int(s[0]))
    def benford_features(Xr):
        B = np.zeros((len(Xr), len(FEATURES)))
        for i in range(len(Xr)):
            for j in range(len(FEATURES)):
                B[i,j] = benford_score(Xr[i,j])
        return B

    sc_b = StandardScaler().fit(np.hstack([X_train, benford_features(X_train_raw)]))
    nb_b = GaussianNB()
    nb_b.fit(sc_b.transform(np.hstack([X_train, benford_features(X_train_raw)])), y_train)
    Xb_test = sc_b.transform(np.hstack([X_test, benford_features(X_test_raw)]))
    pred_nbb = nb_b.predict(Xb_test); prob_nbb = nb_b.predict_proba(Xb_test)

    # Collect results
    models_pred = {
        'Decision Tree': (pred_dt, prob_dt),
        'Random Forest': (pred_rf, prob_rf),
        'Naive Bayes':   (pred_nb, prob_nb),
        'Markov-RF':     (pred_mrf, prob_mrf),
        'KNN':           (pred_knn, prob_knn),
        'RF + Weibull':  (pred_rfw, prob_rfw),
        'NB + Benford':  (pred_nbb, prob_nbb),
    }

    def rmsle(yt, yp):
        return np.sqrt(np.mean((np.log1p(np.array(yp,float)) - np.log1p(np.array(yt,float)))**2))
    def ssim_cm(yt, yp):
        cm = confusion_matrix(yt, yp, labels=list(range(n_classes))).astype(float)
        cm /= (cm.sum(axis=1, keepdims=True) + 1e-9)
        return structural_similarity(cm, np.eye(n_classes), data_range=1.0)
    def asd_score(yt, yp):
        Xl = PCA(n_components=3).fit_transform(X_test)
        dists = []
        for c in np.unique(yt):
            mt, mp = yt==c, yp==c
            if mt.sum()==0 or mp.sum()==0: continue
            dists.append(np.linalg.norm(Xl[mt].mean(0) - Xl[mp].mean(0)))
        return np.mean(dists)

    summary = {}
    for name, (yp, proba) in models_pred.items():
        auc = roc_auc_score(y_bin, proba, multi_class='ovr', average='macro')
        fprs = [roc_curve(y_bin[:,k], proba[:,k])[0].mean() for k in range(n_classes)]
        summary[name] = {
            'Accuracy':   accuracy_score(y_test, yp),
            'F1 / Dice':  f1_score(y_test, yp, average='macro'),
            'Recall':     recall_score(y_test, yp, average='macro'),
            'AUC':        auc,
            'MCC':        matthews_corrcoef(y_test, yp),
            "Cohen's κ":  cohen_kappa_score(y_test, yp),
            'Log Loss':   log_loss(y_test, proba),
            'FPR':        np.mean(fprs),
            'RMSLE':      rmsle(y_test, yp),
            'SSIM':       ssim_cm(y_test, yp),
            'ASD':        asd_score(y_test, yp),
            'NDCG':       ndcg_score(np.eye(n_classes)[y_test], proba),
        }

    return dict(
        df=df, le=le, scaler=scaler, X_std=X_std, X_raw=X_raw,
        y=y, y_str=y_str, y_test=y_test, y_bin=y_bin,
        X_train=X_train, X_test=X_test,
        X_train_raw=X_train_raw, X_test_raw=X_test_raw,
        n_classes=n_classes,
        dt=dt, rf=rf, nb=nb, knn=knn, km=km,
        pred_dt=pred_dt, pred_rf=pred_rf, pred_nb=pred_nb,
        pred_knn=pred_knn, pred_mrf=pred_mrf,
        prob_dt=prob_dt, prob_rf=prob_rf, prob_nb=prob_nb,
        prob_knn=prob_knn, prob_mrf=prob_mrf,
        sil_km=sil_km, dbi_km=dbi_km,
        weibull_params=weibull_params,
        summary=summary,
        y_train=y_train,
        acc_knn=accuracy_score(y_test, pred_knn),
        acc_dt=accuracy_score(y_test, pred_dt),
        acc_rf=accuracy_score(y_test, pred_rf),
        acc_nb=accuracy_score(y_test, pred_nb),
        acc_mrf=accuracy_score(y_test, pred_mrf),
    )


# SIDEBAR
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <span class="sidebar-logo-icon">🌾</span>
        <div class="sidebar-logo-text">Crop Recommendation Analytics</div>
        <div class="sidebar-logo-sub">Probabilité Appliquée · IA 2</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["Accueil & Dataset",
         "Lois Statistiques",
         "Modeles ML",
         "Hybridation",
         "Metriques Avancees",
         "Comparaison Finale",
         "Prediction Live"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("<p style='font-size:0.8rem;color:#a7f3d0;margin-top:0.5rem;'>Encadrant : Pr. Mohammed KAICER</p>", unsafe_allow_html=True)


# LOAD DATA
with st.spinner("⚙️ Chargement et entraînement des modèles..."):
    D = load_data()
    M = train_all_models(D)

df       = M['df']
le       = M['le']
scaler   = M['scaler']
summary  = M['summary']


# HERO BANNER
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🌾 Crop Recommendation Analytics</div>
    <div class="hero-subtitle">
        Analyse complète · Modèles ML · Lois statistiques · Hybridation · Métriques avancées
    </div>
    <div class="hero-badges">
        <span class="badge">2 200 observations</span>
        <span class="badge">22 cultures</span>
        <span class="badge">7 modeles ML</span>
        <span class="badge">Weibull + Benford</span>
        <span class="badge">13 metriques</span>
    </div>
</div>
""", unsafe_allow_html=True)


# PAGE: ACCUEIL & DATASET
if page == "Accueil & Dataset":
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">a.</div>
        <h2 class="section-title">Exploration du Dataset</h2>
    </div>
    """, unsafe_allow_html=True)

    # KPI row
    st.markdown(f"""
    <div class="metrics-row">
        <div class="metric-card">
            <div class="metric-label">Observations</div>
            <div class="metric-value">{df.shape[0]:,}</div>
            <div class="metric-delta">lignes × {df.shape[1]} colonnes</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Cultures</div>
            <div class="metric-value">22</div>
            <div class="metric-delta good">espèces différentes</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Features</div>
            <div class="metric-value">7</div>
            <div class="metric-delta">indicateurs pédoclimatiques</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Valeurs nulles</div>
            <div class="metric-value">0</div>
            <div class="metric-delta good">dataset propre ✓</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("#### Apercu du dataset")
        st.dataframe(df.head(10), use_container_width=True, height=320)
    with col2:
        st.markdown("#### Statistiques descriptives")
        st.dataframe(df[FEATURES].describe().round(3), use_container_width=True, height=320)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Distribution des 22 cultures")
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor('white')
        counts = df['Culture'].value_counts().sort_values()
        colors = plt.cm.get_cmap('Greens', len(counts) + 4)
        bars = ax.barh(counts.index, counts.values,
                       color=[colors(i+4) for i in range(len(counts))],
                       edgecolor='white', linewidth=0.5)
        for bar, v in zip(bars, counts.values):
            ax.text(v + 0.5, bar.get_y() + bar.get_height()/2,
                    str(v), va='center', fontsize=8, color='#064e3b', fontweight='600')
        ax.set_xlabel('Observations', fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('white')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown("#### Matrice de correlation")
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor('white')
        corr = df[FEATURES].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
                    ax=ax, linewidths=.5, center=0,
                    annot_kws={'size': 9}, cbar_kws={'shrink': .8})
        ax.set_title('Corrélations entre features', fontweight='bold', fontsize=11, pad=10)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("#### Boxplots des features par culture")
    feat_choice = st.selectbox("Choisir une feature", FEATURES)
    fig, ax = plt.subplots(figsize=(14, 4))
    fig.patch.set_facecolor('white')
    cultures = sorted(df['Culture'].unique())
    data_bp = [df[df['Culture']==c][feat_choice].values for c in cultures]
    bp = ax.boxplot(data_bp, patch_artist=True, notch=False,
                    medianprops=dict(color='white', lw=2),
                    flierprops=dict(marker='.', ms=3, alpha=.4))
    cmap_bp = plt.cm.get_cmap('tab20', len(cultures))
    for patch, k in zip(bp['boxes'], range(len(cultures))):
        patch.set_facecolor(cmap_bp(k))
    ax.set_xticks(range(1, len(cultures)+1))
    ax.set_xticklabels(cultures, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(feat_choice); ax.set_facecolor('white')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Analyse PCA — Projection 2D")
        pca2_acc = PCA(n_components=2)
        X_2d_acc = pca2_acc.fit_transform(StandardScaler().fit_transform(df[FEATURES].values))
        var_ratio_acc = PCA().fit(StandardScaler().fit_transform(df[FEATURES].values)).explained_variance_ratio_
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor('white')
        cmap20_acc = plt.cm.get_cmap('tab20', len(cultures))
        for i, crop in enumerate(cultures):
            mask = df['Culture'].values == crop
            ax.scatter(X_2d_acc[mask, 0], X_2d_acc[mask, 1],
                       c=[cmap20_acc(i)], label=crop, s=15, alpha=0.7, edgecolors='none')
        ax.set_xlabel(f'PC1 ({var_ratio_acc[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({var_ratio_acc[1]*100:.1f}%)')
        ax.set_title('Projection PCA 2D — 22 cultures', fontweight='bold')
        ax.legend(fontsize=6, loc='upper right', ncol=2, framealpha=0.9)
        ax.set_facecolor('white')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown("#### Détection des outliers (méthode IQR)")
        outlier_data = []
        for feat in FEATURES:
            Q1 = df[feat].quantile(0.25); Q3 = df[feat].quantile(0.75)
            IQR = Q3 - Q1
            lb, ub = Q1 - 1.5*IQR, Q3 + 1.5*IQR
            n_out = ((df[feat] < lb) | (df[feat] > ub)).sum()
            outlier_data.append({
                'Feature': feat, 'N outliers': n_out,
                'Borne inf': round(lb, 2), 'Borne sup': round(ub, 2),
                'Stratégie': 'Conservation ✓'
            })
        df_out = pd.DataFrame(outlier_data)
        st.dataframe(df_out, use_container_width=True, hide_index=True)
        st.markdown("""
        <div class="info-box">
        <b>Stratégie :</b> Les outliers sont <b>conservés</b> car ils sont agronomiquement valides 
        (conditions climatiques ou pédologiques extrêmes réelles).
        </div>
        """, unsafe_allow_html=True)


# PAGE: LOIS STATISTIQUES
elif page == "Lois Statistiques":
    tab1, tab2, tab3 = st.tabs(["Loi de Benford", "Distribution de Weibull", "Weibull par Culture"])

    with tab1:
        st.markdown("""
        <div class="section-header">
            <div class="section-icon">b.</div>
            <h2 class="section-title">Loi de Benford</h2>
        </div>
        <div class="info-box">
            La <strong>loi de Benford</strong> stipule que dans un jeu de données naturel, 
            le chiffre <strong>1</strong> apparaît comme premier chiffre dans ~30% des cas.
        </div>
        <div class="formula-box">P(d) = log₁₀(1 + 1/d)  pour d ∈ {1, 2, ..., 9}</div>
        """, unsafe_allow_html=True)

        benford_expected = np.array([np.log10(1 + 1/d) for d in range(1, 10)])

        def get_first_digits(series):
            digits = []
            for v in series:
                s = str(abs(float(v))).replace('0.','').replace('.','').lstrip('0')
                if s and s[0].isdigit() and s[0] != '0':
                    digits.append(int(s[0]))
            return np.array(digits)

        benford_results = {}
        for feat in FEATURES:
            digits = get_first_digits(df[feat].values)
            obs_freq = np.array([(digits==d).sum() for d in range(1,10)])
            obs_prop = obs_freq / obs_freq.sum()
            expected_counts = benford_expected * obs_freq.sum()
            chi2_stat, chi2_p = stats.chisquare(obs_freq, f_exp=expected_counts)
            mad = np.mean(np.abs(obs_prop - benford_expected))
            conformity = 'Conforme' if mad < 0.006 else ('Acceptable' if mad < 0.012 else ('Marginal' if mad < 0.015 else 'Non conforme'))
            benford_results[feat] = {'obs_prop': obs_prop, 'chi2': round(chi2_stat,4), 'p': round(chi2_p,4), 'MAD': round(mad,5), 'Conformité': conformity}

        df_bsummary = pd.DataFrame({f: {'Chi²': benford_results[f]['chi2'], 'p-value': benford_results[f]['p'],
            'MAD': benford_results[f]['MAD'], 'Conformité': benford_results[f]['Conformité']} for f in benford_results}).T
        st.dataframe(df_bsummary, use_container_width=True)

        feat_b = st.selectbox("Feature à visualiser", FEATURES, key='benford_feat')
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor('white')
        digits_range = np.arange(1, 10)
        obs = benford_results[feat_b]['obs_prop']
        w = 0.38
        ax.bar(digits_range - w/2, obs, w, label='Observé', color='#064e3b', alpha=0.85, edgecolor='white')
        ax.bar(digits_range + w/2, benford_expected, w, label='Benford théorique', color='#f59e0b', alpha=0.85, edgecolor='white')
        ax.plot(digits_range, benford_expected, 'k--', lw=1.5, alpha=0.5)
        conf = benford_results[feat_b]['Conformité']
        col = '#059669' if conf == 'Conforme' else ('#f59e0b' if conf in ['Acceptable','Marginal'] else '#ef4444')
        ax.set_title(f"{feat_b} — MAD={benford_results[feat_b]['MAD']:.4f} — {conf}", fontweight='bold', color=col)
        ax.set_xlabel('Premier chiffre'); ax.set_ylabel('Fréquence')
        ax.set_xticks(digits_range); ax.legend(); ax.set_facecolor('white')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with tab2:
        st.markdown("""
        <div class="section-header">
            <div class="section-icon">c.</div>
            <h2 class="section-title">Distribution de Weibull</h2>
        </div>
        <div class="info-box">
            La distribution de Weibull modélise la <strong>fiabilité</strong> et la durée de vie.
            Elle est très flexible grâce à son paramètre de forme <strong>k</strong>.
        </div>
        <div class="formula-box">f(x; k, λ) = (k/λ)(x/λ)^(k-1) · exp(-(x/λ)^k)
k &lt; 1 : décroissance rapide  |  k = 1 : exponentielle  |  k &gt; 1 : queue droite</div>
        """, unsafe_allow_html=True)

        feat_w = st.selectbox("Feature à analyser", FEATURES, key='weibull_feat')
        data = df[feat_w].values
        data_pos = data - data.min() + 0.01
        c, loc, scale = weibull_min.fit(data_pos, floc=0)
        ks_stat, ks_p = kstest(data_pos, 'weibull_min', args=(c, loc, scale))

        col1, col2, col3 = st.columns(3)
        col1.metric("Paramètre de forme k", f"{c:.4f}")
        col2.metric("Paramètre d'échelle λ", f"{scale:.4f}")
        col3.metric("p-value KS", f"{ks_p:.4f}", "Bonne ✓" if ks_p > 0.05 else "Rejetée ✗")

        fig, ax = plt.subplots(figsize=(9, 4))
        fig.patch.set_facecolor('white')
        ax.hist(data_pos, bins=40, density=True, alpha=0.55, color='#064e3b', edgecolor='white', label='Données réelles')
        x = np.linspace(data_pos.min()*0.95, data_pos.max()*1.05, 300)
        ax.plot(x, weibull_min.pdf(x, c, loc=0, scale=scale), 'r-', lw=2.5, label=f'Weibull(k={c:.2f}, λ={scale:.1f})')
        ax.set_title(f'Ajustement Weibull — {feat_w}', fontweight='bold')
        ax.set_xlabel('Valeur'); ax.set_ylabel('Densité'); ax.legend()
        ax.set_facecolor('white'); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown("#### Paramètres Weibull — toutes les features")
        wp_data = {}
        for feat in FEATURES:
            dp = df[feat].values - df[feat].values.min() + 0.01
            ci, loci, sci = weibull_min.fit(dp, floc=0)
            ksi, kpi = kstest(dp, 'weibull_min', args=(ci, loci, sci))
            wp_data[feat] = {'k (forme)': round(ci,4), 'λ (échelle)': round(sci,4),
                             'KS stat': round(ksi,4), 'p-value': round(kpi,4),
                             'Ajustement': 'Bonne ✓' if kpi > 0.05 else 'Rejetée ✗'}
        st.dataframe(pd.DataFrame(wp_data).T, use_container_width=True)

    with tab3:
        st.markdown("""
        <div class="section-header">
            <div class="section-icon">📈</div>
            <h2 class="section-title">Weibull par Culture — Analyse de l'Azote_N</h2>
        </div>
        <div class="info-box">
            On ajuste une loi de Weibull pour <b>chaque culture</b> sur la feature <b>Azote_N</b> 
            afin de caractériser les besoins azotés spécifiques à chaque culture.
            Le paramètre λ (échelle) reflète le niveau moyen d'azote requis.
        </div>
        """, unsafe_allow_html=True)

        # Calcul Weibull par culture
        cultures_list = le.classes_
        weibull_per_culture = {}
        for culture in cultures_list:
            subset = df[df['Culture'] == culture]['Azote_N'].values
            if len(subset) > 5:
                try:
                    c_c, loc_c, scale_c = weibull_min.fit(subset, floc=0)
                    ks_c, ksp_c = kstest(subset, 'weibull_min', args=(c_c, loc_c, scale_c))
                    weibull_per_culture[culture] = {'k': round(c_c, 3), 'λ': round(scale_c, 2),
                                                     'KS p-value': round(ksp_c, 4)}
                except Exception:
                    weibull_per_culture[culture] = {'k': np.nan, 'λ': np.nan, 'KS p-value': np.nan}

        df_w_cult = pd.DataFrame(weibull_per_culture).T.dropna()
        df_w_cult_sorted = df_w_cult.sort_values('λ', ascending=False)

        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.markdown("#### Paramètres Weibull par culture (trié par λ)")
            st.dataframe(df_w_cult_sorted, use_container_width=True)

        with col2:
            st.markdown("#### λ (échelle) par culture — besoins azotés")
            fig, ax = plt.subplots(figsize=(7, 7))
            fig.patch.set_facecolor('white')
            cmap_cult = plt.cm.get_cmap('RdYlGn', len(df_w_cult_sorted))
            bars = ax.barh(df_w_cult_sorted.index, df_w_cult_sorted['λ'].values,
                           color=[cmap_cult(i) for i in range(len(df_w_cult_sorted))], edgecolor='white')
            for bar, v in zip(bars, df_w_cult_sorted['λ'].values):
                ax.text(v + 0.3, bar.get_y() + bar.get_height()/2, f'{v:.1f}',
                        va='center', fontsize=8, fontweight='600')
            ax.set_xlabel('λ (paramètre d\'échelle Weibull = niveau azote)')
            ax.set_title('Besoin azoté par culture\n(λ élevé = besoin plus élevé)', fontweight='bold')
            ax.set_facecolor('white')
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        # Top 8 cultures — visualisation PDF
        st.markdown("#### Distribution Weibull Azote_N — Top 8 cultures par λ")
        top8_cultures = df_w_cult_sorted.head(8).index.tolist()
        fig, axes = plt.subplots(2, 4, figsize=(18, 8))
        fig.patch.set_facecolor('white')
        axes = axes.flatten()
        for idx, culture in enumerate(top8_cultures):
            ax = axes[idx]
            data_c = df[df['Culture'] == culture]['Azote_N'].values
            c_c = weibull_per_culture[culture]['k']
            sc_c = weibull_per_culture[culture]['λ']
            ax.hist(data_c, bins=20, density=True, alpha=0.55, color='#059669', edgecolor='white')
            x_c = np.linspace(0, max(data_c)*1.1 if len(data_c) > 0 else 100, 200)
            ax.plot(x_c, weibull_min.pdf(x_c, c_c, loc=0, scale=sc_c), 'r-', lw=2)
            ax.set_title(f'{culture}\nk={c_c:.2f}, λ={sc_c:.1f}', fontsize=9, fontweight='bold')
            ax.set_xlabel('Azote_N'); ax.set_facecolor('white')
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.suptitle('Distribution Weibull Azote_N — Top 8 cultures par λ', fontsize=13, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()


# PAGE: MODÈLES ML
elif page == "Modeles ML":
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">d.</div>
        <h2 class="section-title">Modèles Machine Learning</h2>
    </div>
    """, unsafe_allow_html=True)

    model_choice = st.selectbox(
        "Sélectionner un modèle",
        ["Decision Tree", "Random Forest", "Naive Bayes", "KNN", "K-Means (Clustering)", "Markov-RF", "MFTL"]
    )

    model_info = {
        "Decision Tree":        {"icon":"DT","type":"Supervisé · Classification","color":"#064e3b"},
        "Random Forest":        {"icon":"RF","type":"Supervisé · Ensemble","color":"#059669"},
        "Naive Bayes":          {"icon":"NB","type":"Supervisé · Probabiliste","color":"#8b5cf6"},
        "KNN":                  {"icon":"KN","type":"Supervisé · Basé instances","color":"#f59e0b"},
        "K-Means (Clustering)": {"icon":"KM","type":"Non supervisé · Clustering","color":"#3b82f6"},
        "Markov-RF":            {"icon":"MK","type":"Hybride · Markov + Random Forest","color":"#f59e0b"},
        "MFTL":                 {"icon":"MF","type":"Méta-heuristique · Gradient-Free","color":"#ec4899"},
    }

    info = model_info[model_choice]

    if model_choice == "Markov-RF":
        # ── MARKOV-RF SOUS-PAGE ──────────────────────────────────────────────
        st.markdown("""
        <div class="info-box">
            Le <strong>Markov-RF</strong> exploite les dépendances séquentielles entre cultures :<br>
            <code>P_final(yₜ) = α · P_RF(yₜ | xₜ) + (1−α) · P_Markov(yₜ | yₜ₋₁)</code><br>
            avec α = 0.7, donc 70% basé sur RF et 30% basé sur les transitions Markov.
        </div>
        <div class="formula-box">P_final(yₜ) = 0.7 · P_RF(yₜ | xₜ) + 0.3 · P_Markov(yₜ | yₜ₋₁)</div>
        """, unsafe_allow_html=True)

        mrf_metrics = M['summary']['Markov-RF']
        st.markdown(f"""
        <div class="metrics-row">
            <div class="metric-card">
                <div class="metric-label">Accuracy</div>
                <div class="metric-value">{mrf_metrics['Accuracy']*100:.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">F1 / Dice (macro)</div>
                <div class="metric-value">{mrf_metrics['F1 / Dice']:.4f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">AUC (macro)</div>
                <div class="metric-value">{mrf_metrics['AUC']:.4f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">MCC</div>
                <div class="metric-value">{mrf_metrics['MCC']:.4f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Matrice de Transition Markov entre Cultures")
            n_classes = M['n_classes']
            y_full = M['y']
            tm_vis = np.zeros((n_classes, n_classes))
            for i in range(len(y_full)-1):
                tm_vis[y_full[i], y_full[i+1]] += 1
            tm_vis = tm_vis / (tm_vis.sum(axis=1, keepdims=True) + 1e-9)
            fig, ax = plt.subplots(figsize=(8, 7))
            fig.patch.set_facecolor('white')
            sns.heatmap(tm_vis, cmap='YlOrRd', ax=ax, linewidths=0.3,
                        xticklabels=le.classes_, yticklabels=le.classes_,
                        cbar_kws={'shrink': 0.8})
            ax.set_title('Matrice de Transition Markov\nentre Cultures', fontweight='bold', fontsize=11)
            ax.set_xlabel('Culture suivante (yₜ)'); ax.set_ylabel('Culture actuelle (yₜ₋₁)')
            ax.tick_params(labelsize=7)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col2:
            st.markdown("#### Comparaison RF Standard vs Markov-RF")
            rf_acc   = M['summary']['Random Forest']['Accuracy']
            mrf_acc  = mrf_metrics['Accuracy']
            gain_mrf = (mrf_acc - rf_acc) * 100

            fig, axes = plt.subplots(2, 1, figsize=(7, 7))
            fig.patch.set_facecolor('white')
            models_cmp = ['Random Forest', 'Markov-RF']
            accs_cmp   = [rf_acc*100, mrf_acc*100]
            colors_cmp = ['#064e3b', '#f59e0b']
            bars = axes[0].bar(models_cmp, accs_cmp, color=colors_cmp, edgecolor='white', width=0.5)
            for bar, v in zip(bars, accs_cmp):
                axes[0].text(bar.get_x()+bar.get_width()/2, v+0.1, f'{v:.2f}%',
                             ha='center', fontsize=12, fontweight='bold')
            axes[0].set_ylabel('Accuracy (%)')
            axes[0].set_title(f'RF vs Markov-RF — Gain : {gain_mrf:+.2f}%', fontweight='bold')
            axes[0].set_ylim(min(accs_cmp)*0.97, max(accs_cmp)*1.03)
            axes[0].set_facecolor('white')
            axes[0].spines['top'].set_visible(False); axes[0].spines['right'].set_visible(False)

            metrics_mrf = ['F1 / Dice', 'Recall', 'AUC', 'MCC', "Cohen's κ"]
            vals_rf  = [M['summary']['Random Forest'][m] for m in metrics_mrf]
            vals_mrf = [mrf_metrics[m] for m in metrics_mrf]
            x = np.arange(len(metrics_mrf)); w = 0.35
            axes[1].bar(x - w/2, vals_rf,  w, label='Random Forest', color='#064e3b', alpha=0.85, edgecolor='white')
            axes[1].bar(x + w/2, vals_mrf, w, label='Markov-RF',     color='#f59e0b', alpha=0.85, edgecolor='white')
            axes[1].set_xticks(x); axes[1].set_xticklabels(metrics_mrf, rotation=20, ha='right', fontsize=9)
            axes[1].set_ylim(0, 1.1); axes[1].legend(); axes[1].set_title('Métriques comparées', fontweight='bold')
            axes[1].set_facecolor('white')
            axes[1].spines['top'].set_visible(False); axes[1].spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

            st.markdown(f"""
            <div class="{'info-box' if gain_mrf >= 0 else 'warning-box'}">
            <b>Interprétation :</b> L'apport de la chaîne de Markov donne un gain de <b>{gain_mrf:+.2f}%</b>
            en accuracy par rapport au RF standard.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### Matrice de confusion — Markov-RF")
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor('white')
        cm_mrf = confusion_matrix(M['y_test'], M['pred_mrf'])
        sns.heatmap(cm_mrf, annot=False, cmap='Oranges', ax=ax,
                    xticklabels=le.classes_, yticklabels=le.classes_,
                    linewidths=.3, cbar=True)
        ax.set_xlabel('Prédit'); ax.set_ylabel('Réel')
        ax.tick_params(labelsize=7)
        ax.set_title(f'Matrice de confusion — Markov-RF  (Accuracy={mrf_metrics["Accuracy"]*100:.2f}%)', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    elif model_choice == "MFTL":
        # ── MFTL SOUS-PAGE ───────────────────────────────────────────────────
        st.markdown("""
        <div class="info-box">
            <strong>MFTL (Meta-heuristique Gradient-Free)</strong> : optimisation sans gradient via
            une population de particules explorant l'espace des poids d'un modèle linéaire.<br>
            Chaque particule encode un vecteur de poids w ∈ ℝᵈ, évalué par accuracy sur validation.
        </div>
        <div class="formula-box">fitness(w) = accuracy(sign(Xw), y_val) · (1 − λ·‖w‖₁)</div>
        """, unsafe_allow_html=True)

        from sklearn.linear_model import SGDClassifier
        from sklearn.ensemble import BaggingClassifier

        col1, col2 = st.columns([1, 2])
        with col1:
            n_particles = st.slider("Nombre de particules", 10, 100, 30, 5)
            n_iter      = st.slider("Itérations", 10, 100, 30, 5)
            run_mftl    = st.button("Lancer MFTL", use_container_width=True)

        with col2:
            if run_mftl:
                with st.spinner("Optimisation MFTL en cours..."):
                    np.random.seed(42)
                    n_feat = M['X_train'].shape[1]
                    n_cls  = M['n_classes']
                    particles = np.random.randn(n_particles, n_feat * n_cls) * 0.1
                    best_fitness, best_particle = -np.inf, particles[0].copy()

                    X_tr, X_val, y_tr_mf, y_val_mf = train_test_split(
                        M['X_train'], M['y_train'], test_size=0.2, random_state=42)

                    for _ in range(n_iter):
                        for i, p in enumerate(particles):
                            W = p.reshape(n_cls, n_feat)
                            logits = X_tr @ W.T
                            preds  = np.argmax(logits, axis=1)
                            f = accuracy_score(y_tr_mf, preds) * (1 - 0.001 * np.linalg.norm(p, 1))
                            if f > best_fitness:
                                best_fitness = f
                                best_particle = p.copy()
                        particles += np.random.randn(*particles.shape) * 0.05 * (1 - _/n_iter)
                        particles[0] = best_particle

                    W_best   = best_particle.reshape(n_cls, n_feat)
                    logits_t = M['X_test'] @ W_best.T
                    pred_mftl = np.argmax(logits_t, axis=1)
                    prob_mftl = np.exp(logits_t) / np.exp(logits_t).sum(axis=1, keepdims=True)
                    prob_mftl = np.clip(prob_mftl, 1e-9, 1)

                    acc_mftl = accuracy_score(M['y_test'], pred_mftl)
                    f1_mftl  = f1_score(M['y_test'], pred_mftl, average='macro', zero_division=0)
                    auc_mftl = roc_auc_score(M['y_bin'], prob_mftl, multi_class='ovr', average='macro')
                    mcc_mftl = matthews_corrcoef(M['y_test'], pred_mftl)
                    kap_mftl = cohen_kappa_score(M['y_test'], pred_mftl)
                    rec_mftl = recall_score(M['y_test'], pred_mftl, average='macro', zero_division=0)
                    ll_mftl  = log_loss(M['y_test'], prob_mftl)

                    st.markdown(f"""
                    <div class="metrics-row">
                        <div class="metric-card">
                            <div class="metric-label">Accuracy MFTL</div>
                            <div class="metric-value">{acc_mftl*100:.2f}%</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">F1 macro</div>
                            <div class="metric-value">{f1_mftl:.4f}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">AUC macro</div>
                            <div class="metric-value">{auc_mftl:.4f}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">MCC</div>
                            <div class="metric-value">{mcc_mftl:.4f}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Comparaison MFTL vs autres optimiseurs
                    st.markdown("#### MFTL vs Optimiseurs classiques")
                    X_tr2, X_val2, y_tr2, y_val2 = train_test_split(
                        M['X_train'], M['y_train'], test_size=0.2, random_state=42)

                    from sklearn.neural_network import MLPClassifier
                    sgd = SGDClassifier(loss='log_loss', max_iter=200, random_state=42)
                    sgd.fit(M['X_train'], M['y_train'])
                    acc_gd = accuracy_score(M['y_test'], sgd.predict(M['X_test']))

                    mlp = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=200, random_state=42)
                    mlp.fit(M['X_train'], M['y_train'])
                    acc_adam = accuracy_score(M['y_test'], mlp.predict(M['X_test']))

                    bag = BaggingClassifier(
                        estimator=SGDClassifier(loss='log_loss', max_iter=200, random_state=42),
                        n_estimators=10, random_state=42, n_jobs=-1)
                    bag.fit(M['X_train'], M['y_train'])
                    acc_bag = accuracy_score(M['y_test'], bag.predict(M['X_test']))

                    methods   = ['Gradient\nDescent', 'Ensemble\nGrad. Desc.', 'ADAM\n(MLP)', 'MFTL\nGradient-Free']
                    accs_comp = [acc_gd, acc_bag, acc_adam, acc_mftl]
                    colors_m  = ['#ef4444', '#f59e0b', '#8b5cf6', '#059669']

                    fig, ax = plt.subplots(figsize=(7, 5))
                    fig.patch.set_facecolor('white')
                    bars = ax.bar(methods, [a*100 for a in accs_comp], color=colors_m, edgecolor='white', width=0.5)
                    for bar, v in zip(bars, accs_comp):
                        ax.text(bar.get_x()+bar.get_width()/2, v*100+0.3,
                                f'{v*100:.1f}%', ha='center', fontsize=11, fontweight='bold')
                    ax.set_ylabel('Accuracy (%)')
                    ax.set_title('MFTL vs Optimiseurs classiques', fontweight='bold')
                    ax.set_ylim(0, 115)
                    ax.set_facecolor('white')
                    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

                st.markdown(f"""
                <div class="info-box">
                <b>Résultats MFTL ({n_particles} particules, {n_iter} itérations) :</b><br>
                Accuracy = <b>{acc_mftl*100:.2f}%</b> · F1 = <b>{f1_mftl:.4f}</b> ·
                AUC = <b>{auc_mftl:.4f}</b> · MCC = <b>{mcc_mftl:.4f}</b> ·
                Cohen κ = <b>{kap_mftl:.4f}</b> · Recall = <b>{rec_mftl:.4f}</b> ·
                Log Loss = <b>{ll_mftl:.4f}</b>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="text-align:center;padding:3rem;color:#94a3b8;">
                    <div style="font-size:2rem;margin-bottom:1rem;">🧠</div>
                    <div style="font-size:1.1rem;font-weight:500;">
                        Configurez les paramètres et cliquez sur <b>Lancer MFTL</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    elif model_choice != "K-Means (Clustering)":
        pred_map = {
            "Decision Tree": (M['pred_dt'], M['prob_dt']),
            "Random Forest": (M['pred_rf'], M['prob_rf']),
            "Naive Bayes":   (M['pred_nb'], M['prob_nb']),
            "KNN":           (M['pred_knn'], M['prob_knn']),
        }
        pred, prob = pred_map[model_choice]
        acc  = accuracy_score(M['y_test'], pred)
        f1   = f1_score(M['y_test'], pred, average='macro')
        mcc  = matthews_corrcoef(M['y_test'], pred)
        rec  = recall_score(M['y_test'], pred, average='macro')
        auc  = roc_auc_score(M['y_bin'], prob, multi_class='ovr', average='macro')

        st.markdown(f"""
        <div class="metrics-row">
            <div class="metric-card">
                <div class="metric-label">Accuracy</div>
                <div class="metric-value">{acc*100:.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">F1 / Dice (macro)</div>
                <div class="metric-value">{f1:.4f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">MCC</div>
                <div class="metric-value">{mcc:.4f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">AUC (macro)</div>
                <div class="metric-value">{auc:.4f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Matrice de confusion")
            fig, ax = plt.subplots(figsize=(7, 6))
            fig.patch.set_facecolor('white')
            cm = confusion_matrix(M['y_test'], pred)
            sns.heatmap(cm, annot=False, cmap='Greens', ax=ax,
                        xticklabels=le.classes_, yticklabels=le.classes_,
                        linewidths=.3, cbar=True)
            ax.set_xlabel('Prédit'); ax.set_ylabel('Réel')
            ax.tick_params(labelsize=7)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col2:
            st.markdown("#### Courbe ROC (macro OvR)")
            fig, ax = plt.subplots(figsize=(7, 6))
            fig.patch.set_facecolor('white')
            mean_fpr = np.linspace(0, 1, 200)
            tprs = []
            for k in range(M['n_classes']):
                fpr_k, tpr_k, _ = roc_curve(M['y_bin'][:,k], prob[:,k])
                tprs.append(np.interp(mean_fpr, fpr_k, tpr_k))
            mean_tpr = np.mean(tprs, axis=0)
            ax.fill_between(mean_fpr, mean_tpr, alpha=.15, color='#059669')
            ax.plot(mean_fpr, mean_tpr, lw=2.5, color='#064e3b', label=f'AUC = {auc:.4f}')
            ax.plot([0,1],[0,1],'--', color='gray', lw=1.5, label='Aléatoire')
            ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
            ax.set_title(f'ROC — {model_choice}', fontweight='bold')
            ax.legend(); ax.set_facecolor('white')
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        st.markdown("#### Rapport de classification")
        report = classification_report(M['y_test'], pred, target_names=le.classes_, output_dict=True)
        df_report = pd.DataFrame(report).T.round(3)
        st.dataframe(df_report, use_container_width=True)

    else:  # K-Means
        st.markdown(f"""
        <div class="metrics-row">
            <div class="metric-card">
                <div class="metric-label">Silhouette Score</div>
                <div class="metric-value">{M['sil_km']:.4f}</div>
                <div class="metric-delta">↑ meilleur</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Davies-Bouldin</div>
                <div class="metric-value">{M['dbi_km']:.4f}</div>
                <div class="metric-delta">↓ meilleur</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">K clusters</div>
                <div class="metric-value">22</div>
                <div class="metric-delta">= nb de cultures</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">ARI</div>
                <div class="metric-value">{adjusted_rand_score(M['y'], M['km'].labels_):.4f}</div>
                <div class="metric-delta">Adjusted Rand Index</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        pca2 = PCA(n_components=2)
        X_2d = pca2.fit_transform(M['X_std'])
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor('white')
        cmap22 = plt.cm.get_cmap('tab20', 22)
        for k in range(22):
            mask = M['km'].labels_ == k
            axes[0].scatter(X_2d[mask,0], X_2d[mask,1], c=[cmap22(k)], s=15, alpha=.6, edgecolors='none')
        axes[0].set_title(f'K-Means K=22 — PCA\nSilhouette={M["sil_km"]:.3f}', fontweight='bold')
        axes[0].set_xlabel('PC1'); axes[0].set_ylabel('PC2'); axes[0].set_facecolor('white')
        for i, crop in enumerate(sorted(df['Culture'].unique())):
            mask = M['y_str'] == crop
            axes[1].scatter(X_2d[mask,0], X_2d[mask,1], c=[cmap22(i)], label=crop, s=15, alpha=.6, edgecolors='none')
        axes[1].set_title('Vraies cultures — PCA', fontweight='bold')
        axes[1].set_xlabel('PC1'); axes[1].set_ylabel('PC2')
        axes[1].legend(fontsize=6, ncol=2); axes[1].set_facecolor('white')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()



# PAGE: HYBRIDATION
elif page == "Hybridation":
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">e.</div>
        <h2 class="section-title">Hybridation — Lois + Modèles ML</h2>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Weibull + Random Forest", "Benford + Naive Bayes", "Hybridation Étendue"])

    with tab1:
        st.markdown("""
        <div class="info-box">
            <strong>Principe :</strong> Pour chaque observation, on calcule sa log-vraisemblance sous la loi de Weibull 
            ajustée sur les données d'entraînement. Ces 7 scores de typicité sont ajoutés comme features supplémentaires 
            au Random Forest → <strong>14 features</strong> au total.
        </div>
        <div class="formula-box">X_hybride = [X_original (7) | ln f_Weibull(Xⱼ) (7)]</div>
        """, unsafe_allow_html=True)

        rf_std  = summary['Random Forest']
        rf_weib = summary['RF + Weibull']
        gain    = (rf_weib['Accuracy'] - rf_std['Accuracy']) * 100

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy RF Standard",    f"{rf_std['Accuracy']*100:.2f}%")
        col2.metric("Accuracy RF + Weibull",   f"{rf_weib['Accuracy']*100:.2f}%", f"{gain:+.2f}%")
        col3.metric("F1 / Dice RF + Weibull",  f"{rf_weib['F1 / Dice']:.4f}")
        col4.metric("AUC RF + Weibull",        f"{rf_weib['AUC']:.4f}")

        metrics_comp = ['Accuracy', 'F1 / Dice', 'AUC', 'MCC', 'Recall']
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor('white')
        x = np.arange(len(metrics_comp))
        w = 0.35
        v_std  = [rf_std[m] for m in metrics_comp]
        v_weib = [rf_weib[m] for m in metrics_comp]
        b1 = ax.bar(x - w/2, v_std,  w, label='RF Standard',   color='#064e3b', alpha=0.85, edgecolor='white')
        b2 = ax.bar(x + w/2, v_weib, w, label='RF + Weibull',  color='#f59e0b', alpha=0.85, edgecolor='white')
        for bar, v in zip(b1, v_std):
            ax.text(bar.get_x()+bar.get_width()/2, v+.003, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
        for bar, v in zip(b2, v_weib):
            ax.text(bar.get_x()+bar.get_width()/2, v+.003, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold', color='#b45309')
        ax.set_xticks(x); ax.set_xticklabels(metrics_comp)
        ax.set_ylabel('Score'); ax.set_ylim(0, 1.12)
        ax.set_title('Impact de l\'hybridation Weibull sur Random Forest', fontweight='bold')
        ax.legend(); ax.set_facecolor('white')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with tab2:
        st.markdown("""
        <div class="info-box">
            <strong>Principe :</strong> Pour chaque feature, on calcule le score de conformité à Benford 
            (log-probabilité du premier chiffre significatif). Ces 7 scores enrichissent le Naive Bayes 
            → <strong>14 features</strong> au total.
        </div>
        <div class="formula-box">Benford(x) = log₁₀(1 + 1/d_premier(x))
X_hybride = [X_original (7) | Benford(Xⱼ) (7)]</div>
        """, unsafe_allow_html=True)

        nb_std  = summary['Naive Bayes']
        nb_bfd  = summary['NB + Benford']
        gain_nb = (nb_bfd['Accuracy'] - nb_std['Accuracy']) * 100

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy NB Standard",   f"{nb_std['Accuracy']*100:.2f}%")
        col2.metric("Accuracy NB + Benford",  f"{nb_bfd['Accuracy']*100:.2f}%", f"{gain_nb:+.2f}%")
        col3.metric("F1 / Dice NB + Benford", f"{nb_bfd['F1 / Dice']:.4f}")
        col4.metric("AUC NB + Benford",       f"{nb_bfd['AUC']:.4f}")

        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor('white')
        x = np.arange(len(metrics_comp))
        v_std  = [nb_std[m] for m in metrics_comp]
        v_bfd  = [nb_bfd[m] for m in metrics_comp]
        b1 = ax.bar(x - w/2, v_std, w, label='NB Standard',  color='#8b5cf6', alpha=0.85, edgecolor='white')
        b2 = ax.bar(x + w/2, v_bfd, w, label='NB + Benford', color='#f59e0b', alpha=0.85, edgecolor='white')
        for bar, v in zip(b1, v_std):
            ax.text(bar.get_x()+bar.get_width()/2, v+.003, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
        for bar, v in zip(b2, v_bfd):
            ax.text(bar.get_x()+bar.get_width()/2, v+.003, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold', color='#b45309')
        ax.set_xticks(x); ax.set_xticklabels(metrics_comp)
        ax.set_ylabel('Score'); ax.set_ylim(0, 1.12)
        ax.set_title('Impact de l\'hybridation Benford sur Naive Bayes', fontweight='bold')
        ax.legend(); ax.set_facecolor('white')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()



    with tab3:
        st.markdown("""
        <div class="info-box">
            Extension des hybridations à <strong>tous les modèles</strong> (DT, RF, KNN, NB) 
            avec les deux lois statistiques (Weibull & Benford) pour comparer l'impact sur chaque modèle.
        </div>
        """, unsafe_allow_html=True)

        @st.cache_resource
        def train_extended_hybrids(_M):
            X_train  = _M['X_train'];      X_test   = _M['X_test']
            X_train_raw = _M['X_train_raw']; X_test_raw = _M['X_test_raw']
            y_test   = _M['y_test']
            y_bin    = _M['y_bin']
            n_classes= _M['n_classes']
            weibull_params = _M['weibull_params']
            y_tr = _M['y_train']
    
            # ── Weibull features ──
            def weibull_feats(Xr):
                W = np.zeros((len(Xr), len(FEATURES)))
                for j, feat in enumerate(FEATURES):
                    c, loc, scale = weibull_params[feat]
                    vals = np.abs(Xr[:, j]) + 0.01
                    W[:, j] = np.clip(weibull_min.logpdf(vals, c, loc=loc, scale=scale), -100, 0)
                return W
    
            W_train = weibull_feats(X_train_raw)
            W_test  = weibull_feats(X_test_raw)
    
            sc_w = StandardScaler()
            Xw_tr = sc_w.fit_transform(np.hstack([X_train, W_train]))
            Xw_te = sc_w.transform(np.hstack([X_test, W_test]))
    
            # ── Benford features ──
            def benford_score(v):
                s = str(abs(float(v))).replace('0.','').replace('.','').lstrip('0')
                if not s or not s[0].isdigit() or s[0]=='0': return np.log10(1+1/5)
                return np.log10(1+1/int(s[0]))
            def benford_feats(Xr):
                B = np.zeros((len(Xr), len(FEATURES)))
                for i in range(len(Xr)):
                    for j in range(len(FEATURES)):
                        B[i,j] = benford_score(Xr[i,j])
                return B
    
            B_train = benford_feats(X_train_raw)
            B_test  = benford_feats(X_test_raw)
    
            sc_b = StandardScaler()
            Xb_tr = sc_b.fit_transform(np.hstack([X_train, B_train]))
            Xb_te = sc_b.transform(np.hstack([X_test, B_test]))
    
            def calc_metrics(yt, yp, proba):
                auc = roc_auc_score(y_bin, proba, multi_class='ovr', average='macro')
                fpr = np.mean([roc_curve(y_bin[:,k], proba[:,k])[0].mean() for k in range(n_classes)])
                return dict(
                    Accuracy=accuracy_score(yt, yp),
                    F1=f1_score(yt, yp, average='macro'),
                    AUC=auc, MCC=matthews_corrcoef(yt, yp),
                    Kappa=cohen_kappa_score(yt, yp),
                    Recall=recall_score(yt, yp, average='macro'),
                    LogLoss=log_loss(yt, proba), FPR=fpr
                )
    
            results = {}
    
            # ── Weibull hybrids ──
            for name, model in [('DT', DecisionTreeClassifier(max_depth=8, min_samples_split=5, random_state=42)),
                                 ('RF', RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)),
                                 ('KNN', KNeighborsClassifier(n_neighbors=7, n_jobs=-1)),
                                 ('NB', GaussianNB())]:
                model.fit(Xw_tr, y_tr)
                yp = model.predict(Xw_te); prob = model.predict_proba(Xw_te)
                results[f'{name} + Weibull'] = calc_metrics(y_test, yp, prob)
    
            # ── Benford hybrids ──
            for name, model in [('DT', DecisionTreeClassifier(max_depth=8, min_samples_split=5, random_state=42)),
                                 ('RF', RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)),
                                 ('KNN', KNeighborsClassifier(n_neighbors=7, n_jobs=-1)),
                                 ('NB', GaussianNB())]:
                model.fit(Xb_tr, y_tr)
                yp = model.predict(Xb_te); prob = model.predict_proba(Xb_te)
                results[f'{name} + Benford'] = calc_metrics(y_test, yp, prob)
    
            # Base models for gain
            base = {
                'DT': _M['acc_dt'],
                'RF': _M['acc_rf'],
                'KNN': _M['acc_knn'],
                'NB': _M['acc_nb'],
            }
            for key in results:
                base_name = key.split(' + ')[0]
                results[key]['Gain_vs_base'] = results[key]['Accuracy'] - base.get(base_name, 0)
    
            return results, base
    
        with st.spinner("Calcul des hybridations étendues..."):
            ext_results, base_accs = train_extended_hybrids(M)

        df_ext = pd.DataFrame(ext_results).T.round(4)

        tab1, tab2, tab3 = st.tabs(["Tableau comparatif", "Weibull — tous modèles", "Benford — tous modèles"])

        with tab1:
            st.markdown("#### Tableau ultime — toutes hybridations")
            lower_better_ext = ['LogLoss', 'FPR']
            def highlight_ext(col):
                is_lower = col.name in lower_better_ext
                best = col.min() if is_lower else col.max()
                worst = col.max() if is_lower else col.min()
                styles = []
                for v in col:
                    if v == best: styles.append('background-color:#dcfce7;color:#064e3b;font-weight:700')
                    elif v == worst: styles.append('background-color:#fee2e2;color:#991b1b')
                    else: styles.append('')
                return styles
            cols_show = ['Accuracy','F1','AUC','MCC','Kappa','Recall','LogLoss','FPR']
            styled_ext = df_ext[cols_show].style.apply(highlight_ext, axis=0).format("{:.4f}")
            st.dataframe(styled_ext, use_container_width=True, height=350)

            # Heatmap normalisée
            st.markdown("#### Heatmap normalisée — vue globale")
            df_norm_ext = df_ext[cols_show].copy().astype(float)
            for col in df_norm_ext.columns:
                mn, mx = df_norm_ext[col].min(), df_norm_ext[col].max()
                df_norm_ext[col] = (df_norm_ext[col] - mn) / (mx - mn + 1e-9)
                if col in lower_better_ext:
                    df_norm_ext[col] = 1 - df_norm_ext[col]
            fig, ax = plt.subplots(figsize=(14, 7))
            fig.patch.set_facecolor('white')
            sns.heatmap(df_norm_ext, annot=df_ext[cols_show].values, fmt='.3f',
                        cmap='RdYlGn', ax=ax, linewidths=.5, annot_kws={'size': 8},
                        cbar_kws={'label': 'Score normalisé (vert=meilleur)'})
            mid = len([k for k in ext_results if 'Weibull' in k])
            ax.axhline(mid, color='white', lw=3)
            ax.set_title('Toutes hybridations — Weibull (haut) / Benford (bas)', fontweight='bold', fontsize=12, pad=12)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with tab2:
            st.markdown("#### Impact Weibull par modèle")
            weibull_keys = [k for k in ext_results if 'Weibull' in k]
            models_w_names = [k.split(' + ')[0] for k in weibull_keys]
            accs_base_w = [base_accs[m] for m in models_w_names]
            accs_hyb_w  = [ext_results[k]['Accuracy'] for k in weibull_keys]
            gains_w     = [(h - b)*100 for h, b in zip(accs_hyb_w, accs_base_w)]

            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            fig.patch.set_facecolor('white')
            x = np.arange(len(models_w_names)); w = 0.35
            axes[0].bar(x-w/2, [a*100 for a in accs_base_w], w, label='Standard', color='#94a3b8', edgecolor='white')
            b2 = axes[0].bar(x+w/2, [a*100 for a in accs_hyb_w], w, label='+ Weibull', color='#059669', edgecolor='white')
            for bar, v in zip(b2, accs_hyb_w):
                axes[0].text(bar.get_x()+bar.get_width()/2, v*100+0.1, f'{v*100:.1f}%', ha='center', fontsize=9, fontweight='bold')
            axes[0].set_xticks(x); axes[0].set_xticklabels(models_w_names)
            axes[0].set_ylabel('Accuracy (%)'); axes[0].set_title('Standard vs + Weibull', fontweight='bold')
            axes[0].legend(); axes[0].set_facecolor('white')
            axes[0].spines['top'].set_visible(False); axes[0].spines['right'].set_visible(False)

            cols_gain = ['#059669' if g >= 0 else '#ef4444' for g in gains_w]
            brs = axes[1].bar(models_w_names, gains_w, color=cols_gain, edgecolor='white', width=0.5)
            for bar, v in zip(brs, gains_w):
                axes[1].text(bar.get_x()+bar.get_width()/2, v+(0.03 if v >= 0 else -0.12),
                             f'{v:+.2f}%', ha='center', fontweight='bold', fontsize=11)
            axes[1].axhline(0, color='black', lw=1)
            axes[1].set_title('Gain apporté par Weibull', fontweight='bold')
            axes[1].set_facecolor('white')
            axes[1].spines['top'].set_visible(False); axes[1].spines['right'].set_visible(False)

            f1s_w  = [ext_results[k]['F1'] for k in weibull_keys]
            aucs_w = [ext_results[k]['AUC'] for k in weibull_keys]
            axes[2].plot(models_w_names, f1s_w,  'o-', color='#0f3460', lw=2.5, ms=8, label='F1 macro')
            axes[2].plot(models_w_names, aucs_w, 's-', color='#f59e0b', lw=2.5, ms=8, label='AUC macro')
            for i, (f, a) in enumerate(zip(f1s_w, aucs_w)):
                axes[2].text(i, f+0.002, f'{f:.3f}', ha='center', fontsize=9)
                axes[2].text(i, a-0.009, f'{a:.3f}', ha='center', fontsize=9, color='#f59e0b')
            axes[2].set_ylim(0.75, 1.05); axes[2].legend()
            axes[2].set_title('F1 & AUC Hybrides Weibull', fontweight='bold')
            axes[2].set_facecolor('white')
            axes[2].spines['top'].set_visible(False); axes[2].spines['right'].set_visible(False)

            plt.suptitle('Hybridation Weibull — Tous les modèles', fontweight='bold', fontsize=13)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with tab3:
            st.markdown("#### Impact Benford par modèle")
            benford_keys = [k for k in ext_results if 'Benford' in k]
            models_b_names = [k.split(' + ')[0] for k in benford_keys]
            accs_base_b = [base_accs[m] for m in models_b_names]
            accs_hyb_b  = [ext_results[k]['Accuracy'] for k in benford_keys]
            gains_b     = [(h - b)*100 for h, b in zip(accs_hyb_b, accs_base_b)]

            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            fig.patch.set_facecolor('white')
            x = np.arange(len(models_b_names)); w = 0.35
            axes[0].bar(x-w/2, [a*100 for a in accs_base_b], w, label='Standard', color='#94a3b8', edgecolor='white')
            b2 = axes[0].bar(x+w/2, [a*100 for a in accs_hyb_b], w, label='+ Benford', color='#f59e0b', edgecolor='white')
            for bar, v in zip(b2, accs_hyb_b):
                axes[0].text(bar.get_x()+bar.get_width()/2, v*100+0.1, f'{v*100:.1f}%', ha='center', fontsize=9, fontweight='bold')
            axes[0].set_xticks(x); axes[0].set_xticklabels(models_b_names)
            axes[0].set_ylabel('Accuracy (%)'); axes[0].set_title('Standard vs + Benford', fontweight='bold')
            axes[0].legend(); axes[0].set_facecolor('white')
            axes[0].spines['top'].set_visible(False); axes[0].spines['right'].set_visible(False)

            cols_gain_b = ['#059669' if g >= 0 else '#ef4444' for g in gains_b]
            brs = axes[1].bar(models_b_names, gains_b, color=cols_gain_b, edgecolor='white', width=0.5)
            for bar, v in zip(brs, gains_b):
                axes[1].text(bar.get_x()+bar.get_width()/2, v+(0.03 if v >= 0 else -0.12),
                             f'{v:+.2f}%', ha='center', fontweight='bold', fontsize=11)
            axes[1].axhline(0, color='black', lw=1)
            axes[1].set_title('Gain apporté par Benford', fontweight='bold')
            axes[1].set_facecolor('white')
            axes[1].spines['top'].set_visible(False); axes[1].spines['right'].set_visible(False)

            f1s_b  = [ext_results[k]['F1'] for k in benford_keys]
            aucs_b = [ext_results[k]['AUC'] for k in benford_keys]
            axes[2].plot(models_b_names, f1s_b,  'o-', color='#f59e0b', lw=2.5, ms=8, label='F1 macro')
            axes[2].plot(models_b_names, aucs_b, 's-', color='#8b5cf6', lw=2.5, ms=8, label='AUC macro')
            for i, (f, a) in enumerate(zip(f1s_b, aucs_b)):
                axes[2].text(i, f+0.002, f'{f:.3f}', ha='center', fontsize=9)
                axes[2].text(i, a-0.010, f'{a:.3f}', ha='center', fontsize=9, color='#8b5cf6')
            axes[2].set_ylim(0.60, 1.05); axes[2].legend()
            axes[2].set_title('F1 & AUC Hybrides Benford', fontweight='bold')
            axes[2].set_facecolor('white')
            axes[2].spines['top'].set_visible(False); axes[2].spines['right'].set_visible(False)

            plt.suptitle('Hybridation Benford — Tous les modèles', fontweight='bold', fontsize=13)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()


# PAGE: MÉTRIQUES AVANCÉES
elif page == "Metriques Avancees":
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">f.</div>
        <h2 class="section-title">Métriques Avancées d'Évaluation</h2>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["a. Classification", "b. Clustering & Regression", "c. Metriques Theoriques"])

    with tab1:
        metrics_clf = ['Accuracy','F1 / Dice','Recall','AUC','MCC',"Cohen's κ",'Log Loss','FPR']
        models_list = list(summary.keys())

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("#### Tableau comparatif — métriques de classification")
            df_m = pd.DataFrame({m: {n: summary[n][m] for n in models_list} for m in metrics_clf}).T.round(4)
            st.dataframe(df_m, use_container_width=True)

        with col2:
            st.markdown("#### Légende des métriques")
            for metric, desc in {
                'Accuracy': 'Taux de bonnes prédictions',
                'F1 / Dice': '2×P×R/(P+R) — équilibre précision/rappel',
                'Recall': 'Taux de vrais positifs (sensibilité)',
                'AUC': 'Aire sous la courbe ROC',
                'MCC': 'Corrélation Phi — robuste aux déséquilibres',
                "Cohen's κ": 'Accord corrigé du hasard',
                'Log Loss': 'Pénalise la confiance incorrecte',
                'FPR': 'Taux de fausses alarmes (↓ meilleur)',
            }.items():
                st.markdown(f"<div style='font-size:0.82rem;margin:0.2rem 0;'><b>{metric}</b>: {desc}</div>", unsafe_allow_html=True)

        metric_viz = st.selectbox("Visualiser une métrique", metrics_clf)
        fig, ax = plt.subplots(figsize=(12, 4))
        fig.patch.set_facecolor('white')
        vals   = [summary[n][metric_viz] for n in models_list]
        colors = ['#ef4444' if metric_viz in ['Log Loss','FPR'] and v == max(vals)
                  else '#059669' if v == max(vals) else '#94a3b8' for v in vals]
        bars = ax.bar(models_list, vals, color=colors, edgecolor='white', width=0.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v + max(vals)*0.01,
                    f'{v:.4f}', ha='center', fontsize=10, fontweight='bold')
        arrow = '↓ meilleur' if metric_viz in ['Log Loss','FPR'] else '↑ meilleur'
        ax.set_title(f'{metric_viz} — Comparaison des modèles ({arrow})', fontweight='bold')
        ax.set_ylabel(metric_viz)
        plt.setp(ax.get_xticklabels(), rotation=25, ha='right')
        ax.set_facecolor('white'); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Metriques de clustering")
            st.markdown(f"""
            <div class="metric-card" style="margin-bottom:0.8rem;">
                <div class="metric-label">Silhouette Score (K-Means)</div>
                <div class="metric-value">{M['sil_km']:.4f}</div>
                <div class="metric-delta">↑ meilleur · ∈ [-1, 1]</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Davies-Bouldin Index (K-Means)</div>
                <div class="metric-value">{M['dbi_km']:.4f}</div>
                <div class="metric-delta">↓ meilleur · ≥ 0</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="info-box">
            <b>Silhouette</b> : mesure la cohésion intra-cluster et la séparation inter-cluster.<br>
            <b>Davies-Bouldin</b> : ratio de la dispersion intra-cluster sur la séparation inter-cluster.
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("#### RMSLE - Root Mean Squared Log Error")
            rmsle_vals = {n: summary[n]['RMSLE'] for n in models_list}
            st.markdown("""<div class="formula-box">RMSLE = √(1/n · Σ(log(ŷ+1) − log(y+1))²)</div>""", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('white')
            bars = ax.bar(rmsle_vals.keys(), rmsle_vals.values(),
                          color=['#059669' if v==min(rmsle_vals.values()) else '#94a3b8' for v in rmsle_vals.values()],
                          edgecolor='white')
            for bar, v in zip(bars, rmsle_vals.values()):
                ax.text(bar.get_x()+bar.get_width()/2, v+0.01, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
            ax.set_title('RMSLE — ↓ meilleur', fontweight='bold')
            plt.setp(ax.get_xticklabels(), rotation=25, ha='right')
            ax.set_facecolor('white'); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

    with tab3:
        st.markdown("#### Metriques avancees : ASD, SSIM, DCG/NDCG")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="model-card">
                <div class="model-card-title">ASD</div>
                <div class="model-card-type">Average Symmetric Surface Distance</div>
                <div class="info-box" style="margin:0">
                    <b>Définition originale</b> (segmentation médicale) : distance moyenne entre les surfaces de deux volumes segmentés A et B.<br><br>
                    <b>Adaptation classification</b> : distance moyenne entre centroïdes des classes prédites et réelles dans l'espace PCA.<br><br>
                    <b>↓ meilleur</b> (centroïdes proches = bonne localisation)
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="model-card">
                <div class="model-card-title">SSIM</div>
                <div class="model-card-type">Structural Similarity Index Measure</div>
                <div class="info-box" style="margin:0">
                    <b>Définition originale</b> (vision par ordi) : compare deux images selon luminance, contraste et structure.<br><br>
                    <b>Adaptation classification</b> : compare la matrice de confusion normalisée avec la matrice identité (classificateur parfait).<br><br>
                    <b>↑ meilleur</b> · ∈ [-1, 1]
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="model-card">
                <div class="model-card-title">DCG / NDCG</div>
                <div class="model-card-type">Discounted Cumulative Gain</div>
                <div class="info-box" style="margin:0">
                    <b>Définition</b> : évalue la qualité du ranking des classes par probabilité prédite.<br><br>
                    <b>NDCG</b> = DCG / IDCG (normalisé par le cas idéal où la vraie classe est toujours en position 1).<br><br>
                    <b>↑ meilleur</b> · ∈ [0, 1]
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### Comparaison ASD · SSIM · NDCG")
        adv_metrics = ['ASD', 'SSIM', 'NDCG']
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.patch.set_facecolor('white')
        for ax, metric in zip(axes, adv_metrics):
            vals = [summary[n][metric] for n in models_list]
            is_lower = metric == 'ASD'
            best = min(vals) if is_lower else max(vals)
            colors_m = ['#059669' if v==best else '#94a3b8' for v in vals]
            bars = ax.bar(models_list, vals, color=colors_m, edgecolor='white')
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2, v+max(vals)*0.01,
                        f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
            arrow = '↓ meilleur' if is_lower else '↑ meilleur'
            ax.set_title(f'{metric} ({arrow})', fontweight='bold')
            plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=8)
            ax.set_facecolor('white'); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()



# PAGE: COMPARAISON FINALE
elif page == "Comparaison Finale":
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">g.</div>
        <h2 class="section-title">Comparaison Finale — Tous les Modèles</h2>
    </div>
    """, unsafe_allow_html=True)

    df_sum = pd.DataFrame(summary).T.round(4)
    lower_better = ['Log Loss', 'FPR', 'RMSLE', 'ASD']

    # Highlight best in each column
    def highlight_best(col):
        is_lower = col.name in lower_better
        best = col.min() if is_lower else col.max()
        worst = col.max() if is_lower else col.min()
        styles = []
        for v in col:
            if v == best:
                styles.append('background-color:#dcfce7;color:#064e3b;font-weight:700')
            elif v == worst:
                styles.append('background-color:#fee2e2;color:#991b1b')
            else:
                styles.append('')
        return styles

    styled = df_sum.style.apply(highlight_best, axis=0).format("{:.4f}")
    st.dataframe(styled, use_container_width=True, height=300)

    st.markdown("""
    <div class="warning-box">
    (+) <b>Vert</b> = meilleur score · (-) <b>Rouge</b> = moins bon score<br>
    Pour Log Loss, FPR, RMSLE, ASD : <b>↓ meilleur</b> · Pour toutes les autres : <b>↑ meilleur</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Heatmap normalisee - vue globale")
    df_norm = df_sum.copy()
    for col in df_norm.columns:
        mn, mx = df_norm[col].min(), df_norm[col].max()
        df_norm[col] = (df_norm[col] - mn) / (mx - mn) if mx > mn else 0.5
        if col in lower_better:
            df_norm[col] = 1 - df_norm[col]

    fig, ax = plt.subplots(figsize=(16, 5))
    fig.patch.set_facecolor('white')
    sns.heatmap(df_norm, annot=df_sum.values, fmt='.3f', cmap='RdYlGn',
                ax=ax, linewidths=.5, annot_kws={'size': 9},
                cbar_kws={'label': 'Score normalisé (vert=meilleur)'})
    ax.set_title('Tableau de bord final — valeurs réelles, couleur normalisée', fontweight='bold', fontsize=12, pad=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha='right', fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Best model podium
    st.markdown("#### Classement par Accuracy")
    sorted_models = sorted(summary.items(), key=lambda x: x[1]['Accuracy'], reverse=True)
    medals = ["1er", "2eme", "3eme"]
    cols = st.columns(len(sorted_models))
    for i, (name, metrics) in enumerate(sorted_models):
        medal = medals[i] if i < 3 else f"#{i+1}"
        with cols[i]:
            st.markdown(f"""
            <div class="model-card" style="text-align:center;padding:1rem;">
                <div style="font-size:1.8rem;">{medal}</div>
                <div class="model-card-title">{name}</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:1.3rem;color:#064e3b;font-weight:700;">
                    {metrics['Accuracy']*100:.2f}%
                </div>
                <div style="font-size:0.75rem;color:#64748b;margin-top:0.3rem;">
                    F1: {metrics['F1 / Dice']:.3f} · AUC: {metrics['AUC']:.3f}
                </div>
            </div>
            """, unsafe_allow_html=True)


# PAGE: PRÉDICTION LIVE
elif page == "Prediction Live":
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">h.</div>
        <h2 class="section-title">Prédiction Live — Recommandation de Culture</h2>
    </div>
    <div class="info-box">
        Entrez les conditions pédoclimatiques de votre parcelle et obtenez une recommandation 
        de culture avec les probabilités de chaque modèle.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.markdown("#### Parametres pedoclimatiques")
        azote    = st.slider("Azote N (mg/kg)",      0, 150, 60)
        phosphore= st.slider("Phosphore P (mg/kg)",  0, 150, 45)
        potassium= st.slider("Potassium K (mg/kg)",  0, 210, 45)
        temp     = st.slider("Temperature (degC)",        5.0, 50.0, 25.0, 0.5)
        humidite = st.slider("Humidite (%)",            10.0, 100.0, 65.0, 0.5)
        ph       = st.slider("pH du sol",               3.0, 10.0, 6.5, 0.1)
        pluv     = st.slider("Pluviometrie (mm)",       20.0, 300.0, 120.0, 1.0)

        model_pred_choice = st.selectbox(
            "Modèle de prédiction",
            ["Random Forest", "Decision Tree", "Naive Bayes", "KNN"]
        )

        predict_btn = st.button("Predire la culture optimale", use_container_width=True)

    with col2:
        if predict_btn:
            X_input = np.array([[azote, phosphore, potassium, temp, humidite, ph, pluv]])
            X_scaled = M['scaler'].transform(X_input)

            model_map = {
                "Random Forest": M['rf'],
                "Decision Tree": M['dt'],
                "Naive Bayes":   M['nb'],
                "KNN":           M['knn'],
            }
            model_sel = model_map[model_pred_choice]
            pred_class   = model_sel.predict(X_scaled)[0]
            pred_proba   = model_sel.predict_proba(X_scaled)[0]
            pred_culture = le.inverse_transform([pred_class])[0]
            confidence   = pred_proba[pred_class] * 100

            st.markdown(f"""
            <div class="prediction-result">
                <div style="font-size:0.9rem;color:#a7f3d0;text-transform:uppercase;letter-spacing:1px;">Culture recommandée</div>
                <div class="prediction-crop">{pred_culture}</div>
                <div class="prediction-conf">Confiance : {confidence:.1f}%</div>
                <div style="margin-top:0.8rem;font-size:0.85rem;color:#d1fae5;">via {model_pred_choice}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### Top 8 cultures par probabilite")
            top_idx  = np.argsort(pred_proba)[::-1][:8]
            top_crops = [(le.inverse_transform([i])[0], pred_proba[i]*100) for i in top_idx]

            fig, ax = plt.subplots(figsize=(7, 4))
            fig.patch.set_facecolor('white')
            names_t = [c[0] for c in top_crops]
            vals_t  = [c[1] for c in top_crops]
            colors_t = ['#059669' if i==0 else '#94a3b8' for i in range(len(vals_t))]
            bars = ax.barh(names_t[::-1], vals_t[::-1], color=colors_t[::-1], edgecolor='white')
            for bar, v in zip(bars, vals_t[::-1]):
                ax.text(v + 0.3, bar.get_y()+bar.get_height()/2,
                        f'{v:.1f}%', va='center', fontsize=9, fontweight='bold')
            ax.set_xlabel('Probabilité (%)'); ax.set_xlim(0, max(vals_t)*1.2)
            ax.set_facecolor('white'); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
        else:
            st.markdown("""
            <div style="text-align:center;padding:4rem 2rem;color:#94a3b8;">
                <div style="font-size:2rem;margin-bottom:1rem;color:#94a3b8;">[ ... ]</div>
                <div style="font-size:1.1rem;font-weight:500;">
                    Ajustez les paramètres<br>et cliquez sur <b>Prédire</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    Crop Recommendation Analytics - Probabilite Appliquee pour l'IA 2 - Pr. Mohammed KAICER<br>
   
</div>
""", unsafe_allow_html=True)
