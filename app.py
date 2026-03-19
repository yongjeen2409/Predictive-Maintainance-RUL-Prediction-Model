import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import mean_squared_error

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Predictive Maintenance", layout="wide")

# ==========================================
# CONSTANTS & DEFINITIONS
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUL_CAP = 125
SEQ_LEN = 40
D_MODEL = 96
N_HEAD = 3
NUM_LAYERS = 3
DROPOUT = 0.2
HEALTH_DIM = 64

op_cols = ['op_setting_1', 'op_setting_2', 'op_setting_3']
all_sensor_cols = [f'sensor_{i}' for i in range(1, 22)]

# ==========================================
# DYNAMIC LOADER & ARCHITECTURE
# ==========================================
@st.cache_resource
def load_production_core():
    if not os.path.exists('production_core.pt'):
        return None, None, None, None
    
    package = torch.load('production_core.pt', map_location=device, weights_only=False)
    pipeline = package['pipeline']
    
    important_sensors = pipeline['sensors']
    feature_cols = important_sensors + [f'{c}_diff' for c in important_sensors] + \
                   [f'{c}_mean_{w}' for c in important_sensors for w in [5, 10, 20, 30]] + \
                   [f'{c}_std_{w}' for c in important_sensors for w in [5, 10, 20, 30]] + \
                   [f'{c}_slope_{w}' for c in important_sensors for w in [10, 20]]
    
    class Generalized_PM_Architecture(nn.Module):
        def __init__(self, num_features=len(feature_cols), d_model=D_MODEL, nhead=N_HEAD, num_layers=NUM_LAYERS, dropout=DROPOUT):
            super().__init__()
            self.conv1 = nn.Conv1d(num_features, d_model, kernel_size=3, padding=1, dilation=1)
            self.norm1 = nn.GroupNorm(1, d_model) 
            self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=2, dilation=2)
            self.norm2 = nn.GroupNorm(1, d_model)
            self.conv3 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=4, dilation=4)
            self.norm3 = nn.GroupNorm(1, d_model)
            self.relu = nn.ReLU()

            pe = torch.zeros(SEQ_LEN, d_model)
            position = torch.arange(0, SEQ_LEN, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pos_encoder', pe.unsqueeze(0)) 

            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True, norm_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
            self.attn = nn.Sequential(nn.Linear(d_model, 128), nn.GELU(), nn.Linear(128, 1))

            self.health_embedding = nn.Sequential(nn.Linear(d_model, HEALTH_DIM), nn.GELU(), nn.Dropout(dropout))
            self.rul_head = nn.Linear(HEALTH_DIM, 2) 

            self.decoder_expansion = nn.Linear(HEALTH_DIM, d_model)
            self.decoder_upsample = nn.ConvTranspose1d(in_channels=d_model, out_channels=d_model, kernel_size=SEQ_LEN)
            self.decoder_conv = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(d_model, num_features, kernel_size=3, padding=1)
            )

        def forward(self, x):
            x_in = x.permute(0, 2, 1) 
            res1 = self.relu(self.norm1(self.conv1(x_in)))
            res2 = self.relu(self.norm2(self.conv2(res1)) + res1) 
            h    = self.relu(self.norm3(self.conv3(res2)) + res2) 
            h = h.permute(0, 2, 1) 
            h = h + self.pos_encoder
            h = self.transformer(h)
            attn_scores = self.attn(h).squeeze(-1) 
            attn_weights = torch.softmax(attn_scores, dim=1) 
            context_vector = torch.sum(attn_weights.unsqueeze(-1) * h, dim=1) 
            health_idx = self.health_embedding(context_vector)
            rul_out = self.rul_head(health_idx)
            pred_rul = self.relu(rul_out[:, 0]) 
            log_var = rul_out[:, 1]             
            dec_h = self.decoder_expansion(health_idx).unsqueeze(-1)                
            dec_h = self.decoder_upsample(dec_h)       
            reconstruction = self.decoder_conv(dec_h).permute(0, 2, 1) 
            return pred_rul, log_var, reconstruction, health_idx

    models = []
    for weights in package['ensemble_weights']:
        model = Generalized_PM_Architecture(num_features=len(feature_cols)).to(device)
        model.load_state_dict(weights)
        model.eval()
        models.append(model)
        
    return pipeline, models, important_sensors, feature_cols

pipeline, ensemble_models, IMPORTANT_SENSORS, feature_cols = load_production_core()

# ==========================================
# FEATURE ENGINEERING & MATH HELPERS
# ==========================================
def _rolling_slope_fast(series, w):
    x = np.arange(w, dtype=np.float32)
    x -= x.mean()
    x_var = (x ** 2).sum()
    def _slope(arr):
        arr = arr - arr.mean()
        return (x[:len(arr)] * arr).sum() / (x_var + 1e-8)
    return series.rolling(w, min_periods=w).apply(_slope, raw=True).fillna(0)

def add_generalized_features(df, sensors, span=5, windows=[5, 10, 20, 30]):
    df[sensors] = df.groupby('unique_engine_id')[sensors].transform(lambda x: x.ewm(span=span, adjust=False).mean()).astype(np.float32)
    new_features = {}
    for col in sensors:
        grp = df.groupby('unique_engine_id')[col]
        new_features[f'{col}_diff'] = grp.diff().fillna(0).astype(np.float32)
        for w in windows:
            new_features[f'{col}_mean_{w}'] = grp.transform(lambda x: x.rolling(w, min_periods=1).mean()).astype(np.float32)
            new_features[f'{col}_std_{w}'] = grp.transform(lambda x: x.rolling(w, min_periods=1).std().fillna(0)).astype(np.float32)
        for w in [10, 20]:
            new_features[f'{col}_slope_{w}'] = grp.transform(lambda x: _rolling_slope_fast(x, w)).astype(np.float32)
    df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
    return df.fillna(0)

def create_sequences(df, label_col=None, is_test=False):
    X, y, eids = [], [], []
    for engine_id, group in df.groupby('unique_engine_id', sort=False):
        data = group[feature_cols].values
        labels = group[label_col].values if label_col else np.zeros(len(data))
        if len(data) < SEQ_LEN:
            pad_size = SEQ_LEN - len(data)
            data = np.pad(data, ((pad_size, 0), (0, 0)), 'edge')
            labels = np.pad(labels, (pad_size, 0), 'edge')
        if is_test:
            X.append(data[-SEQ_LEN:])
            y.append(labels[-1])
            eids.append(engine_id)
        else:
            for i in range(len(data) - SEQ_LEN + 1):
                X.append(data[i : i + SEQ_LEN])
                y.append(labels[i + SEQ_LEN - 1])
                eids.append(engine_id)
    if len(X) == 0:
        X = np.zeros((1, SEQ_LEN, len(feature_cols)), dtype=np.float32)
        y = np.zeros(1, dtype=np.float32)
        eids = np.array(["Unknown"])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(eids, dtype=object)

def calculate_cmapss_score(y_true, y_pred):
    d = y_pred - y_true
    score = 0
    for error in d:
        if error < 0:
            score += np.exp(-error / 13.0) - 1
        else:
            score += np.exp(error / 10.0) - 1
    return score

# ==========================================
# SHARED PLOT STYLE
# ==========================================
ACTUAL_COLOR = '#2c3e50'
PRED_COLOR   = '#e74c3c'
DANGER_COLOR = '#c0392b'
SAFE_COLOR   = '#27ae60'

def style_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(fontsize=10, framealpha=0.8)
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# ==========================================
# NAVIGATION SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063822.png", width=60)
    st.title("SOTA Engine Core")
    st.markdown("---")
    page = st.selectbox("Operation Mode", ["📈 Benchmark Performance", "🧪 Real-time Inference"])

# ==========================================
# PAGE 1: MODEL PERFORMANCE
# ==========================================
if page == "📈 Benchmark Performance":
    st.title("📈 Fleet Operations & Benchmarks")
    st.markdown("Evaluate the Generalized Ensemble against historical CMAPSS test fleets.")

    if not pipeline:
        st.error("⚠️ `production_core.pt` not found. Please place the exported model file in this directory.")
    elif not os.path.exists('./CMAPSSData'):
        st.error("⚠️ `./CMAPSSData` folder missing. Required for benchmark evaluation.")
    else:
        if st.button("▶ Run Full Fleet Evaluation", type="primary"):
            with st.spinner("Processing 707 test engines..."):
                col_names = ['engine_id', 'cycle'] + op_cols + all_sensor_cols
                test_list, rul_list = [], []

                for i in range(1, 5):
                    te = pd.read_csv(f'./CMAPSSData/test_FD00{i}.txt', sep=r'\s+', header=None, names=col_names)
                    te['dataset'] = f'FD00{i}'
                    te['unique_engine_id'] = te['dataset'] + '_' + te['engine_id'].astype(str)
                    test_list.append(te)

                    r = pd.read_csv(f'./CMAPSSData/RUL_FD00{i}.txt', sep=r'\s+', header=None, names=['RUL'])
                    r['dataset'] = f'FD00{i}'
                    r['engine_id'] = r.index + 1
                    r['unique_engine_id'] = r['dataset'] + '_' + r['engine_id'].astype(str)
                    rul_list.append(r)

                test_df = pd.concat(test_list, ignore_index=True)
                true_rul_df = pd.concat(rul_list, ignore_index=True)

                true_rul_df['RUL'] = true_rul_df['RUL'].clip(upper=RUL_CAP)
                test_max_cycles = test_df.groupby('unique_engine_id', sort=False)['cycle'].max().reset_index()
                test_max_cycles = test_max_cycles.merge(true_rul_df[['unique_engine_id', 'RUL']], on='unique_engine_id')
                test_df = test_df.merge(test_max_cycles[['unique_engine_id', 'RUL']], on='unique_engine_id', how='left')
                test_df.rename(columns={'RUL': 'true_rul'}, inplace=True)

                kmeans  = pipeline['kmeans']
                scalers = pipeline['scalers']
                test_df[all_sensor_cols] = test_df[all_sensor_cols].astype(np.float32)
                test_df['regime'] = kmeans.predict(test_df[op_cols])

                for regime in range(6):
                    if regime in scalers:
                        mask = test_df['regime'] == regime
                        if mask.sum() > 0:
                            test_df.loc[mask, IMPORTANT_SENSORS] = scalers[regime].transform(test_df.loc[mask, IMPORTANT_SENSORS])

                test_df = add_generalized_features(test_df, IMPORTANT_SENSORS)
                X_test_seq, y_test_seq, test_eids = create_sequences(test_df, label_col='true_rul', is_test=True)
                X_test_tensor = torch.tensor(X_test_seq, device=device)

                ensemble_predictions, ensemble_variances = [], []
                with torch.no_grad():
                    for model in ensemble_models:
                        pred_rul, log_var, _, _ = model(X_test_tensor)
                        ensemble_predictions.append(np.clip(pred_rul.cpu().numpy().flatten(), 0, RUL_CAP))
                        ensemble_variances.append(torch.exp(log_var).cpu().numpy().flatten())

                ensemble_preds  = np.array(ensemble_predictions)
                final_mu        = np.mean(ensemble_preds, axis=0)
                aleatoric_var   = np.mean(np.array(ensemble_variances), axis=0)
                epistemic_var   = np.var(ensemble_preds, axis=0)
                total_sigma     = np.sqrt(aleatoric_var + epistemic_var)

                rmse       = np.sqrt(mean_squared_error(y_test_seq, final_mu))
                nasa_score = calculate_cmapss_score(y_test_seq, final_mu)

                st.session_state['eval_data'] = {
                    'test_eids':  np.array(test_eids),
                    'y_test_seq': y_test_seq,
                    'final_mu':   final_mu,
                    'total_sigma': total_sigma,
                    'rmse':       rmse,
                    'nasa_score': nasa_score,
                }

        # ------------------------------------------
        # RENDER DASHBOARD
        # ------------------------------------------
        if 'eval_data' in st.session_state:
            data = st.session_state['eval_data']

            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            col1.metric("Global Fleet RMSE",            f"{data['rmse']:.2f} Cycles")
            col2.metric("NASA Asymmetric Penalty Score", f"{data['nasa_score']:.1f}")
            col3.metric("Ensemble Size",                 f"{len(ensemble_models)} Models Active")
            st.markdown("---")

            # ---- Actual vs Predicted (single view, no tabs) ----
            st.subheader("🎯 Actual vs Predicted RUL")



            SMOOTH_W = 8
            CRITICAL_THRESHOLD = 30

            ds_options2 = {
                "All Fleets (707 engines)":           None,
                "FD001 — Sea Level":                  "FD001",
                "FD002 — 6 Conditions":               "FD002",
                "FD003 — Sea Level + Fault 2":        "FD003",
                "FD004 — 6 Conditions + Fault 2":     "FD004",
            }
            selected_label2 = st.selectbox(
                "Select fleet:", list(ds_options2.keys()), key="avp_ds"
            )
            ds_key2 = ds_options2[selected_label2]

            mask2 = (
                np.ones(len(data['test_eids']), dtype=bool)
                if ds_key2 is None
                else np.array([ds_key2 in eid for eid in data['test_eids']])
            )

            if mask2.sum() > 0:
                y_true2 = data['y_test_seq'][mask2]
                y_pred2 = data['final_mu'][mask2]

                sort_idx2 = np.argsort(y_true2)[::-1]
                y_true_s  = y_true2[sort_idx2]
                y_pred_s  = y_pred2[sort_idx2]

                def smooth(arr, w):
                    if w <= 1:
                        return arr.copy()
                    kernel = np.ones(w) / w
                    pad    = np.pad(arr, w // 2, mode='reflect')
                    return np.convolve(pad, kernel, mode='valid')[:len(arr)]

                y_true_sm = smooth(y_true_s, SMOOTH_W)
                y_pred_sm = smooth(y_pred_s, SMOOTH_W)
                residual  = y_pred_s - y_true_s
                res_sm    = smooth(residual, SMOOTH_W)

                x2    = np.arange(len(y_true_s))
                y_min = max(0,       min(y_true_sm.min(), y_pred_sm.min()) - 8)
                y_max = min(RUL_CAP, max(y_true_sm.max(), y_pred_sm.max()) + 8)

                fig2, axes = plt.subplots(
                    2, 1, figsize=(16, 10),
                    gridspec_kw={'height_ratios': [3, 1.2]},
                    dpi=130
                )

                # ── Top panel ──────────────────────────────────────────────────
                axes[0].plot(x2, y_true_sm, label='Actual RUL',
                             color=ACTUAL_COLOR, linewidth=2.8, zorder=3)
                axes[0].plot(x2, y_pred_sm, label='Predicted RUL',
                             color=PRED_COLOR, linewidth=2.0, linestyle='--', zorder=3)

                axes[0].fill_between(x2, y_true_sm, y_pred_sm,
                                     where=(y_pred_sm > y_true_sm),
                                     color=DANGER_COLOR, alpha=0.12, label='Overestimate region')
                axes[0].fill_between(x2, y_true_sm, y_pred_sm,
                                     where=(y_pred_sm <= y_true_sm),
                                     color=SAFE_COLOR, alpha=0.10, label='Underestimate region')

                axes[0].axhline(CRITICAL_THRESHOLD, color=DANGER_COLOR,
                                linestyle=':', linewidth=1.2, alpha=0.6,
                                label=f'Critical zone (≤{CRITICAL_THRESHOLD} cycles)')
                axes[0].fill_between(x2, 0, CRITICAL_THRESHOLD,
                                     color=DANGER_COLOR, alpha=0.05)
                axes[0].set_ylim(y_min, y_max)
                style_ax(axes[0],
                         f"Actual vs Predicted — {selected_label2}",
                         "", "Remaining useful life (cycles)")

                # ── Bottom panel ───────────────────────────────────────────────
                axes[1].plot(x2, res_sm, color='#8e44ad', linewidth=1.4,
                             label='Residual (pred − actual)')
                axes[1].axhline(0, color='black', linewidth=0.8)
                axes[1].fill_between(x2, res_sm, 0, where=(res_sm > 0),
                                     color=DANGER_COLOR, alpha=0.22,
                                     label='Overestimate (dangerous)')
                axes[1].fill_between(x2, res_sm, 0, where=(res_sm <= 0),
                                     color=SAFE_COLOR, alpha=0.18,
                                     label='Underestimate (safe)')

                res_abs = max(abs(res_sm.min()), abs(res_sm.max())) + 5
                axes[1].set_ylim(-res_abs, res_abs)
                style_ax(axes[1],
                         "Prediction error (smoothed residual)",
                         "Engine rank (sorted healthiest → critical)",
                         "Error (cycles)")

                plt.tight_layout(h_pad=2.5)
                st.pyplot(fig2)
                plt.close(fig2)

                # ── Overall error summary ──────────────────────────────────────
                over  = int(np.sum(residual > 0))
                under = int(np.sum(residual < 0))
                exact = int(np.sum(residual == 0))
                ds_rmse2 = np.sqrt(mean_squared_error(y_true2, y_pred2))
                total_n  = len(y_true2)

                st.markdown("#### Overall prediction breakdown")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Subset RMSE",               f"{ds_rmse2:.2f} cycles")
                c2.metric("Overestimates (dangerous)",  f"{over}  ({over/total_n*100:.1f}%)")
                c3.metric("Underestimates (safe)",      f"{under}  ({under/total_n*100:.1f}%)")
                c4.metric("Exact matches",              f"{exact}  ({exact/total_n*100:.1f}%)")

                # ── Critical zone breakdown ────────────────────────────────────
                st.markdown("---")
                st.markdown(
                    f"#### 🚨 Critical zone analysis  —  engines with actual RUL ≤ {CRITICAL_THRESHOLD} cycles"
                )

                # Compute critical zone stats from raw (unsorted, unsmoothed) arrays
                crit_mask    = y_true2 <= CRITICAL_THRESHOLD
                n_crit       = int(crit_mask.sum())

                if n_crit > 0:
                    crit_true = y_true2[crit_mask]
                    crit_pred = y_pred2[crit_mask]
                    crit_res  = crit_pred - crit_true

                    # An overestimate in the critical zone: model predicts > threshold
                    # (model thinks engine is healthy, but it is actually in danger)
                    crit_over      = int(np.sum(crit_pred > CRITICAL_THRESHOLD))
                    crit_under     = int(np.sum(crit_pred <= CRITICAL_THRESHOLD))
                    crit_miss_rate = crit_over / n_crit * 100
                    crit_catch_rate= crit_under / n_crit * 100
                    crit_rmse      = np.sqrt(mean_squared_error(crit_true, crit_pred))
                    mean_over_err  = float(np.mean(crit_res[crit_res > 0])) if crit_over > 0 else 0.0
                    mean_under_err = float(np.mean(np.abs(crit_res[crit_res <= 0]))) if crit_under > 0 else 0.0

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Engines in critical zone", n_crit)
                    m2.metric("✅ Catch rate (correctly flagged)",
                              f"{crit_catch_rate:.1f}%",
                              delta=None)
                    m3.metric("❌ Miss rate (danger not flagged)",
                              f"{crit_miss_rate:.1f}%",
                              delta=None)
                    m4.metric("Critical zone RMSE", f"{crit_rmse:.2f} cycles")

                    st.markdown("&nbsp;")
                    d1, d2 = st.columns(2)

                    with d1:
                        st.markdown(
                            f"""
<div style="background:#FCEBEB;border-left:4px solid #E24B4A;padding:14px 16px;border-radius:6px">
<b style="color:#A32D2D">⚠️ Dangerous overestimates in critical zone</b><br>
<span style="font-size:2rem;font-weight:600;color:#A32D2D">{crit_over}</span>
<span style="color:#A32D2D"> engines  ({crit_miss_rate:.1f}%)</span><br>
<span style="font-size:0.85rem;color:#791F1F">
Model predicted these engines were healthy (RUL > {CRITICAL_THRESHOLD}) but they had ≤ {CRITICAL_THRESHOLD} cycles left.
Average overestimate: <b>+{mean_over_err:.1f} cycles</b>
</span>
</div>
""", unsafe_allow_html=True)

                    with d2:
                        st.markdown(
                            f"""
<div style="background:#EAF3DE;border-left:4px solid #639922;padding:14px 16px;border-radius:6px">
<b style="color:#27500A">✅ Safe underestimates in critical zone</b><br>
<span style="font-size:2rem;font-weight:600;color:#27500A">{crit_under}</span>
<span style="color:#27500A"> engines  ({crit_catch_rate:.1f}%)</span><br>
<span style="font-size:0.85rem;color:#3B6D11">
Model correctly flagged these engines as critical (RUL ≤ {CRITICAL_THRESHOLD}).
Average early-alert margin: <b>−{mean_under_err:.1f} cycles</b>
</span>
</div>
""", unsafe_allow_html=True)

                    # Mini bar chart: catch vs miss
                    st.markdown("&nbsp;")
                    fig3, ax3 = plt.subplots(figsize=(7, 2.8), dpi=120)
                    bars = ax3.barh(
                        ['Miss (dangerous)', 'Catch (safe)'],
                        [crit_miss_rate, crit_catch_rate],
                        color=[DANGER_COLOR, SAFE_COLOR],
                        height=0.45
                    )
                    for bar, val in zip(bars, [crit_miss_rate, crit_catch_rate]):
                        ax3.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                                 f"{val:.1f}%", va='center', fontsize=11, fontweight='bold')
                    ax3.set_xlim(0, 105)
                    ax3.set_xlabel("% of critical-zone engines")
                    ax3.set_title(f"Critical zone alert performance  (n = {n_crit} engines)", fontsize=11)
                    ax3.spines['top'].set_visible(False)
                    ax3.spines['right'].set_visible(False)
                    ax3.grid(axis='x', linestyle='--', alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig3)
                    plt.close(fig3)

                else:
                    st.info(f"No engines in the selected fleet have actual RUL ≤ {CRITICAL_THRESHOLD} cycles.")

# ==========================================
# PAGE 2: REAL-TIME INFERENCE
# ==========================================
elif page == "🧪 Real-time Inference":
    import torch.optim as optim

    st.title("🧪 Real-time Inference & Fine-tuning")

    if not pipeline:
        st.error("⚠️ `production_core.pt` not found. Train the model first.")
        st.stop()

    # ── Helper: build architecture class using loaded feature_cols ──────────
    def build_model():
        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                nf = len(feature_cols)
                self.conv1 = nn.Conv1d(nf, D_MODEL, kernel_size=3, padding=1, dilation=1)
                self.norm1 = nn.GroupNorm(1, D_MODEL)
                self.conv2 = nn.Conv1d(D_MODEL, D_MODEL, kernel_size=3, padding=2, dilation=2)
                self.norm2 = nn.GroupNorm(1, D_MODEL)
                self.conv3 = nn.Conv1d(D_MODEL, D_MODEL, kernel_size=3, padding=4, dilation=4)
                self.norm3 = nn.GroupNorm(1, D_MODEL)
                self.relu  = nn.ReLU()

                pe = torch.zeros(SEQ_LEN, D_MODEL)
                pos = torch.arange(0, SEQ_LEN, dtype=torch.float).unsqueeze(1)
                div = torch.exp(torch.arange(0, D_MODEL, 2).float() * (-np.log(10000.0) / D_MODEL))
                pe[:, 0::2] = torch.sin(pos * div)
                pe[:, 1::2] = torch.cos(pos * div)
                self.register_buffer('pos_encoder', pe.unsqueeze(0))

                enc = nn.TransformerEncoderLayer(
                    d_model=D_MODEL, nhead=N_HEAD, dim_feedforward=D_MODEL * 4,
                    dropout=DROPOUT, batch_first=True, norm_first=True)
                self.transformer = nn.TransformerEncoder(enc, num_layers=NUM_LAYERS,
                                                         enable_nested_tensor=False)
                self.attn = nn.Sequential(nn.Linear(D_MODEL, 128), nn.GELU(), nn.Linear(128, 1))

                self.health_embedding = nn.Sequential(
                    nn.Linear(D_MODEL, HEALTH_DIM), nn.GELU(), nn.Dropout(DROPOUT))
                self.rul_head = nn.Linear(HEALTH_DIM, 2)

                self.decoder_expansion = nn.Linear(HEALTH_DIM, D_MODEL)
                self.decoder_upsample  = nn.ConvTranspose1d(D_MODEL, D_MODEL, kernel_size=SEQ_LEN)
                self.decoder_conv = nn.Sequential(
                    nn.Conv1d(D_MODEL, D_MODEL, 3, padding=1), nn.GELU(),
                    nn.Conv1d(D_MODEL, nf, 3, padding=1))

            def forward(self, x):
                h = x.permute(0, 2, 1)
                r1 = self.relu(self.norm1(self.conv1(h)))
                r2 = self.relu(self.norm2(self.conv2(r1)) + r1)
                h  = self.relu(self.norm3(self.conv3(r2)) + r2)
                h  = h.permute(0, 2, 1) + self.pos_encoder
                h  = self.transformer(h)
                aw = torch.softmax(self.attn(h).squeeze(-1), dim=1)
                cv = torch.sum(aw.unsqueeze(-1) * h, dim=1)
                hi = self.health_embedding(cv)
                out = self.rul_head(hi)
                pred_rul = self.relu(out[:, 0])
                log_var  = out[:, 1]
                dec = self.decoder_expansion(hi).unsqueeze(-1)
                dec = self.decoder_conv(self.decoder_upsample(dec)).permute(0, 2, 1)
                return pred_rul, log_var, dec, hi
        return _Model()

    # ── Helper: apply pipeline and featurise an uploaded df ─────────────────
    def prepare_df(df_raw):
        """Apply KMeans + scalers + feature engineering to a raw uploaded df."""
        df = df_raw.copy()
        df[all_sensor_cols] = df[all_sensor_cols].astype(np.float32)
        df['regime'] = pipeline['kmeans'].predict(df[op_cols])
        for regime, scaler in pipeline['scalers'].items():
            mask = df['regime'] == regime
            if mask.sum() > 0:
                df.loc[mask, IMPORTANT_SENSORS] = scaler.transform(
                    df.loc[mask, IMPORTANT_SENSORS])
        df = add_generalized_features(df, IMPORTANT_SENSORS)
        return df

    # ── Helper: ensemble TTA inference ──────────────────────────────────────
    def run_inference(models_list, X_tensor):
        SMOOTH = 8

        def _tta(model, Xt):
            preds_aug = []
            for offset in range(5):
                drop = offset * 2
                if offset == 0:
                    x = Xt
                else:
                    pad = Xt[:, 0:1, :].repeat(1, drop, 1)
                    x   = torch.cat([pad, Xt[:, :-drop, :]], dim=1)
                with torch.no_grad():
                    p, _, _, _ = model(x)
                preds_aug.append(np.clip(p.cpu().numpy().flatten(), 0, RUL_CAP))
            return np.mean(preds_aug, axis=0)

        all_preds, all_vars = [], []
        for m in models_list:
            m.eval()
            all_preds.append(_tta(m, X_tensor))
            with torch.no_grad():
                _, lv, _, _ = m(X_tensor)
            all_vars.append(torch.exp(lv).cpu().numpy().flatten())

        mu       = np.mean(all_preds, axis=0)
        aleat    = np.mean(all_vars,  axis=0)
        epist    = np.var(all_preds,  axis=0)
        sigma    = np.sqrt(aleat + epist)
        return mu, sigma

    # ── Mode selector ────────────────────────────────────────────────────────
    mode = st.radio("Select mode", ["🔍 Run Inference", "🔧 Fine-tune on New Data"],
                    horizontal=True)
    st.markdown("---")

    # =========================================================================
    # MODE A: RUN INFERENCE
    # =========================================================================
    if mode == "🔍 Run Inference":
        st.subheader("🔍 Run Inference on New Sensor Data")

        # Model selector — show fine-tuned option only if file exists
        model_options = ["Original (production_core.pt)"]
        if os.path.exists("finetuned_core.pt"):
            model_options.append("Fine-tuned (finetuned_core.pt)")
        chosen_model = st.selectbox("Model to use:", model_options)

        st.markdown(
            "Upload a CSV with columns: `engine_id`, `cycle`, "
            "`op_setting_1/2/3`, `sensor_1` … `sensor_21`. "
            "No RUL column needed."
        )
        uploaded = st.file_uploader("Upload sensor CSV", type="csv", key="infer_upload")

        if uploaded:
            raw = pd.read_csv(uploaded)
            raw['dataset']          = 'USER'
            raw['unique_engine_id'] = 'USER_' + raw['engine_id'].astype(str)

            with st.spinner("Applying pipeline and running inference..."):
                df_proc = prepare_df(raw)
                X_seq, _, eids = create_sequences(df_proc, label_col=None, is_test=True)
                X_t = torch.tensor(X_seq, dtype=torch.float32, device=device)

                # Load weights
                pkg_path = "finetuned_core.pt" if "Fine-tuned" in chosen_model else "production_core.pt"
                pkg  = torch.load(pkg_path, map_location=device, weights_only=False)
                weight_list = pkg['ensemble_weights']

                models_inf = []
                for w in weight_list:
                    m = build_model().to(device)
                    m.load_state_dict(w)
                    models_inf.append(m)

                mu, sigma = run_inference(models_inf, X_t)

            # Results table
            results_df = pd.DataFrame({
                'Engine':         eids,
                'Predicted RUL':  np.round(mu).astype(int),
                '±2σ (cycles)':   np.round(2 * sigma, 1),
                'Status': [
                    '🚨 CRITICAL'  if r <= 30  else
                    '⚠️ WARNING'   if r <= 60  else
                    '✅ Healthy'
                    for r in mu
                ]
            })

            st.success(f"Inference complete — {len(eids)} engines processed.")
            st.dataframe(results_df, use_container_width=True)

            # RUL bar chart
            fig_inf, ax_inf = plt.subplots(figsize=(max(8, len(eids) * 0.5), 4), dpi=110)
            colours = [DANGER_COLOR if r <= 30 else '#e67e22' if r <= 60 else SAFE_COLOR
                       for r in mu]
            ax_inf.bar(range(len(eids)), mu, color=colours, width=0.6)
            ax_inf.errorbar(range(len(eids)), mu, yerr=2 * sigma,
                            fmt='none', color='black', capsize=3, linewidth=0.8)
            ax_inf.axhline(30, color=DANGER_COLOR, linestyle=':', linewidth=1.2,
                           label='Critical (30 cycles)')
            ax_inf.axhline(60, color='#e67e22',    linestyle=':', linewidth=1.0,
                           label='Warning (60 cycles)')
            ax_inf.set_xticks(range(len(eids)))
            ax_inf.set_xticklabels([str(e).replace('USER_', '') for e in eids],
                                    rotation=45, ha='right', fontsize=9)
            ax_inf.set_ylabel("Predicted RUL (cycles)")
            ax_inf.set_title("Per-engine RUL predictions with ±2σ uncertainty")
            ax_inf.legend(fontsize=9)
            ax_inf.spines['top'].set_visible(False)
            ax_inf.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_inf)
            plt.close(fig_inf)

    # =========================================================================
    # MODE B: FINE-TUNE
    # =========================================================================
    else:
        st.subheader("🔧 Fine-tune on Your Machine Data")

        st.info(
            "The original `production_core.pt` is **never modified**. "
            "Fine-tuning updates only the health embedding and RUL head layers, "
            "then saves the result as `finetuned_core.pt`."
        )

        st.markdown(
            "Upload a **labelled** CSV with columns: `engine_id`, `cycle`, "
            "`op_setting_1/2/3`, `sensor_1` … `sensor_21`, **and a `RUL` column** "
            "(the remaining useful life for each row, capped at 125)."
        )
        ft_file = st.file_uploader("Upload labelled training CSV", type="csv", key="ft_upload")

        col_ep, col_lr = st.columns(2)
        ft_epochs = col_ep.number_input("Fine-tune epochs", min_value=5, max_value=60,
                                         value=15, step=5)
        ft_lr     = col_lr.number_input("Learning rate", min_value=1e-5, max_value=1e-3,
                                         value=1e-4, step=1e-5, format="%.5f")

        if ft_file and st.button("▶ Start Fine-tuning", type="primary"):

            raw_ft = pd.read_csv(ft_file)
            raw_ft['dataset']          = 'FINETUNE'
            raw_ft['unique_engine_id'] = 'FT_' + raw_ft['engine_id'].astype(str)

            if 'RUL' not in raw_ft.columns:
                st.error("CSV must contain a `RUL` column. Fine-tuning aborted.")
                st.stop()

            with st.spinner("Preparing data..."):
                raw_ft['RUL'] = raw_ft['RUL'].clip(upper=RUL_CAP).astype(np.float32)
                df_ft = prepare_df(raw_ft)
                X_ft, y_ft, _ = create_sequences(df_ft, label_col='RUL', is_test=False)

            n_seq = len(X_ft)
            st.write(f"Training sequences generated: **{n_seq}**")

            # Split last 15% of engines for validation
            unique_engines = raw_ft['unique_engine_id'].unique()
            n_val_eng  = max(1, int(len(unique_engines) * 0.15))
            val_engines = set(unique_engines[-n_val_eng:])
            # Rebuild sequences split by engine
            df_tr_ft = df_ft[~df_ft['unique_engine_id'].isin(val_engines)].copy()
            df_vl_ft = df_ft[ df_ft['unique_engine_id'].isin(val_engines)].copy()
            X_tr, y_tr, _ = create_sequences(df_tr_ft, label_col='RUL', is_test=False)
            X_vl, y_vl, _ = create_sequences(df_vl_ft, label_col='RUL', is_test=False)

            X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
            y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)
            X_vl_t = torch.tensor(X_vl, dtype=torch.float32, device=device)

            from torch.utils.data import TensorDataset, DataLoader

            train_ds = TensorDataset(X_tr_t, y_tr_t)
            loader   = DataLoader(train_ds, batch_size=64, shuffle=True)

            # Load original weights into a fresh model
            pkg_orig   = torch.load('production_core.pt', map_location=device,
                                    weights_only=False)
            # Use the first ensemble member as the base (best single model is seed index 0)
            base_weights = pkg_orig['ensemble_weights'][0]
            ft_model = build_model().to(device)
            ft_model.load_state_dict(base_weights)

            # Freeze everything except health_embedding and rul_head
            frozen_prefixes = ('conv1', 'conv2', 'conv3', 'norm1', 'norm2', 'norm3',
                               'transformer', 'attn', 'pos_encoder',
                               'decoder_expansion', 'decoder_upsample', 'decoder_conv')
            for name, param in ft_model.named_parameters():
                if name.startswith(frozen_prefixes):
                    param.requires_grad = False

            trainable = sum(p.numel() for p in ft_model.parameters() if p.requires_grad)
            st.write(f"Trainable parameters (health_embedding + rul_head): **{trainable:,}**")

            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, ft_model.parameters()),
                lr=ft_lr, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7)
            mse_loss  = nn.MSELoss()

            def _het_loss(pred, true, lv):
                prec = torch.exp(-lv)
                err  = pred - true
                w    = torch.where(err > 0, torch.full_like(err, 2.5), torch.ones_like(err))
                return torch.mean(0.5 * prec * w * err**2 + 0.5 * lv)

            # Live progress
            progress_bar  = st.progress(0)
            status_text   = st.empty()
            chart_ph      = st.empty()
            val_rmse_hist = []
            best_val      = float('inf')
            best_weights_ft = None

            for ep in range(int(ft_epochs)):
                ft_model.train()
                ep_loss = 0.0
                for bx, by in loader:
                    optimizer.zero_grad()
                    pr, lv, rec, _ = ft_model(bx)
                    loss = _het_loss(pr, by, lv) + 0.05 * mse_loss(rec, bx)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(ft_model.parameters(), 1.0)
                    optimizer.step()
                    ep_loss += loss.item()

                ft_model.eval()
                with torch.no_grad():
                    vp, _, _, _ = ft_model(X_vl_t)
                    vp_np = np.clip(vp.cpu().numpy().flatten(), 0, RUL_CAP)
                val_rmse = float(np.sqrt(mean_squared_error(y_vl, vp_np)))
                scheduler.step(val_rmse)
                val_rmse_hist.append(val_rmse)

                if val_rmse < best_val:
                    best_val = val_rmse
                    best_weights_ft = {k: v.cpu().clone()
                                       for k, v in ft_model.state_dict().items()}

                pct = int((ep + 1) / ft_epochs * 100)
                progress_bar.progress(pct)
                status_text.markdown(
                    f"Epoch **{ep+1}/{ft_epochs}** — "
                    f"Val RMSE: **{val_rmse:.2f}** — "
                    f"Best: **{best_val:.2f}**"
                )

                # Update live chart every 3 epochs
                if (ep + 1) % 3 == 0 or ep == 0:
                    fig_ft, ax_ft = plt.subplots(figsize=(7, 3), dpi=100)
                    ax_ft.plot(range(1, len(val_rmse_hist) + 1), val_rmse_hist,
                               color=PRED_COLOR, linewidth=2)
                    ax_ft.axhline(best_val, color=SAFE_COLOR, linestyle='--',
                                  linewidth=1.2, label=f'Best: {best_val:.2f}')
                    ax_ft.set_xlabel("Epoch")
                    ax_ft.set_ylabel("Val RMSE")
                    ax_ft.set_title("Fine-tuning validation RMSE")
                    ax_ft.legend(fontsize=9)
                    ax_ft.spines['top'].set_visible(False)
                    ax_ft.spines['right'].set_visible(False)
                    plt.tight_layout()
                    chart_ph.pyplot(fig_ft)
                    plt.close(fig_ft)

            # Save as finetuned_core.pt — NEVER touches production_core.pt
            finetune_pkg = {
                'pipeline':         pkg_orig['pipeline'],   # same pipeline
                'ensemble_weights': [best_weights_ft],      # single fine-tuned model
                'base_rmse':        best_val,
                'ft_epochs':        ft_epochs,
                'ft_lr':            ft_lr,
            }
            torch.save(finetune_pkg, 'finetuned_core.pt')

            st.success(
                f"Fine-tuning complete! Best val RMSE: **{best_val:.2f}**.  "
                f"`finetuned_core.pt` saved — select it in Run Inference to use it."
            )
            st.balloons()