import argparse
import io
import os
import tempfile
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

try:
    import gradio as gr
except ImportError:
    raise ImportError("pip install gradio")
try:
    import shap
except ImportError:
    raise ImportError("pip install shap")
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY = True
except ImportError:
    PLOTLY = False

MODEL_PATH = "./models/cybershield-model"
TOXIC_CLASS = 0
SHAP_MAX_EVALS = 150
BATCH_SIZE = 32

SEVERITY = [
    (0.85, "SEVERE",   "#7f1d1d", "🔴"),
    (0.65, "MODERATE", "#c2410c", "🟠"),
    (0.50, "MILD",     "#ca8a04", "🟡"),
    (0.00, "CLEAN",    "#15803d", "🟢"),
]

_tokenizer = None
_model = None
_device = None
_explainer = None


def load_model():
    global _tokenizer, _model, _device, _explainer
    if _tokenizer is not None:
        return
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'. Run train.py first.")
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    _model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH).to(_device)
    _model.eval()

    def _predict_fn(texts):
        inputs = _tokenizer(list(texts), return_tensors="pt",
                            truncation=True, padding=True,
                            max_length=128, return_attention_mask=True
                            ).to(_device)
        with torch.no_grad():
            logits = _model(**inputs).logits
        return torch.softmax(logits.float(), dim=1).cpu().numpy()

    masker = shap.maskers.Text(_tokenizer)
    _explainer = shap.Explainer(_predict_fn, masker,
                                output_names=["BULLYING", "NON-BULLYING"])
    print(f"✅  Model ready on {_device}")

# SEVERITY HELPER


def get_severity(toxic_prob: float):
    for threshold, label, color, icon in SEVERITY:
        if toxic_prob >= threshold:
            return label, color, icon
    return "CLEAN", "#15803d", "🟢"

# BATCH INFERENCE (no SHAP)


def batch_predict(texts: list[str]) -> list[float]:
    load_model()
    probs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        inputs = _tokenizer(batch, return_tensors="pt",
                            truncation=True, padding=True,
                            max_length=128, return_attention_mask=True
                            ).to(_device)
        with torch.no_grad():
            logits = _model(**inputs).logits
        p = torch.softmax(logits.float(), dim=1)[:, TOXIC_CLASS].cpu().numpy()
        probs.extend(p.tolist())
    return probs

# TAB 1 — SINGLE ANALYSE


def analyse_single(text: str):
    if not text or not text.strip():
        empty = "<p style='color:#94a3b8;padding:12px'>Enter a comment above.</p>"
        return empty, [], [], 0.0, ""

    load_model()

    inputs = _tokenizer([text], return_tensors="pt", truncation=True,
                        padding=True, max_length=128,
                        return_attention_mask=True).to(_device)
    with torch.no_grad():
        logits = _model(**inputs).logits
    probs = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
    toxic_prob = float(probs[TOXIC_CLASS])

    sev_label, sev_color, sev_icon = get_severity(toxic_prob)
    conf = toxic_prob if toxic_prob >= 0.5 else float(probs[1])
    pred = "BULLYING" if toxic_prob >= 0.5 else "NON-BULLYING"

    pred_color = "#dc2626" if pred == "BULLYING" else "#16a34a"
    pred_bg = "#fef2f2" if pred == "BULLYING" else "#f0fdf4"
    pred_border = "#fecaca" if pred == "BULLYING" else "#bbf7d0"
    pred_icon = "⚠" if pred == "BULLYING" else "✓"

    label_html = f"""
<div style="padding:16px 20px;background:{pred_bg};border:1.5px solid {pred_border};
            border-radius:12px;font-family:system-ui,sans-serif">
  <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px">
    <div style="display:flex;align-items:center;gap:10px">
      <span style="font-size:26px">{pred_icon}</span>
      <span style="font-size:24px;font-weight:700;color:{pred_color}">{pred}</span>
    </div>
    <div style="background:{sev_color};color:#fff;padding:3px 12px;border-radius:999px;
                font-size:12px;font-weight:600;letter-spacing:.05em">
      {sev_icon} {sev_label}
    </div>
  </div>
  <div style="margin-top:10px;display:flex;gap:20px;font-size:13px;color:#475569;flex-wrap:wrap">
    <span>Confidence&nbsp;<strong style="color:#0f172a">{conf:.1%}</strong></span>
    <span>Bullying prob&nbsp;<strong style="color:{pred_color}">{toxic_prob:.1%}</strong></span>
    <span>Severity&nbsp;<strong style="color:{sev_color}">{sev_label}</strong></span>
  </div>
</div>"""

    # SHAP
    sv = _explainer([text], max_evals=SHAP_MAX_EVALS)
    tokens = list(sv.data[0])
    raw = sv.values[0]
    scores = raw[:, TOXIC_CLASS] if raw.ndim == 2 else raw

    def _bucket(sc):
        if sc > 0.15:
            return "Severe Bullying"
        elif sc > 0.03:
            return "Bullying"
        elif sc < -0.15:
            return "High safe"
        elif sc < -0.03:
            return "Safe"
        return None

    highlights = [(str(t), _bucket(s)) for t, s in zip(tokens, scores)]

    mx = float(np.abs(scores).max()) or 1.0
    top_idx = np.argsort(np.abs(scores))[::-1][:5]
    table = []
    for rank, idx in enumerate(top_idx, 1):
        tok = str(tokens[idx])
        sc = float(scores[idx])
        dirn = "→ BULLYING" if sc > 0 else "→ SAFE"
        bar = "█" * max(1, int(abs(sc) / mx * 12))
        table.append([rank, tok, dirn, f"{sc:+.4f}", bar])

    sev_html = f"""
<div style="padding:12px 16px;background:#f8fafc;border-radius:8px;border:1px solid #e2e8f0;
            font-family:system-ui,sans-serif;font-size:13px">
  <strong>Severity guide:</strong>&nbsp;
  <span style="color:#15803d">🟢 CLEAN &lt;50%</span> &nbsp;
  <span style="color:#ca8a04">🟡 MILD 50–65%</span> &nbsp;
  <span style="color:#c2410c">🟠 MODERATE 65–85%</span> &nbsp;
  <span style="color:#7f1d1d">🔴 SEVERE &gt;85%</span>
</div>"""

    return label_html, highlights, table, toxic_prob, sev_html


def clear_single():
    empty = "<p style='color:#94a3b8;padding:12px'>Prediction will appear here.</p>"
    return "", empty, [], [], 0.0, ""

# TAB 2 — FILE UPLOAD


def analyse_file(file_obj):
    if file_obj is None:
        return None, None, "<p style='color:#94a3b8'>Upload a file first.</p>"

    # Read file
    path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
    ext = os.path.splitext(path)[-1].lower()

    try:
        if ext == ".csv":
            df = pd.read_csv(path)
            # Auto-detect text column
            text_col = next(
                (c for c in ["text", "comment", "sentence", "content", "comment_text"]
                 if c in df.columns), df.columns[0])
            texts = df[text_col].astype(str).tolist()
        else:  # .txt
            with open(path, encoding="utf-8", errors="replace") as f:
                texts = [ln.strip() for ln in f if ln.strip()]
            df = pd.DataFrame({"text": texts})
            text_col = "text"
    except Exception as e:
        return None, None, f"<p style='color:#dc2626'>Error reading file: {e}</p>"

    if len(texts) == 0:
        return None, None, "<p style='color:#dc2626'>No text found in file.</p>"

    # Batch predict
    gr.Info(f"Analysing {len(texts)} comments…")
    toxic_probs = batch_predict(texts)

    # Build results dataframe
    results = []
    for txt, tp in zip(texts, toxic_probs):
        pred = "BULLYING" if tp >= 0.5 else "NON-BULLYING"
        sev, _, _ = get_severity(tp)
        results.append({
            "text":        txt[:120],
            "prediction":  pred,
            "severity":    sev,
            "toxic_prob":  round(tp, 4),
        })

    result_df = pd.DataFrame(results)

    # Save downloadable CSV
    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix="_cybershield_results.csv")
    result_df.to_csv(tmp.name, index=False)

    # Summary HTML
    n = len(results)
    n_tox = sum(1 for r in results if r["prediction"] == "BULLYING")
    n_sev = sum(1 for r in results if r["severity"] == "SEVERE")
    n_mod = sum(1 for r in results if r["severity"] == "MODERATE")
    pct_tox = n_tox / n * 100

    summary_html = f"""
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;
            font-family:system-ui,sans-serif;padding:4px">
  <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;padding:14px;text-align:center">
    <div style="font-size:24px;font-weight:700;color:#15803d">{n - n_tox}</div>
    <div style="font-size:11px;color:#64748b;margin-top:2px">CLEAN</div>
  </div>
  <div style="background:#fefce8;border:1px solid #fde68a;border-radius:10px;padding:14px;text-align:center">
    <div style="font-size:24px;font-weight:700;color:#ca8a04">
      {sum(1 for r in results if r["severity"]=="MILD")}</div>
    <div style="font-size:11px;color:#64748b;margin-top:2px">MILD</div>
  </div>
  <div style="background:#fff7ed;border:1px solid #fed7aa;border-radius:10px;padding:14px;text-align:center">
    <div style="font-size:24px;font-weight:700;color:#c2410c">{n_mod}</div>
    <div style="font-size:11px;color:#64748b;margin-top:2px">MODERATE</div>
  </div>
  <div style="background:#fef2f2;border:1px solid #fecaca;border-radius:10px;padding:14px;text-align:center">
    <div style="font-size:24px;font-weight:700;color:#dc2626">{n_sev}</div>
    <div style="font-size:11px;color:#64748b;margin-top:2px">SEVERE</div>
  </div>
</div>
<div style="margin-top:12px;padding:12px 16px;background:#f8fafc;border-radius:8px;
            font-family:system-ui,sans-serif;font-size:13px;color:#475569">
  <strong>{n}</strong> comments analysed &nbsp;·&nbsp;
  <strong style="color:#dc2626">{n_tox} bullying ({pct_tox:.1f}%)</strong> &nbsp;·&nbsp;
  <strong style="color:#dc2626">{n_sev} severe</strong> flagged
  &nbsp;·&nbsp; Results saved to CSV ↓
</div>"""

    return result_df, tmp.name, summary_html

# TAB 3 — DASHBOARD


def build_dashboard(file_obj):
    if file_obj is None:
        return None, None, None, "<p style='color:#94a3b8'>Upload a file in the File Analysis tab first.</p>"

    path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
    ext = os.path.splitext(path)[-1].lower()

    try:
        if ext == ".csv":
            df = pd.read_csv(path)
            text_col = next(
                (c for c in ["text", "comment", "sentence", "content", "comment_text"]
                 if c in df.columns), df.columns[0])
            texts = df[text_col].astype(str).tolist()
        else:
            with open(path, encoding="utf-8", errors="replace") as f:
                texts = [ln.strip() for ln in f if ln.strip()]
    except Exception as e:
        return None, None, None, f"<p style='color:#dc2626'>Error: {e}</p>"

    toxic_probs = batch_predict(texts)
    sevs = [get_severity(tp)[0] for tp in toxic_probs]

    # Chart 1: Severity donut
    sev_counts = pd.Series(sevs).value_counts()
    sev_colors = {"CLEAN": "#22c55e", "MILD": "#eab308",
                  "MODERATE": "#f97316", "SEVERE": "#ef4444"}
    fig1 = go.Figure(go.Pie(
        labels=sev_counts.index.tolist(),
        values=sev_counts.values.tolist(),
        hole=0.55,
        marker_colors=[sev_colors.get(s, "#94a3b8") for s in sev_counts.index],
        textfont_size=13,
    ))
    fig1.update_layout(
        title="Severity Distribution",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_family="system-ui",
        margin=dict(t=40, b=20, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
        title_font_size=15,
    )

    fig2 = go.Figure(go.Histogram(
        x=toxic_probs,
        nbinsx=20,
        marker_color="#ef4444",
        opacity=0.75,
        name="Bullying prob",
    ))
    fig2.add_vline(x=0.5, line_dash="dash", line_color="#64748b",
                   annotation_text="threshold", annotation_position="top right")
    fig2.update_layout(
        title="Bullying Probability Distribution",
        xaxis_title="Bullying probability",
        yaxis_title="Count",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f8fafc",
        font_family="system-ui",
        margin=dict(t=40, b=40, l=40, r=20),
        title_font_size=15,
        bargap=0.05,
    )

    top_df = pd.DataFrame({"text": texts, "toxic_prob": toxic_probs})
    top_df = top_df.nlargest(15, "toxic_prob")
    top_df["short"] = top_df["text"].str[:50] + "…"
    top_df["color"] = top_df["toxic_prob"].apply(
        lambda p: "#ef4444" if p >= 0.85 else "#f97316" if p >= 0.65 else "#eab308")

    fig3 = go.Figure(go.Bar(
        x=top_df["toxic_prob"],
        y=top_df["short"],
        orientation="h",
        marker_color=top_df["color"].tolist(),
        text=[f"{p:.0%}" for p in top_df["toxic_prob"]],
        textposition="outside",
    ))
    fig3.update_layout(
        title="Top 15 Most Bullying Comments",
        xaxis_title="Bullying probability",
        xaxis_range=[0, 1.15],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f8fafc",
        font_family="system-ui",
        height=max(300, len(top_df)*35),
        margin=dict(t=40, b=40, l=20, r=60),
        yaxis=dict(autorange="reversed"),
        title_font_size=15,
    )

    stats_html = f"""
<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;
            font-family:system-ui,sans-serif">
  <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:16px;text-align:center">
    <div style="font-size:26px;font-weight:700;color:#0f172a">{len(texts)}</div>
    <div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:.05em;margin-top:2px">Total comments</div>
  </div>
  <div style="background:#fef2f2;border:1px solid #fecaca;border-radius:10px;padding:16px;text-align:center">
    <div style="font-size:26px;font-weight:700;color:#dc2626">
      {sum(1 for p in toxic_probs if p>=0.5)}</div>
    <div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:.05em;margin-top:2px">Bullying detected</div>
  </div>
  <div style="background:#fff7ed;border:1px solid #fed7aa;border-radius:10px;padding:16px;text-align:center">
    <div style="font-size:26px;font-weight:700;color:#c2410c">
      {np.mean(toxic_probs):.1%}</div>
    <div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:.05em;margin-top:2px">Avg bullying prob</div>
  </div>
</div>"""

    return fig1, fig2, fig3, stats_html

# EXAMPLES


EXAMPLES = [
    ["You are so stupid, get out"],
    ["Have a wonderful day!"],
    ["I will find you and hurt you badly"],
    ["तुम बेवकूफ हो"],
    ["तुम बहुत अच्छे इंसान हो"],
    ["నువ్వు చాలా వేస్ట్ ఫెలో"],
    ["மீரు చాలా మంచి వాళ్లు"],
    ["நீ முட்டாள்"],
    ["tum bilkul bekar ho yaar"],
    ["you are very helpful, thank you"],
]

COLOUR_MAP = {
    "High bullying": "#ef4444",
    "Bullying":      "#fca5a5",
    "High safe":  "#22c55e",
    "Safe":       "#86efac",
}

# BUILD UI


def build_app():
    with gr.Blocks(
        title="CyberShield XAI",
        theme=gr.themes.Soft(
            primary_hue="red",
            secondary_hue="slate",
            font=[gr.themes.GoogleFont("DM Sans"), "system-ui"],
        ),
        css="""
        footer{display:none!important}
        .tab-nav button{font-size:14px!important;font-weight:600!important}
        """,
    ) as app:

        # ── Header ────────────────────────────────────────────────
        gr.Markdown("""
# 🛡️ CyberShield — Multilingual Bullying Detector

Detects bullying/abusive comments across **English · Hindi · Telugu · Tamil · Kannada · Malayalam · Hinglish**
using **XLM-RoBERTa** + **SHAP explainability** + **Severity scoring**.
        """)

        # ══════════════════════════════════════════════════════════
        with gr.Tabs():

            # ── TAB 1: Single comment ──────────────────────────
            with gr.Tab("💬 Analyse Comment"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3):
                        t1_input = gr.Textbox(
                            label="Comment",
                            placeholder="Type or paste in any supported language…",
                            lines=4, max_lines=8,
                        )
                        with gr.Row():
                            t1_submit = gr.Button(
                                "Analyse", variant="primary", scale=3)
                            t1_clear = gr.Button(
                                "Clear", variant="secondary", scale=1)
                        gr.Examples(examples=EXAMPLES, inputs=t1_input,
                                    label="Examples", examples_per_page=5)

                    with gr.Column(scale=2):
                        t1_label = gr.HTML(
                            "<p style='color:#94a3b8;padding:12px'>"
                            "Prediction will appear here.</p>")
                        t1_slider = gr.Slider(
                            label="Bullying probability", minimum=0, maximum=1,
                            step=0.001, value=0, interactive=False,
                            info="threshold = 0.5")

                gr.Markdown("---\n### 🔍 SHAP Token Explanation")
                gr.Markdown(
                    "*Red tokens push toward BULLYING · Green push toward SAFE · "
                    "Scores sum exactly to the prediction gap (SHAP guarantee)*")
                t1_highlights = gr.HighlightedText(
                    label="Token-level contribution",
                    combine_adjacent=False,
                    show_legend=True,
                    color_map=COLOUR_MAP,
                    value=[],
                )
                t1_sev_html = gr.HTML("")
                gr.Markdown("### 📊 Top Contributing Tokens")
                t1_table = gr.Dataframe(
                    headers=["Rank", "Token", "Direction",
                             "SHAP score", "Magnitude"],
                    datatype=["number", "str", "str", "str", "str"],
                    row_count=(5, "fixed"), col_count=(5, "fixed"),
                    interactive=False, value=[],
                )

                _out1 = [t1_label, t1_highlights,
                         t1_table, t1_slider, t1_sev_html]
                t1_submit.click(fn=analyse_single,
                                inputs=t1_input, outputs=_out1)
                t1_input.submit(fn=analyse_single,
                                inputs=t1_input, outputs=_out1)
                t1_clear.click(fn=clear_single, inputs=None,
                               outputs=[t1_input] + _out1)

            # ── TAB 2: File upload ─────────────────────────────
            with gr.Tab("📂 File Analysis"):
                gr.Markdown(
                    "Upload a **CSV** (must have a `text` / `comment` column) "
                    "or a **TXT** file (one comment per line). "
                    "Results are shown in a table and available to download.")

                t2_file = gr.File(
                    label="Upload CSV or TXT",
                    file_types=[".csv", ".txt"],
                )
                t2_btn = gr.Button("Analyse File", variant="primary")
                t2_summary = gr.HTML("")
                t2_table = gr.Dataframe(
                    headers=["text", "prediction", "severity", "toxic_prob"],
                    datatype=["str", "str", "str", "number"],
                    interactive=False,
                    wrap=True,
                )
                t2_download = gr.File(
                    label="⬇ Download results CSV", visible=True)

                t2_btn.click(
                    fn=analyse_file,
                    inputs=t2_file,
                    outputs=[t2_table, t2_download, t2_summary],
                )

            # ── TAB 3: Dashboard ───────────────────────────────
            with gr.Tab("📊 Dashboard"):
                gr.Markdown(
                    "Upload the **same file** you used in File Analysis "
                    "to generate interactive charts.")

                t3_file = gr.File(
                    label="Upload CSV or TXT",
                    file_types=[".csv", ".txt"],
                )
                t3_btn = gr.Button("Generate Dashboard", variant="primary")
                t3_stats = gr.HTML("")

                with gr.Row():
                    t3_donut = gr.Plot(label="Severity distribution")
                    t3_hist = gr.Plot(label="Probability distribution")
                t3_bar = gr.Plot(label="Top 15 most bullying comments")

                t3_btn.click(
                    fn=build_dashboard,
                    inputs=t3_file,
                    outputs=[t3_donut, t3_hist, t3_bar, t3_stats],
                )

        # ── Footer ────────────────────────────────────────────────
        gr.Markdown(
            "**Model:** XLM-RoBERTa-base  "
            "**XAI:** SHAP PartitionExplainer · "
            "**Severity:** CLEAN / MILD / MODERATE / SEVERE"
        )

    return app


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share",  action="store_true")
    parser.add_argument("--port",   type=int, default=7860)
    parser.add_argument("--server", type=str, default="127.0.0.1")
    args = parser.parse_args()

    print("\n🛡️  CyberShield — Full Web App")
    print(f"   Model  : {MODEL_PATH}")
    print(f"   Device : {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"   Share  : {args.share}\n")

    print("Loading model and SHAP explainer…")
    load_model()
    print("✅  Ready\n")

    app = build_app()
    from pyngrok import ngrok

    public_url = ngrok.connect(args.port)
    print("🌐 Public URL:", public_url)
    app.launch(
        server_name=args.server,
        server_port=args.port,
        share=args.share,
        show_error=True,
        inbrowser=True,
    )
