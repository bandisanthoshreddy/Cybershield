"""
CyberShield XAI Layer — SHAP-based Explainability
===================================================
Explains WHY the model flagged a comment as toxic or non-toxic
by computing per-token SHAP importance scores.

Install:
    pip install shap

Usage:
    python xai.py                          # runs built-in test cases
    python xai.py --text "your text here"  # single custom input
    python xai.py --report                 # also saves xai_report.html
"""

import argparse
import os
import sys
import html
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

try:
    import shap
except ImportError:
    print("❌  SHAP not installed.  Run:  pip install shap")
    sys.exit(1)

# CONFIG

MODEL_PATH = "./models/cybershield-model"
REPORT_PATH = "./xai_report.html"
TOXIC_THRESHOLD = 0.5
TOXIC_CLASS = 0
SHAP_MAX_EVALS = 200

# LOAD MODEL


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'. Run train.py first.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH).to(device)
    model.eval()
    print(f"✅  Model loaded  |  device: {device}\n")
    return tokenizer, model, device

# PREDICT (SHAP-compatible)


def make_predict_fn(tokenizer, model, device):
    def predict(texts):
        if isinstance(texts, str):
            texts = [texts]
        inputs = tokenizer(
            list(texts),
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
            return_attention_mask=True,
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
        return probs
    return predict


def predict_single(text, predict_fn):
    probs = predict_fn([text])[0]
    toxic_prob = probs[TOXIC_CLASS]
    if toxic_prob > TOXIC_THRESHOLD:
        label, confidence = "BULLYING",     float(toxic_prob)
    else:
        label, confidence = "NON-BULLYING", float(probs[1])
    return label, confidence, float(toxic_prob)

# SHAP EXPLAINER


def build_explainer(predict_fn, tokenizer):
    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(
        predict_fn, masker,
        output_names=["BULLYING", "NON-BULLYING"],)
    return explainer


def explain(text, explainer):
    sv = explainer([text], max_evals=SHAP_MAX_EVALS)
    tokens = sv.data[0]
    scores = sv.values[0][:, TOXIC_CLASS]
    return tokens, scores

# CONSOLE OUTPUT


RESET = "\033[0m"
RBOLD = "\033[1;31m"
RED = "\033[31m"
GBOLD = "\033[1;32m"
GRN = "\033[32m"
GREY = "\033[90m"
CYAN = "\033[36m"
YEL = "\033[33m"


def console_explain(text, tokens, scores, label, confidence, toxic_prob):
    icon = "🔴" if label == "BULLYING" else "🟢"
    print("─" * 65)
    print(f"  {CYAN}Input{RESET}      : {text}")
    print(f"  {CYAN}Prediction{RESET} : {icon} {label}")
    print(f"  {CYAN}Confidence{RESET} : {confidence:.4f}")
    print(f"  {CYAN}Bullying Prob{RESET} : {toxic_prob:.4f}")
    print()

    if len(scores) == 0:
        print("  (no tokens)")
        print("─" * 65)
        return

    mx = float(np.abs(scores).max()) or 1.0
    parts = []
    for tok, sc in zip(tokens, scores):
        intensity = abs(sc) / mx
        if sc > 0.005:
            c = RBOLD if intensity > 0.5 else RED
        elif sc < -0.005:
            c = GBOLD if intensity > 0.5 else GRN
        else:
            c = GREY
        parts.append(f"{c}{tok}{RESET}")

    print("  Token map  (🔴 = pushes BULLYING | 🟢 = pushes SAFE):")
    print("  " + " ".join(parts))
    print()

    top = np.argsort(np.abs(scores))[::-1][:5]
    print("  Top contributors:")
    for rank, idx in enumerate(top, 1):
        tok = tokens[idx]
        sc = scores[idx]
        arrow = f"{RED}→ BULLYING{RESET}" if sc > 0 else f"{GRN}→ SAFE{RESET}"
        bar = "█" * int(abs(sc)/mx*12)
        print(f"    {rank}. {YEL}'{tok}'{RESET}  {arrow}  {sc:+.4f}  {bar}")
    print("─" * 65)

# HTML REPORT


def _tok_span(token, score, mx):
    intensity = min(abs(score)/(mx or 1.0), 1.0)
    alpha = 0.15 + 0.65*intensity
    safe = html.escape(str(token))
    if score > 0.005:
        bg, border = f"rgba(239,68,68,{alpha:.2f})", "rgba(220,38,38,.6)"
    elif score < -0.005:
        bg, border = f"rgba(34,197,94,{alpha:.2f})", "rgba(22,163,74,.6)"
    else:
        bg, border = "rgba(148,163,184,.15)", "transparent"
    return f'<span class="tok" style="background:{bg};border:1px solid {border}">{safe}</span>'


def build_html_report(results):
    cards = ""
    for r in results:
        lbl, conf, tp = r["label"], r["confidence"], r["toxic_prob"]
        tokens, scores = r["tokens"], r["scores"]
        tx = html.escape(r["text"])
        mx = float(np.abs(scores).max()) if len(scores) else 1.0
        bc = "#ef4444" if lbl == "BULLYING" else "#22c55e"
        ico = "⚠" if lbl == "BULLYING" else "✓"
        tok_html = "".join(_tok_span(t, s, mx) for t, s in zip(tokens, scores))

        top_rows = ""
        for rank, idx in enumerate(np.argsort(np.abs(scores))[::-1][:5], 1):
            tok = html.escape(str(tokens[idx]))
            sc = scores[idx]
            col = "#ef4444" if sc > 0 else "#22c55e"
            dir_ = "→ BULLYING" if sc > 0 else "→ SAFE"
            bw = int(abs(sc)/mx*100)
            top_rows += f"""<tr>
              <td class="rk">{rank}</td><td><code>{tok}</code></td>
              <td style="color:{col};font-weight:600">{dir_}</td>
              <td>{sc:+.4f}</td>
              <td><div class="bar" style="width:{bw}%;background:{col}"></div></td>
            </tr>"""

        cls = "card-toxic" if lbl == "BULLYING" else "card-safe"
        cards += f"""
        <div class="card {cls}">
          <div class="card-header">
            <span class="itext">"{tx}"</span>
            <span class="badge" style="background:{bc}">{ico} {lbl}</span>
          </div>
          <div class="card-meta">Confidence: <strong>{conf:.4f}</strong> &nbsp;·&nbsp; Bullying Prob: <strong>{tp:.4f}</strong></div>
          <div class="slabel">Token Explanation</div>
          <div class="tok-row">{tok_html}</div>
          <div class="legend">
            <span class="li lt-hi">High bullying</span>
            <span class="li lt-lo">Low bullying</span>
            <span class="li ls">Safe</span>
            <span class="li ln">Neutral</span>
          </div>
          <div class="slabel">Top Contributing Tokens</div>
          <table class="ttab"><thead><tr>
            <th>#</th><th>Token</th><th>Direction</th><th>Score</th><th>Magnitude</th>
          </tr></thead><tbody>{top_rows}</tbody></table>
        </div>"""

    n_toxic = sum(1 for r in results if r["label"] == "BULLYING")

    return f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CyberShield XAI Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
:root{{--bg:#0a0e1a;--sur:#111827;--sur2:#1a2235;--bdr:#1e2d45;
       --txt:#e2e8f0;--mut:#64748b;--tox:#ef4444;--saf:#22c55e;--acc:#38bdf8;
       --mono:'Space Mono',monospace;--sans:'DM Sans',sans-serif}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--txt);font-family:var(--sans);font-size:15px;line-height:1.6;min-height:100vh}}
.hdr{{background:linear-gradient(135deg,#0f172a 0%,#1e1b4b 50%,#0f172a 100%);
      border-bottom:1px solid var(--bdr);padding:48px 40px 40px;position:relative;overflow:hidden}}
.hdr::before{{content:'';position:absolute;inset:0;
  background:radial-gradient(ellipse 60% 80% at 70% 50%,rgba(56,189,248,.08) 0%,transparent 70%)}}
.hi{{position:relative;max-width:900px;margin:0 auto}}
.htag{{font-family:var(--mono);font-size:11px;letter-spacing:.15em;color:var(--acc);
       text-transform:uppercase;margin-bottom:12px}}
h1{{font-family:var(--mono);font-size:clamp(28px,4vw,42px);font-weight:700;
    letter-spacing:-.02em;color:#fff;line-height:1.1;margin-bottom:10px}}
h1 span{{color:var(--acc)}}
.sub{{color:var(--mut);font-size:14px;max-width:540px}}
.stats{{display:flex;gap:28px;margin-top:28px;flex-wrap:wrap}}
.stat{{display:flex;flex-direction:column;gap:2px}}
.sv{{font-family:var(--mono);font-size:22px;font-weight:700;color:var(--acc)}}
.sl{{font-size:11px;color:var(--mut);letter-spacing:.08em;text-transform:uppercase}}
.main{{max-width:900px;margin:0 auto;padding:40px}}
.why{{background:var(--sur);border:1px solid var(--bdr);border-left:3px solid var(--acc);
      border-radius:12px;padding:24px 28px;margin-bottom:36px}}
.why h2{{font-family:var(--mono);font-size:13px;letter-spacing:.1em;color:var(--acc);
         text-transform:uppercase;margin-bottom:12px}}
.why p{{color:#94a3b8;font-size:13.5px;line-height:1.7;margin-bottom:8px}}
.why p:last-child{{margin-bottom:0}}
.why strong{{color:var(--txt)}}
.card{{background:var(--sur);border:1px solid var(--bdr);border-radius:14px;
       margin-bottom:28px;overflow:hidden}}
.card:hover{{border-color:#2d3f5a}}
.card-toxic{{border-top:3px solid var(--tox)}}
.card-safe{{border-top:3px solid var(--saf)}}
.card-header{{display:flex;align-items:flex-start;justify-content:space-between;
              gap:16px;padding:20px 24px 14px}}
.itext{{font-size:15px;color:var(--txt);flex:1}}
.badge{{flex-shrink:0;font-family:var(--mono);font-size:11px;font-weight:700;
        letter-spacing:.08em;color:#fff;padding:4px 12px;border-radius:999px;white-space:nowrap}}
.card-meta{{padding:0 24px 16px;font-size:12.5px;color:var(--mut);font-family:var(--mono)}}
.slabel{{font-family:var(--mono);font-size:10px;letter-spacing:.12em;text-transform:uppercase;
         color:var(--mut);padding:12px 24px 8px;border-top:1px solid var(--bdr)}}
.tok-row{{padding:10px 20px 16px;line-height:2.4;font-family:var(--mono);font-size:13px}}
.tok{{display:inline-block;padding:1px 5px;margin:2px;border-radius:4px;cursor:default;
      transition:transform .1s}}
.tok:hover{{transform:translateY(-1px)}}
.legend{{display:flex;gap:10px;padding:0 24px 16px;flex-wrap:wrap}}
.li{{font-size:11px;padding:2px 10px;border-radius:999px;font-family:var(--mono)}}
.lt-hi{{background:rgba(239,68,68,.7);color:#fff}}
.lt-lo{{background:rgba(239,68,68,.25);color:#fca5a5}}
.ls{{background:rgba(34,197,94,.4);color:#86efac}}
.ln{{background:rgba(148,163,184,.15);color:var(--mut)}}
.ttab{{width:100%;border-collapse:collapse;font-size:13px;margin-bottom:4px}}
.ttab th{{padding:8px 16px;text-align:left;font-family:var(--mono);font-size:10px;
          letter-spacing:.1em;text-transform:uppercase;color:var(--mut);
          border-top:1px solid var(--bdr);background:var(--sur2)}}
.ttab td{{padding:9px 16px;border-top:1px solid var(--bdr);color:var(--txt)}}
.ttab code{{font-family:var(--mono);background:var(--sur2);padding:1px 6px;
            border-radius:4px;font-size:12px}}
.rk{{color:var(--mut);font-family:var(--mono);font-size:12px;width:30px}}
.bar{{height:6px;border-radius:3px;min-width:4px}}
footer{{text-align:center;padding:32px;color:var(--mut);font-size:12px;
        font-family:var(--mono);border-top:1px solid var(--bdr);margin-top:20px}}
</style></head><body>
<header class="hdr"><div class="hi">
  <div class="htag">CyberShield · XAI Layer · SHAP Explainability</div>
  <h1>Token-Level <span>BULLYING</span> Explanations</h1>
  <p class="sub">Each prediction is explained using SHAP (SHapley Additive exPlanations),
  showing exactly which tokens pushed the model toward BULLYING or NON-BULLYING.</p>
  <div class="stats">
    <div class="stat"><div class="sv">{len(results)}</div><div class="sl">Inputs analysed</div></div>
    <div class="stat"><div class="sv">{n_toxic}</div><div class="sl">Bullying detected</div></div>
    <div class="stat"><div class="sv">SHAP</div><div class="sl">Method used</div></div>
    <div class="stat"><div class="sv">XLM-R</div><div class="sl">Base model</div></div>
  </div>
</div></header>
<main class="main">
  <div class="why">
    <h2>Why SHAP?</h2>
    <p><strong>Attention weights</strong> show which tokens the model "looked at" — not which tokens
    <em>caused</em> the prediction. Research (Jain &amp; Wallace 2019) showed attention and prediction
    importance are largely uncorrelated.</p>
    <p><strong>LIME</strong> approximates the model locally by randomly masking words. Fast, but
    unstable — run it twice on the same input and you can get different explanations.</p>
    <p><strong>SHAP</strong> is grounded in cooperative game theory. It treats each token as a "player"
    and distributes prediction credit fairly using Shapley values — guaranteed to be
    <strong>efficient</strong> (scores sum to the actual prediction gap),
    <strong>symmetric</strong> (equal contributors get equal scores), and
    <strong>stable</strong> (same input always produces the same explanation). For multilingual
    social-media text with subword tokenisation, SHAP's mask-based approach is semantically
    correct and model-agnostic — no gradient access needed.</p>
  </div>
  {cards}
</main>
<footer>CyberShield · Multilingual BULLYING Detection · XAI Layer · SHAP {shap.__version__}</footer>
</body></html>"""

# MAIN


def main():
    parser = argparse.ArgumentParser(description="CyberShield XAI")
    parser.add_argument("--text",   type=str, default=None)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    tokenizer, model, device = load_model()
    predict_fn = make_predict_fn(tokenizer, model, device)

    print("Building SHAP explainer (first run compiles masker ~10s)...")
    explainer = build_explainer(predict_fn, tokenizer)
    print("✅  Explainer ready\n")

    test_inputs = [args.text] if args.text else [
        "You are so stupid, get out of here",
        "Have a wonderful day, take care!",
        "I will find you and hurt you badly",
        "तुम बेवकूफ हो",
        "तुम बहुत अच्छे इंसान हो",
        "నువ్వు చాలా వేస్ట్ ఫెలో",
        "మీరు చాలా మంచి వాళ్లు",
        "நீ முட்டாள்",
        "tum bilkul bekar ho yaar",
        "you are very helpful, thank you",
    ]

    results = []
    print("=" * 65)
    print("  CyberShield — XAI Explanations")
    print("=" * 65 + "\n")

    for text in test_inputs:
        label, conf, tp = predict_single(text, predict_fn)
        print(f"Computing SHAP: '{text[:55]}{'...' if len(text)>55 else ''}'")
        tokens, scores = explain(text, explainer)
        console_explain(text, tokens, scores, label, conf, tp)
        results.append(dict(text=text, label=label, confidence=conf,
                            toxic_prob=tp, tokens=tokens, scores=scores))

    n_toxic = sum(1 for r in results if r["label"] == "BULLYING")
    print(f"\n{'─'*65}")
    print(f"  Summary: {n_toxic}/{len(results)} inputs flagged BULLYING")
    print(f"{'─'*65}\n")

    if args.report or not args.text:
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(build_html_report(results))
        print(f"✅  HTML report saved → {REPORT_PATH}")
        print("    Open in any browser for colour-coded token explanations.\n")


if __name__ == "__main__":
    main()
