"""Crop recommendation web app (version 2: number boxes + cleaner layout).

Run:  python -m streamlit run app.py
Needs model.pkl in the same folder.
"""
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Crop Recommendation", page_icon="🌾", layout="wide")

st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(90deg, #2e7d32, #66bb6a);
        padding: 1.6rem 2rem;
        border-radius: 14px;
        margin-bottom: 1rem;
    }
    .hero h1 { margin: 0; color: white; font-size: 2rem; }
    .hero p  { margin: 0.4rem 0 0; color: #e8f5e9; font-size: 1.05rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

EMOJI = {
    "rice": "🌾", "maize": "🌽", "banana": "🍌", "mango": "🥭", "grapes": "🍇",
    "watermelon": "🍉", "muskmelon": "🍈", "apple": "🍎", "orange": "🍊",
    "papaya": "🍈", "coconut": "🥥", "cotton": "☁️", "jute": "🌿", "coffee": "☕",
    "pomegranate": "🍎", "lentil": "🫘", "chickpea": "🫘", "kidneybeans": "🫘",
    "pigeonpeas": "🫛", "mothbeans": "🫘", "mungbean": "🫛", "blackgram": "🫘",
}


def emoji(crop):
    return EMOJI.get(crop, "🌱")


@st.cache_resource
def load_bundle():
    return joblib.load("model.pkl")


bundle = load_bundle()
model = bundle["model"]
features = bundle["features"]
ranges = bundle["ranges"]

LABELS = {
    "N": "Nitrogen (N)",
    "P": "Phosphorus (P)",
    "K": "Potassium (K)",
    "temperature": "Temperature (°C)",
    "humidity": "Humidity (%)",
    "ph": "Soil pH",
    "rainfall": "Rainfall (mm)",
}

# Starting values (a typical rice field) so the first result looks sensible
DEFAULTS = {"N": 80.0, "P": 45.0, "K": 40.0, "temperature": 23.0,
            "humidity": 82.0, "ph": 6.5, "rainfall": 230.0}


def top_crops(row, n=3):
    probs = model.predict_proba(pd.DataFrame([row])[features])[0]
    ranked = sorted(zip(model.classes_, probs), key=lambda p: p[1], reverse=True)
    return ranked[:n]


# ---------- Chatbot helpers (answers come from your dataset, no API key needed) ----------
@st.cache_data
def load_stats():
    try:
        d = pd.read_csv("Crop_recommendation.csv")
    except FileNotFoundError:
        return None, None
    return d.groupby("label")[features].mean().round(1), d[features].std()


DEFINITIONS = {
    "nitrogen": "Nitrogen (N) helps leaves and stems grow. Low nitrogen shows as pale, yellowing leaves.",
    "phosphorus": "Phosphorus (P) supports root growth, flowering and seed formation.",
    "potassium": "Potassium (K) helps plants resist disease and drought and improves fruit quality.",
    "ph": "Soil pH shows how acidic or alkaline the soil is. 7 is neutral, below 7 is acidic, above 7 is alkaline. Most crops like 6 to 7.5.",
    "humidity": "Humidity is the amount of moisture in the air, in percent. Some crops, like rice, prefer high humidity.",
    "rainfall": "Rainfall is the amount of rain in millimetres. It is one of the strongest clues for choosing a crop.",
    "temperature": "Temperature is the average air temperature in degrees Celsius. Each crop grows best in its own range.",
    "npk": "NPK means Nitrogen (N), Phosphorus (P) and Potassium (K), the three main nutrients plants need from soil.",
}
FERTILIZER_HINT = {
    "N": "nitrogen-rich fertilizer or compost (for example urea)",
    "P": "phosphorus fertilizer (for example DAP or bone meal)",
    "K": "potash fertilizer (for example muriate of potash or wood ash)",
}


def crop_conditions(crop, stats):
    row = stats.loc[crop]
    lines = [f"- {LABELS[f]}: about {row[f]:.1f}" for f in features]
    return f"{emoji(crop)} **Typical conditions for {crop.title()}** (average in the dataset):\n\n" + "\n".join(lines)


def explain_why(vals, crop, stats, spread):
    row = stats.loc[crop]
    diff = {f: abs(vals[f] - row[f]) / spread[f] for f in features}
    ordered = sorted(diff, key=diff.get)
    close = ", ".join(f"{LABELS[f]} (yours {vals[f]:.1f}, typical {row[f]:.1f})" for f in ordered[:3])
    far = ", ".join(f"{LABELS[f]} (yours {vals[f]:.1f}, typical {row[f]:.1f})" for f in ordered[-2:])
    return (
        f"{emoji(crop)} **{crop.title()}** was chosen because these values are closest to what "
        f"{crop.title()} usually needs: {close}.\n\nThe values furthest from typical are: {far}."
    )


def answer(question, vals, top):
    import re
    q = question.lower()
    words = set(re.findall(r"[a-z]+", q))
    stats, spread = load_stats()
    best_crop, best_p = top[0]

    if any(w in words for w in ("why", "reason", "explain", "because")):
        if stats is None:
            return "I need Crop_recommendation.csv in the app folder to explain the reason."
        return explain_why(vals, best_crop, stats, spread)

    if words & {"confidence", "sure", "certain", "confident"}:
        second, p2 = top[1]
        text = (f"The model is {best_p * 100:.1f}% sure about {best_crop.title()}. "
                f"The next choice is {second.title()} at {p2 * 100:.1f}%.")
        if best_p < 0.6:
            text += " The confidence is low because your values sit between several crops."
        return text

    if words & {"fertilizer", "improve", "nutrient", "nutrients", "soil", "increase", "add"}:
        if stats is None:
            return "I need Crop_recommendation.csv in the app folder to compare nutrients."
        row = stats.loc[best_crop]
        tips = []
        for f in ("N", "P", "K"):
            if vals[f] < 0.8 * row[f]:
                tips.append(f"- {LABELS[f]} is low (yours {vals[f]:.0f}, typical {row[f]:.0f}). Consider {FERTILIZER_HINT[f]}.")
            elif vals[f] > 1.25 * row[f]:
                tips.append(f"- {LABELS[f]} is high (yours {vals[f]:.0f}, typical {row[f]:.0f}). Avoid adding more of it.")
        if not tips:
            tips.append(f"- Your N, P and K are in a normal range for {best_crop.title()}.")
        return ("For " + best_crop.title() + ":\n\n" + "\n".join(tips) +
                "\n\nThese are general hints from the dataset. Please confirm with a soil test or a local agriculture officer.")

    if words & {"alternative", "alternatives", "other", "second", "options", "else"}:
        return "Top choices for your values:\n\n" + "\n".join(
            f"- {emoji(c)} {c.title()}: {p * 100:.1f}%" for c, p in top)

    if stats is not None:
        for crop in stats.index:
            if crop in q:
                return crop_conditions(crop, stats)

    for key, text in DEFINITIONS.items():
        if key in words:
            return text

    if words & {"model", "algorithm", "data", "dataset", "trained"}:
        return (f"The recommendation uses a {bundle['name']} model trained on a public crop dataset "
                "with 22 crops and 7 inputs: N, P, K, temperature, humidity, pH and rainfall.")

    return ("I can help with: why this crop, how sure the model is, other options, soil and fertilizer hints, "
            "what a crop needs (for example 'what does rice need?'), and what N, P, K, pH or rainfall mean.")


# ---------- Header ----------
st.markdown(
    """
    <div class="hero">
        <h1>🌾 Crop Recommendation System</h1>
        <p>Type your soil and weather values and get the best crop for your field.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Inputs: number boxes ----------
with st.form("inputs"):
    st.subheader("Enter your field values")
    cols = st.columns(3)
    values = {}
    for i, f in enumerate(features):
        lo, hi = ranges[f]
        default = min(max(DEFAULTS.get(f, (lo + hi) / 2), lo), hi)
        with cols[i % 3]:
            values[f] = st.number_input(
                LABELS.get(f, f),
                min_value=float(lo),
                max_value=float(hi),
                value=float(default),
                step=1.0 if f != "ph" else 0.1,
                help=f"Allowed range: {lo:.1f} to {hi:.1f}",
            )
    submitted = st.form_submit_button("Recommend crop", type="primary")

if submitted:
    st.session_state["values"] = values

vals = st.session_state.get("values")
if vals is None:
    st.info("Enter your values above and click **Recommend crop**.")
    st.stop()

# ---------- Results ----------
tab1, tab2, tab3, tab4 = st.tabs(["Recommendation", "Why this crop", "What-if", "Ask assistant"])
top = top_crops(vals)

with tab1:
    best_crop, best_p = top[0]
    with st.container(border=True):
        st.markdown(f"### {emoji(best_crop)} Best crop: {best_crop.title()}")
        st.metric("Confidence", f"{best_p * 100:.1f}%")
    st.markdown("#### Top 3 recommendations")
    for crop, p in top:
        st.write(f"{emoji(crop)} **{crop.title()}** : {p * 100:.1f}%")
        st.progress(float(p))

with tab2:
    if bundle.get("importances"):
        st.markdown("#### Which inputs matter most for this model")
        imp = pd.Series(bundle["importances"]).sort_values(ascending=False)
        imp.index = [LABELS.get(i, i) for i in imp.index]
        st.bar_chart(imp)
        st.caption(
            f"Most important: {imp.index[0]}. Least important: {imp.index[-1]}."
        )
    else:
        st.info("Feature importance is not available for this model.")

with tab3:
    st.markdown("#### What if the rainfall changes?")
    delta = st.number_input(
        "Change in rainfall (mm). Use a negative number for less rain.",
        min_value=-200.0, max_value=200.0, value=-100.0, step=10.0,
    )
    changed = dict(vals)
    lo, hi = ranges["rainfall"]
    changed["rainfall"] = min(max(vals["rainfall"] + delta, lo), hi)
    new_crop, new_p = top_crops(changed, n=1)[0]
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.caption(f"Now: rainfall {vals['rainfall']:.0f} mm")
            st.markdown(f"### {emoji(top[0][0])} {top[0][0].title()}")
            st.metric("Confidence", f"{top[0][1] * 100:.1f}%")
    with c2:
        with st.container(border=True):
            st.caption(f"After change: rainfall {changed['rainfall']:.0f} mm")
            st.markdown(f"### {emoji(new_crop)} {new_crop.title()}")
            st.metric("Confidence", f"{new_p * 100:.1f}%")

with tab4:
    st.markdown("#### Ask about your result or about crops")
    st.caption("Try: why this crop? | what does rice need? | what is potassium? | how can I improve my soil?")
    if "chat" not in st.session_state:
        st.session_state["chat"] = []
    for role, text in st.session_state["chat"]:
        with st.chat_message(role):
            st.markdown(text)
    question = st.chat_input("Ask a question")
    if question:
        st.session_state["chat"].append(("user", question))
        st.session_state["chat"].append(("assistant", answer(question, vals, top)))
        st.rerun()

st.caption(
    "This is a decision aid based on a public dataset, not a replacement for expert "
    "agricultural advice."
)
