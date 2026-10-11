"""Crop recommendation web app (version 3).

New in this version: farm theme, local weather forecast and map (free Open-Meteo API,
no key needed) that can auto-fill temperature, humidity and rainfall, result cards.

Run:  python -m streamlit run app.py
Needs model.pkl and Crop_recommendation.csv in the same folder.
"""
import re

import joblib
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Crop Recommendation", page_icon="🌾", layout="wide")

st.markdown(
    """
    <style>
    .hero {background: linear-gradient(90deg,#2e7d32,#66bb6a); padding:1.6rem 2rem;
           border-radius:16px; margin-bottom:1.2rem;}
    .hero h1 {margin:0; color:white; font-size:2rem;}
    .hero p {margin:.4rem 0 0; color:#e8f5e9; font-size:1.05rem;}
    .step {font-weight:700; font-size:1.25rem; margin:1.4rem 0 .5rem 0;}
    .result {background: linear-gradient(135deg,#2e7d32,#43a047); color:white;
             border-radius:20px; padding:1.6rem; text-align:center; margin-bottom:1rem;}
    .result .big {font-size:3.6rem; line-height:1.1;}
    .result .crop {font-size:2.2rem; font-weight:800;}
    .result .conf {font-size:1.1rem; opacity:.95; margin-top:.2rem;}
    .bar {background: rgba(255,255,255,.3); border-radius:8px; height:10px;
          margin-top:.8rem; overflow:hidden;}
    .bar div {background:#fff; height:100%; border-radius:8px;}
    .tcard {background: rgba(46,125,50,.10); border:1px solid rgba(46,125,50,.35);
            border-radius:16px; padding:1rem; text-align:center;}
    .tcard .em {font-size:2rem;}
    .tcard .nm {font-weight:700; font-size:1.1rem;}
    .tcard .pc {font-size:1.3rem; font-weight:800; color:#43a047;}
    .tcard .bar {background: rgba(46,125,50,.25);}
    .tcard .bar div {background:#43a047;}
    .wcard {background: rgba(30,136,229,.10); border:1px solid rgba(30,136,229,.30);
            border-radius:14px; padding:.6rem .3rem; text-align:center; font-size:.9rem;}
    .wcard .wi {font-size:1.6rem;}
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
    "N": "Nitrogen (N)", "P": "Phosphorus (P)", "K": "Potassium (K)",
    "temperature": "Temperature (°C)", "humidity": "Humidity (%)",
    "ph": "Soil pH", "rainfall": "Rainfall (mm)",
}
ICON_LABELS = {
    "N": "🍃 Nitrogen (N)", "P": "🌼 Phosphorus (P)", "K": "🍂 Potassium (K)",
    "temperature": "🌡️ Temperature (°C)", "humidity": "💧 Humidity (%)",
    "ph": "🧪 Soil pH", "rainfall": "🌧️ Rainfall (mm)",
}
STEP = {"N": 1.0, "P": 1.0, "K": 1.0, "temperature": 0.5, "humidity": 1.0,
        "ph": 0.1, "rainfall": 5.0}
DEFAULTS = {"N": 80.0, "P": 45.0, "K": 40.0, "temperature": 23.0,
            "humidity": 82.0, "ph": 6.5, "rainfall": 230.0}


def clamp(f, v):
    lo, hi = ranges[f]
    return float(min(max(v, lo), hi))


def top_crops(row, n=3):
    probs = model.predict_proba(pd.DataFrame([row])[features])[0]
    ranked = sorted(zip(model.classes_, probs), key=lambda p: p[1], reverse=True)
    return ranked[:n]


# ---------- Weather (Open-Meteo, free, no API key) ----------
@st.cache_data(ttl=1800, show_spinner="Getting weather...")
def fetch_weather(city):
    g = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1, "language": "en", "format": "json"},
        timeout=10,
    )
    g.raise_for_status()
    found = g.json().get("results")
    if not found:
        return None
    p = found[0]

    w = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": p["latitude"], "longitude": p["longitude"],
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
            "hourly": "temperature_2m,relative_humidity_2m",
            "past_days": 30, "forecast_days": 7, "timezone": "auto",
        },
        timeout=15,
    )
    w.raise_for_status()
    data = w.json()
    daily = pd.DataFrame(data["daily"])
    hourly = pd.DataFrame(data["hourly"])
    for col in daily.columns.drop("time"):
        daily[col] = pd.to_numeric(daily[col], errors="coerce")
    for col in hourly.columns.drop("time"):
        hourly[col] = pd.to_numeric(hourly[col], errors="coerce")

    past_daily, future = daily.iloc[:-7], daily.iloc[-7:].fillna(0)
    past_hourly = hourly.iloc[:-7 * 24]
    parts = [p.get("name"), p.get("admin1"), p.get("country")]
    return {
        "place": ", ".join(x for x in parts if x),
        "lat": float(p["latitude"]), "lon": float(p["longitude"]),
        "future": future.reset_index(drop=True),
        "avg_temp": float(past_hourly["temperature_2m"].mean()),
        "avg_humidity": float(past_hourly["relative_humidity_2m"].mean()),
        "rain30": float(past_daily["precipitation_sum"].sum()),
    }


def rain_icon(mm):
    return "☀️" if mm < 1 else ("🌦️" if mm < 5 else "🌧️")


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
        <p>Check your local weather, enter your soil values and find the best crop for your field.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Starting values for the input boxes (kept in session state so weather can fill them)
for f in features:
    key = f"in_{f}"
    if key not in st.session_state:
        st.session_state[key] = clamp(f, DEFAULTS.get(f, sum(ranges[f]) / 2))

# ---------- Step 1: weather forecast and map ----------
st.markdown('<div class="step">📍 Step 1: Check your local weather (optional)</div>',
            unsafe_allow_html=True)
c_city, c_btn = st.columns([4, 1])
city = c_city.text_input(
    "City or town", placeholder="Type your city or town, for example Chennai",
    key="city", label_visibility="collapsed",
)
get_weather = c_btn.button("Get weather")

if get_weather:
    if not city.strip():
        st.warning("Please type a city or town name first.")
    else:
        info, failed = None, False
        try:
            info = fetch_weather(city.strip())
        except Exception:
            failed = True
        if failed:
            st.error("Could not reach the weather service. Please check your internet "
                     "connection, or enter the values by hand below.")
        elif info is None:
            st.warning("I could not find that place. Try a nearby bigger town.")
        else:
            st.session_state["weather"] = info
            st.session_state["in_temperature"] = clamp("temperature", info["avg_temp"])
            st.session_state["in_humidity"] = clamp("humidity", info["avg_humidity"])
            st.session_state["in_rainfall"] = clamp("rainfall", info["rain30"])

w = st.session_state.get("weather")
if w:
    st.success(
        f"Weather for {w['place']}. Temperature, humidity and rainfall below were filled "
        "from the last 30 days. You can change them."
    )
    m1, m2 = st.columns([1, 2])
    with m1:
        st.map(pd.DataFrame({"lat": [w["lat"]], "lon": [w["lon"]]}), zoom=7)
    with m2:
        a, b, c = st.columns(3)
        a.metric("Avg temperature", f"{w['avg_temp']:.1f} °C")
        b.metric("Avg humidity", f"{w['avg_humidity']:.0f} %")
        c.metric("Rainfall (30 days)", f"{w['rain30']:.0f} mm")
        st.caption(
            "These are averages and totals for the last 30 days. The dataset's rainfall "
            "scale may not match exactly, so please check the values before you continue."
        )
    st.markdown("**Next 7 days**")
    fcols = st.columns(7)
    for col, (_, r) in zip(fcols, w["future"].iterrows()):
        day = pd.to_datetime(r["time"]).strftime("%a %d")
        col.markdown(
            f'<div class="wcard"><b>{day}</b><div class="wi">{rain_icon(r["precipitation_sum"])}</div>'
            f'{r["temperature_2m_max"]:.0f}° / {r["temperature_2m_min"]:.0f}°<br>'
            f'<small>{r["precipitation_sum"]:.1f} mm</small></div>',
            unsafe_allow_html=True,
        )

# ---------- Step 2: inputs ----------
st.markdown('<div class="step">🧪 Step 2: Enter your field values</div>', unsafe_allow_html=True)
with st.form("inputs"):
    cols = st.columns(3)
    values = {}
    for i, f in enumerate(features):
        lo, hi = ranges[f]
        with cols[i % 3]:
            values[f] = st.number_input(
                ICON_LABELS.get(f, f),
                min_value=float(lo), max_value=float(hi),
                step=STEP.get(f, 1.0), format="%.1f",
                key=f"in_{f}",
                help=f"Allowed range: {lo:.1f} to {hi:.1f}",
            )
    submitted = st.form_submit_button("🌱 Recommend crop", type="primary")

if submitted:
    st.session_state["values"] = values

vals = st.session_state.get("values")
if vals is None:
    st.info("Enter your values above and click **Recommend crop**.")
    st.stop()

# ---------- Results ----------
st.markdown('<div class="step">🌾 Your result</div>', unsafe_allow_html=True)
tab1, tab2, tab3, tab4 = st.tabs(
    ["🌾 Recommendation", "📊 Why this crop", "🔀 What-if", "💬 Ask assistant"]
)
top = top_crops(vals)

with tab1:
    best_crop, best_p = top[0]
    st.markdown(
        f"""
        <div class="result">
            <div class="big">{emoji(best_crop)}</div>
            <div class="crop">{best_crop.title()}</div>
            <div class="conf">Confidence {best_p * 100:.1f}%</div>
            <div class="bar"><div style="width:{best_p * 100:.1f}%"></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("#### Top 3 crops")
    tcols = st.columns(3)
    for col, (crop, p) in zip(tcols, top):
        col.markdown(
            f"""
            <div class="tcard">
                <div class="em">{emoji(crop)}</div>
                <div class="nm">{crop.title()}</div>
                <div class="pc">{p * 100:.1f}%</div>
                <div class="bar"><div style="width:{p * 100:.1f}%"></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab2:
    if bundle.get("importances"):
        st.markdown("#### Which inputs matter most for this model")
        imp = pd.Series(bundle["importances"]).sort_values(ascending=False)
        imp.index = [LABELS.get(i, i) for i in imp.index]
        st.bar_chart(imp)
        st.caption(f"Most important: {imp.index[0]}. Least important: {imp.index[-1]}.")
    else:
        st.info("Feature importance is not available for this model.")

with tab3:
    st.markdown("#### What if the rainfall changes?")
    delta = st.number_input(
        "Change in rainfall (mm). Use a negative number for less rain.",
        min_value=-200.0, max_value=200.0, value=-100.0, step=10.0,
    )
    changed = dict(vals)
    changed["rainfall"] = clamp("rainfall", vals["rainfall"] + delta)
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
    "agricultural advice. Weather data from Open-Meteo."
)
