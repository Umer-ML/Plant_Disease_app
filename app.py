import streamlit as st
import numpy as np
import json
from PIL import Image
import tensorflow as tf
try:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="LeafScan AI — Plant Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Premium CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg: #050A0A;
  --surface: #0C1414;
  --surface2: #112020;
  --green: #00FF88;
  --green2: #00CC6A;
  --teal: #00E5CC;
  --amber: #FFB800;
  --red: #FF4455;
  --text: #F0FAF5;
  --muted: #6B9E8A;
  --border: rgba(0,255,136,0.12);
  --border2: rgba(0,255,136,0.25);
}

/* ── Hide Streamlit UI ── */
#MainMenu, footer, header, .stDeployButton { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] { display: none; }

/* ── Body ── */
html, body, .stApp {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'DM Sans', sans-serif !important;
}

/* ── Animated Grid Background ── */
.stApp::before {
  content: '';
  position: fixed; inset: 0; z-index: 0;
  background-image:
    linear-gradient(rgba(0,255,136,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,255,136,0.03) 1px, transparent 1px);
  background-size: 40px 40px;
  animation: gridMove 20s linear infinite;
  pointer-events: none;
}
@keyframes gridMove {
  0% { background-position: 0 0; }
  100% { background-position: 40px 40px; }
}

/* ── Navbar ── */
.navbar {
  position: sticky; top: 0; z-index: 100;
  display: flex; align-items: center; justify-content: space-between;
  padding: 16px 48px;
  background: rgba(5,10,10,0.85);
  backdrop-filter: blur(24px);
  border-bottom: 1px solid var(--border);
}
.nav-logo {
  font-family: 'Syne', sans-serif; font-weight: 800; font-size: 20px;
  background: linear-gradient(135deg, var(--green), var(--teal));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  letter-spacing: -0.02em;
}
.nav-right { display: flex; align-items: center; gap: 16px; }
.nav-tag {
  font-size: 11px; color: var(--muted); letter-spacing: 0.15em;
  text-transform: uppercase; padding: 5px 14px;
  border: 1px solid var(--border); border-radius: 20px;
}
.nav-dot {
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--green); box-shadow: 0 0 12px var(--green);
  animation: pulse 2s ease infinite;
}
@keyframes pulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.5; transform: scale(0.8); }
}

/* ── Hero ── */
.hero {
  padding: 100px 48px 80px;
  text-align: center; position: relative;
}
.hero-eyebrow {
  font-size: 11px; letter-spacing: 0.25em; text-transform: uppercase;
  color: var(--green); margin-bottom: 28px;
  display: flex; align-items: center; justify-content: center; gap: 16px;
  animation: fadeUp 0.8s ease both;
}
.eyebrow-line { width: 48px; height: 1px; background: var(--green); opacity: 0.4; }
.hero h1 {
  font-family: 'Syne', sans-serif; font-weight: 800;
  font-size: clamp(48px, 8vw, 96px); line-height: 0.92;
  letter-spacing: -0.04em; margin-bottom: 28px;
  animation: fadeUp 0.8s 0.1s ease both;
}
.hero-line1 { display: block; color: var(--text); }
.hero-line2 {
  display: block;
  background: linear-gradient(135deg, var(--green) 0%, var(--teal) 50%, var(--green) 100%);
  background-size: 200% auto;
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  animation: shimmer 3s linear infinite;
}
@keyframes shimmer {
  0% { background-position: 0% center; }
  100% { background-position: 200% center; }
}
.hero-sub {
  font-size: 17px; color: var(--muted); max-width: 480px;
  margin: 0 auto 56px; line-height: 1.7; font-weight: 300;
  animation: fadeUp 0.8s 0.2s ease both;
}
.stats-row {
  display: flex; justify-content: center; gap: 64px;
  animation: fadeUp 0.8s 0.3s ease both;
}
.stat { text-align: center; }
.stat-num {
  font-family: 'Syne', sans-serif; font-weight: 800; font-size: 36px;
  background: linear-gradient(135deg, var(--green), var(--teal));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.stat-label { font-size: 12px; color: var(--muted); letter-spacing: 0.1em; margin-top: 4px; }

/* ── Upload Section ── */
.upload-wrap { padding: 0 48px 80px; max-width: 960px; margin: 0 auto; }
.section-label {
  font-size: 11px; letter-spacing: 0.2em; text-transform: uppercase;
  color: var(--green); margin-bottom: 28px;
  display: flex; align-items: center; gap: 16px;
}
.section-label::after { content: ''; flex: 1; height: 1px; background: var(--border); }

/* Streamlit file uploader override */
[data-testid="stFileUploader"] {
  background: var(--surface) !important;
  border: 1.5px dashed rgba(0,255,136,0.25) !important;
  border-radius: 20px !important;
  padding: 48px !important;
  transition: all 0.3s ease !important;
}
[data-testid="stFileUploader"]:hover {
  border-color: rgba(0,255,136,0.6) !important;
  background: var(--surface2) !important;
}
[data-testid="stFileUploader"] label {
  color: var(--text) !important;
  font-family: 'Syne', sans-serif !important;
  font-size: 18px !important;
  font-weight: 600 !important;
}
[data-testid="stFileUploaderDropzone"] {
  background: transparent !important;
  border: none !important;
}
[data-testid="baseButton-secondary"] {
  background: rgba(0,255,136,0.1) !important;
  color: var(--green) !important;
  border: 1px solid var(--border2) !important;
  border-radius: 10px !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 600 !important;
  transition: all 0.3s !important;
}
[data-testid="baseButton-secondary"]:hover {
  background: rgba(0,255,136,0.2) !important;
  transform: translateY(-2px) !important;
}

/* ── Result Cards ── */
.res-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 20px; padding: 28px;
  position: relative; overflow: hidden;
  animation: fadeUp 0.6s ease both;
}
.res-card::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
  background: linear-gradient(90deg, var(--green), var(--teal));
}

.res-label {
  font-size: 10px; letter-spacing: 0.18em; text-transform: uppercase;
  color: var(--muted); margin-bottom: 8px;
}
.res-value {
  font-family: 'Syne', sans-serif; font-weight: 700;
  font-size: 22px; color: var(--text); line-height: 1.2;
}

.badge-healthy {
  display: inline-flex; align-items: center; gap: 8px;
  padding: 10px 20px; border-radius: 30px; font-size: 14px; font-weight: 500;
  background: rgba(0,255,136,0.1); border: 1px solid rgba(0,255,136,0.3);
  color: var(--green); font-family: 'Syne', sans-serif;
}
.badge-diseased {
  display: inline-flex; align-items: center; gap: 8px;
  padding: 10px 20px; border-radius: 30px; font-size: 14px; font-weight: 500;
  background: rgba(255,68,85,0.1); border: 1px solid rgba(255,68,85,0.3);
  color: var(--red); font-family: 'Syne', sans-serif;
}

.conf-big {
  font-family: 'Syne', sans-serif; font-weight: 800; font-size: 52px;
  background: linear-gradient(135deg, var(--green), var(--teal));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  line-height: 1;
}

/* Progress bars */
[data-testid="stProgress"] > div > div {
  background: linear-gradient(90deg, var(--green), var(--teal)) !important;
  border-radius: 4px !important;
  box-shadow: 0 0 8px rgba(0,255,136,0.3) !important;
}
[data-testid="stProgress"] > div {
  background: rgba(255,255,255,0.05) !important;
  border-radius: 4px !important;
}

/* Desc card */
.desc-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 20px; padding: 28px; position: relative;
  border-left: 3px solid var(--amber);
  animation: fadeUp 0.6s 0.1s ease both;
}
.desc-text { font-size: 15px; color: var(--muted); line-height: 1.75; font-weight: 300; }

/* Image display */
[data-testid="stImage"] img {
  border-radius: 16px !important;
  border: 1px solid var(--border) !important;
}

/* Metrics */
[data-testid="stMetric"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 16px !important; padding: 20px !important;
}
[data-testid="stMetricLabel"] {
  font-size: 10px !important; letter-spacing: 0.15em !important;
  text-transform: uppercase !important; color: var(--muted) !important;
}
[data-testid="stMetricValue"] {
  font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
  color: var(--text) !important;
}

/* Divider */
hr { border-color: var(--border) !important; margin: 40px 0 !important; }

/* Button */
[data-testid="baseButton-primary"] {
  background: linear-gradient(135deg, var(--green), var(--teal)) !important;
  color: #050A0A !important; border: none !important;
  border-radius: 14px !important; font-family: 'Syne', sans-serif !important;
  font-weight: 700 !important; font-size: 15px !important;
  padding: 14px 32px !important; width: 100% !important;
  transition: all 0.3s ease !important;
  box-shadow: 0 8px 32px rgba(0,255,136,0.15) !important;
}
[data-testid="baseButton-primary"]:hover {
  transform: translateY(-3px) !important;
  box-shadow: 0 20px 40px rgba(0,255,136,0.3) !important;
}

/* Spinner */
[data-testid="stSpinner"] { color: var(--green) !important; }

/* Footer */
.footer-wrap {
  text-align: center; padding: 40px;
  font-size: 12px; color: var(--muted); letter-spacing: 0.08em;
  border-top: 1px solid var(--border); margin-top: 40px;
}

@keyframes fadeUp {
  from { opacity: 0; transform: translateY(24px); }
  to { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# ── Navbar ────────────────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
  <div class="nav-logo">LeafScan AI</div>
  <div class="nav-right">
    <div class="nav-tag">MobileNetV2 · 95.53%</div>
    <div class="nav-dot"></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">
    <span class="eyebrow-line"></span>
    AI-Powered Plant Diagnostics
    <span class="eyebrow-line"></span>
  </div>
  <h1>
    <span class="hero-line1">Detect Plant</span>
    <span class="hero-line2">Diseases Instantly</span>
  </h1>
  <p class="hero-sub">
    Upload any leaf image. Our deep learning model diagnoses
    38 plant diseases across 14 species in under a second.
  </p>
  <div class="stats-row">
    <div class="stat">
      <div class="stat-num">95.5%</div>
      <div class="stat-label">Accuracy</div>
    </div>
    <div class="stat">
      <div class="stat-num">38</div>
      <div class="stat-label">Disease Classes</div>
    </div>
    <div class="stat">
      <div class="stat-num">87K</div>
      <div class="stat-label">Training Images</div>
    </div>
    <div class="stat">
      <div class="stat-num">14</div>
      <div class="stat-label">Plant Species</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Load Model & Classes ──────────────────────────────────────────────
# @st.cache_resource
# def load_model():
#     interpreter = tf.lite.Interpreter(model_path="plant_disease.tflite")
#     interpreter.allocate_tensors()
#     return interpreter
@st.cache_resource
def load_model():
    interpreter = Interpreter(model_path="plant_disease.tflite")
    interpreter.allocate_tensors()
    return interpreter

@st.cache_resource
def load_classes():
    with open("class_names.json") as f:
        return json.load(f)

interpreter = load_model()
class_names = load_classes()

# ── Disease Info ──────────────────────────────────────────────────────
DISEASE_INFO = {
    "Apple___Apple_scab": ("Fungal disease causing dark, scabby lesions on leaves and fruit surface.", "Apply fungicide early spring. Remove fallen leaves. Prune for air circulation."),
    "Apple___Black_rot": ("Fungal infection causing black rotting spots on fruit and leaves.", "Remove infected fruit and branches. Apply copper-based fungicide."),
    "Apple___Cedar_apple_rust": ("Fungal disease causing bright orange spots on leaves.", "Apply preventive fungicide in spring. Remove nearby cedar trees if possible."),
    "Apple___healthy": ("Your apple plant is perfectly healthy! Great job.", "Continue regular monitoring, proper watering, and seasonal pruning."),
    "Blueberry___healthy": ("Your blueberry plant is in excellent condition!", "Maintain acidic soil pH 4.5-5.5. Water consistently and mulch well."),
    "Cherry_(including_sour)___Powdery_mildew": ("Fungal disease causing white powdery coating on leaves and shoots.", "Apply sulfur-based fungicide. Improve air circulation. Avoid overhead watering."),
    "Cherry_(including_sour)___healthy": ("Your cherry plant is perfectly healthy!", "Continue good cultural practices. Monitor for pests during growing season."),
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": ("Fungal disease causing rectangular gray lesions on corn leaves.", "Use resistant varieties. Apply fungicide at first signs. Rotate crops annually."),
    "Corn_(maize)___Common_rust_": ("Fungal disease causing rusty brown pustules on both leaf surfaces.", "Apply fungicide at early stages. Use resistant hybrids. Scout fields regularly."),
    "Corn_(maize)___Northern_Leaf_Blight": ("Fungal disease causing long cigar-shaped gray-green lesions.", "Use resistant varieties. Apply fungicide if severe. Practice crop rotation."),
    "Corn_(maize)___healthy": ("Your corn plant is growing beautifully!", "Ensure adequate nitrogen levels. Monitor for pest pressure during silking."),
    "Grape___Black_rot": ("Fungal disease causing circular brown lesions and fruit mummification.", "Remove infected material. Apply fungicide from bud break. Improve canopy airflow."),
    "Grape___Esca_(Black_Measles)": ("Complex fungal disease causing tiger-stripe pattern and sudden vine death.", "No cure available. Remove infected wood. Protect pruning wounds with fungicide."),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": ("Fungal disease causing dark brown angular spots on grape leaves.", "Apply copper fungicide. Improve drainage. Remove infected leaves promptly."),
    "Grape___healthy": ("Your grape vine is perfectly healthy!", "Maintain proper training system. Monitor irrigation. Scout for pests weekly."),
    "Orange___Haunglongbing_(Citrus_greening)": ("Serious bacterial disease spread by psyllid insects. Causes yellowing and misshapen fruit.", "No cure. Remove infected trees to prevent spread. Control psyllid population."),
    "Peach___Bacterial_spot": ("Bacterial disease causing dark angular spots on leaves and fruit.", "Apply copper bactericide. Use resistant varieties. Avoid overhead irrigation."),
    "Peach___healthy": ("Your peach tree is in great health!", "Thin fruit for size and quality. Monitor for borers and brown rot."),
    "Pepper,_bell___Bacterial_spot": ("Bacterial disease causing water-soaked spots that turn brown and scabby.", "Use disease-free transplants. Apply copper spray. Practice crop rotation."),
    "Pepper,_bell___healthy": ("Your bell pepper plant looks wonderful!", "Maintain consistent watering. Side-dress with fertilizer at first fruit set."),
    "Potato___Early_blight": ("Fungal disease causing dark brown concentric rings (target spots) on lower leaves.", "Apply fungicide at first sign. Remove infected leaves. Avoid overhead watering."),
    "Potato___Late_blight": ("Very serious disease causing rapid dark water-soaked lesions. Act immediately!", "Apply systemic fungicide NOW. Remove infected plants. Improve drainage urgently."),
    "Potato___healthy": ("Your potato plant is growing healthily!", "Hill soil around stems. Monitor for Colorado potato beetle. Maintain soil moisture."),
    "Raspberry___healthy": ("Your raspberry plant is thriving!", "Prune old canes after fruiting. Apply balanced fertilizer in early spring."),
    "Soybean___healthy": ("Your soybean crop looks excellent!", "Monitor for soybean aphids and sudden death syndrome. Ensure adequate potassium."),
    "Squash___Powdery_mildew": ("Fungal disease causing white powder coating on leaves, reducing yield.", "Apply potassium bicarbonate or sulfur fungicide. Ensure good airflow between plants."),
    "Strawberry___Leaf_scorch": ("Fungal disease causing purple spots with tan centers and scorched leaf edges.", "Remove infected leaves. Apply fungicide. Use certified disease-free plants."),
    "Strawberry___healthy": ("Your strawberry plants are perfectly healthy!", "Renovate beds after harvest. Monitor for spider mites in hot dry weather."),
    "Tomato___Bacterial_spot": ("Bacterial disease causing small dark water-soaked spots on leaves and fruit.", "Apply copper bactericide. Use resistant varieties. Sanitize garden tools."),
    "Tomato___Early_blight": ("Fungal disease causing concentric ring spots starting on older leaves.", "Remove lower infected leaves. Apply fungicide. Stake plants for airflow."),
    "Tomato___Late_blight": ("Destructive disease causing large dark water-soaked lesions. Spreads rapidly!", "Apply systemic fungicide immediately. Remove infected material. Improve drainage."),
    "Tomato___Leaf_Mold": ("Fungal disease causing yellow spots on upper surface, olive mold below.", "Reduce humidity. Improve ventilation. Apply fungicide at first symptoms."),
    "Tomato___Septoria_leaf_spot": ("Fungal disease causing small circular spots with dark borders and gray centers.", "Remove infected lower leaves. Apply fungicide. Avoid wetting foliage."),
    "Tomato___Spider_mites Two-spotted_spider_mite": ("Pest infestation causing yellow stippling and fine webbing on leaf undersides.", "Apply miticide or neem oil. Increase humidity. Introduce predatory mites."),
    "Tomato___Target_Spot": ("Fungal disease causing concentric ring spots on leaves, stems and fruit.", "Apply fungicide. Improve plant spacing. Remove crop debris after harvest."),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": ("Viral disease causing severe yellowing, curling, and stunted plant growth.", "No cure. Control whitefly vector. Remove infected plants. Use resistant varieties."),
    "Tomato___Tomato_mosaic_virus": ("Viral disease causing mosaic discoloration and distortion of leaves.", "No cure. Wash hands before handling plants. Control aphids. Remove infected plants."),
    "Tomato___healthy": ("Your tomato plant is in perfect health!", "Maintain consistent watering. Prune suckers. Support with cages or stakes."),
}

# ── Predict Function ──────────────────────────────────────────────────
def predict(image):
    img = image.resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, 0)

    inp = interpreter.get_input_details()
    out = interpreter.get_output_details()
    interpreter.set_tensor(inp[0]['index'], arr)
    interpreter.invoke()
    preds = interpreter.get_tensor(out[0]['index'])[0]

    top_idx    = int(np.argmax(preds))
    confidence = float(preds[top_idx]) * 100
    raw_name   = class_names[str(top_idx)]

    parts   = raw_name.split("___")
    plant   = parts[0].replace("_", " ")
    disease = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
    status  = "Healthy" if "healthy" in raw_name.lower() else "Diseased"
    desc, rec = DISEASE_INFO.get(raw_name, ("No description available.", "Consult a local agronomist."))

    top5_idx = np.argsort(preds)[-5:][::-1]
    top5 = [(class_names[str(i)].replace("___", " — ").replace("_", " "), float(preds[i]) * 100)
            for i in top5_idx]

    return plant, disease, status, confidence, desc, rec, top5

# ── Upload UI ─────────────────────────────────────────────────────────
st.markdown('<div class="upload-wrap">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Diagnosis Engine</div>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "🍃 Drop your leaf image here or click to browse",
    type=["jpg", "jpeg", "png"],
    label_visibility="visible"
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    with st.spinner("🔬 Analyzing leaf structure..."):
        plant, disease, status, conf, desc, rec, top5 = predict(image)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Image + Main Result ──
    col_img, col_main = st.columns([1, 1.6], gap="large")

    with col_img:
        st.image(image, use_column_width=True, caption="Uploaded Leaf")

    with col_main:
        # Status
        if status == "Healthy":
            st.markdown('<div class="res-card"><div class="res-label">Diagnosis Status</div><span class="badge-healthy">✦ &nbsp; Healthy Plant</span>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="res-card"><div class="res-label">Diagnosis Status</div><span class="badge-diseased">⚠ &nbsp; Disease Detected</span>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.metric("🌱 Plant", plant)
        with c2:
            st.metric("🦠 Disease", disease)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="res-label">Confidence Score</div><div class="conf-big">{conf:.1f}%</div>', unsafe_allow_html=True)
        st.progress(int(conf))
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: Description + Recommendation ──
    col_desc, col_rec = st.columns(2, gap="large")

    with col_desc:
        st.markdown(f"""
        <div class="desc-card">
          <div class="res-label">📋 Disease Description</div>
          <br>
          <div class="desc-text">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_rec:
        st.markdown(f"""
        <div class="desc-card" style="border-left-color: var(--teal)">
          <div class="res-label">💡 Recommended Action</div>
          <br>
          <div class="desc-text">{rec}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 3: Top 5 ──
    st.markdown('<div class="section-label">Top 5 Predictions</div>', unsafe_allow_html=True)

    for name, prob in top5:
        cols = st.columns([3, 1])
        with cols[0]:
            st.progress(int(prob), text=f"**{name[:45]}**")
        with cols[1]:
            st.markdown(f"<div style='text-align:right;color:var(--green);font-family:Syne,sans-serif;font-weight:600;padding-top:8px'>{prob:.1f}%</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("↑ Scan Another Leaf", type="primary"):
        st.rerun()

else:
    # Idle state hint
    st.markdown("""
    <div style='text-align:center;padding:48px 0;color:var(--muted);font-size:14px'>
      <div style='font-size:48px;margin-bottom:16px'>🌿</div>
      <div style='font-family:Syne,sans-serif;font-weight:600;font-size:18px;color:var(--text);margin-bottom:8px'>Ready to Diagnose</div>
      Upload a clear photo of a plant leaf to get instant AI-powered disease detection
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer-wrap">
  Built with MobileNetV2 Transfer Learning &nbsp;·&nbsp; PlantVillage Dataset
  &nbsp;·&nbsp; 95.53% Test Accuracy &nbsp;·&nbsp; 38 Disease Classes
</div>
""", unsafe_allow_html=True)
