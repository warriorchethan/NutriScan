import os
import html
import time
import hashlib
import streamlit as st
from PIL import Image
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()

# 1. Define the exact JSON structure we want back from Gemini
class NutriScanResult(BaseModel):
    score: int = Field(description="Score from 1 to 10 based on nutritional quality and processing.")
    card_color: str = Field(description="Must be strictly 'Red', 'Yellow', or 'Green'. (1-5: Red, 6-8: Yellow, 9-10: Green)")
    calories_estimate: str = Field(description="Estimated calories per serving (from label or estimated by item type).")
    health_impacts: list[str] = Field(description="Physiological benefits (if Green) or specific chemical/metabolic risks (if Red/Yellow), tailored to the active Demographic Mode.")
    demographic_warnings: list[str] = Field(description="Specific ingredients flagged for the chosen demographic mode (e.g., Red 40 for Kids, Seed Oils for Adults, Phosphates/Sodium for Seniors).")
    detected_allergens: list[str] = Field(description="List of user-selected allergens found or suspected in the ingredient list.")
    nova_group: int = Field(description="NOVA classification integer strictly from 1 to 4.")
    nova_explanation: str = Field(description="Short scientific explanation of why this food falls into this NOVA group.")
    greenwashing_detected: bool = Field(description="True if marketing claims contradict the ingredient reality.")
    greenwashing_analysis: str = Field(description="Explanation comparing front-of-pack buzzwords (or perceived healthiness) against actual chemical ingredients.")
    swaps: list[str] = Field(description="Provide 2 specific clean-label product or food alternatives if score <= 8. Empty list if score is 9-10.")

# 2. Function to call Gemini Vision with Session Caching (Using gemini-3.8-flash)
def analyze_image(image: Image.Image, mode: str, allergens: list[str], marketing_claims: str) -> NutriScanResult:
    # Speed optimization: Resize image in memory before sending to API
    img_copy = image.copy()
    img_copy.thumbnail((800, 800))

    # Initialize session cache so re-scanning the same image + mode uses 0 API quota
    if "scan_cache" not in st.session_state:
        st.session_state["scan_cache"] = {}

    img_bytes = img_copy.tobytes()
    cache_key = hashlib.md5(
        img_bytes + f"{mode}|{','.join(sorted(allergens))}|{marketing_claims}".encode()
    ).hexdigest()

    if cache_key in st.session_state["scan_cache"]:
        return st.session_state["scan_cache"][cache_key]

    client = genai.Client()

    prompt = f"""
    You are NutriScan, a clinical nutritionist and food scientist AI. Analyze this food label image.
    
    USER PROFILE SETTINGS:
    - Active Demographic Mode: {mode}
      * If 'Kids Mode': Aggressively hunt for artificial dyes (Red 40, Yellow 5, Blue 1), HFCS, and hyperactivity-linked preservatives.
      * If 'Adult Mode': Target metabolic disruptors, industrial seed oils, trans fats, hidden sugars, and artificial sweeteners.
      * If 'Senior Mode': Flag hidden sodium, added phosphates, potassium additives, and heart/kidney stressors.
    - User Allergens to Watch For: {', '.join(allergens) if allergens else 'None'}
    - Front-of-Package Marketing Claims: "{marketing_claims if marketing_claims else 'Analyze based on visible text or typical product category marketing'}"
    
    Evaluate the ingredients strictly and populate the required structured analysis. If the score is 8 or below, you MUST provide 2 specific, healthier clean-label product alternatives in the swaps list.
    """

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=NutriScanResult,
        temperature=0.2,
    )

    # Use the active gemini-3.8-flash model
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=[img_copy, prompt],
        config=config,
    )
    
    parsed_result = NutriScanResult.model_validate_json(response.text)
    st.session_state["scan_cache"][cache_key] = parsed_result
    return parsed_result

# 3. Streamlit UI Configuration & Aura+++++++ CSS
st.set_page_config(page_title="NutriScan AI // Molecular Food Intelligence", page_icon="🧬", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@500;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@700;800&display=swap');

.stApp {
    background-color: #060911;
    background-image: 
        radial-gradient(circle at 15% 15%, rgba(16, 185, 129, 0.12) 0%, transparent 40%),
        radial-gradient(circle at 85% 20%, rgba(6, 182, 212, 0.12) 0%, transparent 40%),
        radial-gradient(circle at 50% 85%, rgba(139, 92, 246, 0.10) 0%, transparent 50%);
    color: #f8fafc;
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.stApp label, .stApp p, .stApp span, [data-testid="stWidgetLabel"] {
    color: #e2e8f0 !important;
}

[data-testid="stSidebar"] {
    background: rgba(8, 12, 22, 0.92) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

.hero-box {
    padding: 28px 32px;
    border-radius: 22px;
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(15, 23, 42, 0.5) 100%);
    border: 1px solid rgba(255, 255, 255, 0.12);
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5);
    margin-bottom: 24px;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 5px 14px;
    border-radius: 999px;
    background: rgba(16, 185, 129, 0.14);
    border: 1px solid rgba(16, 185, 129, 0.4);
    color: #34d399 !important;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.pulse-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #10b981;
    box-shadow: 0 0 10px #10b981;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 44px;
    font-weight: 800;
    margin: 0 0 8px 0;
    background: linear-gradient(90deg, #ffffff 0%, #6ee7b7 50%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.bento-card {
    background: rgba(15, 23, 42, 0.72);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 18px;
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
}

.card-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #94a3b8 !important;
    margin-bottom: 10px;
    display: flex;
    justify-content: space-between;
}

@keyframes criticalPulse {
    0%, 100% { box-shadow: 0 0 20px rgba(244, 63, 94, 0.4); border-color: #f43f5e; }
    50% { box-shadow: 0 0 40px rgba(244, 63, 94, 0.85); border-color: #fda4af; }
}

.allergen-hud {
    background: linear-gradient(135deg, rgba(136, 19, 55, 0.85), rgba(76, 5, 25, 0.95));
    border: 2px solid #f43f5e;
    border-radius: 16px;
    padding: 18px 24px;
    margin-bottom: 22px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
    animation: criticalPulse 1.5s infinite;
}

.allergen-pill {
    background: rgba(244, 63, 94, 0.35);
    border: 1px solid #fda4af;
    color: #fff !important;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 12px;
    padding: 5px 12px;
    border-radius: 999px;
}

.nova-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
    margin: 12px 0;
}

.nova-step {
    padding: 8px;
    border-radius: 8px;
    text-align: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 700;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    color: #64748b;
}

.nova-active-1 { background: rgba(16, 185, 129, 0.25); border-color: #10b981; color: #6ee7b7 !important; }
.nova-active-2 { background: rgba(56, 189, 248, 0.25); border-color: #38bdf8; color: #bae6fd !important; }
.nova-active-3 { background: rgba(245, 158, 11, 0.25); border-color: #f59e0b; color: #fde68a !important; }
.nova-active-4 { background: rgba(244, 63, 94, 0.28); border-color: #f43f5e; color: #fecdd3 !important; }

.item-pill {
    padding: 11px 14px;
    border-radius: 12px;
    margin-bottom: 8px;
    font-size: 13.5px;
    display: flex;
    gap: 10px;
    align-items: flex-start;
}

.swap-item { background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.3); color: #d1fae5; }
.warn-item { background: rgba(245, 158, 11, 0.12); border-left: 4px solid #f59e0b; color: #fef3c7; }
.impact-item { background: rgba(255, 255, 255, 0.04); border: 1px solid rgba(255, 255, 255, 0.08); color: #e2e8f0; }

div.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #10b981 0%, #06b6d4 50%, #6366f1 100%) !important;
    border: none !important;
    color: white !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    padding: 14px 24px !important;
    border-radius: 14px !important;
    box-shadow: 0 8px 25px rgba(6, 182, 212, 0.4) !important;
}
</style>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("""
    <div>
        <div class="status-pill"><span class="pulse-dot"></span> BIO-LENS CONFIG</div>
        <h2 style="margin:0; font-family:'Space Grotesk',sans-serif; color:#ffffff;">⚙️ Smart Filters</h2>
        <p style="color:#94a3b8; font-size:13px; margin-top:4px;">Calibrate the AI toxicology engine for your physiology.</p>
    </div>
    """, unsafe_allow_html=True)

    demographic_mode = st.radio(
        "👤 Demographic Mode",
        ["Kids Mode 🧒", "Adult Mode 🧑", "Senior Mode 🧓"],
        help="Changes how the AI weighs specific chemical risks."
    )

    mode_descriptions = {
        "Kids Mode 🧒": "Targets synthetic azo dyes (Red 40, Yellow 5), HFCS, and hyperactivity triggers.",
        "Adult Mode 🧑": "Targets industrial seed oils, hidden glycemic spikes, and metabolic disruptors.",
        "Senior Mode 🧓": "Targets hidden sodium, inorganic phosphates, and cardiovascular/renal stressors."
    }
    st.caption(f"🔬 **Active Protocol:** {mode_descriptions.get(demographic_mode, '')}")

    st.divider()

    selected_allergens = st.multiselect(
        "⚠️ Allergen Radar",
        ["Dairy", "Gluten", "Peanuts", "Tree Nuts", "Soy", "Eggs", "Shellfish", "Sesame"],
        default=["Gluten", "Dairy"]
    )

    st.divider()
    st.info("🏆 **Pitch Tip:** Toggle between **Kids** & **Senior** mode on the same photo to show live dynamic risk adaptation!")

# ==================== MAIN DASHBOARD ====================
st.markdown(f"""
<div class="hero-box">
    <div class="status-pill">
        <span class="pulse-dot"></span> NUTRISCAN VISION ENGINE // ACTIVE LENS: {html.escape(demographic_mode.upper())} • ALLERGEN SHIELDS: {len(selected_allergens)}
    </div>
    <div class="hero-title">🥗 NutriScan AI</div>
    <p style="color:#94a3b8; font-size:16px; margin:0;">
        Expose harmful chemicals, decode ultra-processed formulations, and crush deceptive front-of-box marketing.
    </p>
</div>
""", unsafe_allow_html=True)

col_input1, col_input2 = st.columns([1.05, 1], gap="large")

with col_input1:
    st.markdown('<div class="card-label"><span>📸 01 // OPTICAL LABEL CAPTURE</span><span>JPG • PNG • WEBP</span></div>', unsafe_allow_html=True)
    input_method = st.radio("Select Image Source:", ["Upload Photo", "Use Camera"], horizontal=True)
    if input_method == "Upload Photo":
        uploaded_file = st.file_uploader("Snap or upload an ingredient list", type=["jpg", "jpeg", "png", "webp"])
    else:
        uploaded_file = st.camera_input("Take a picture of the ingredient label")

with col_input2:
    st.markdown('<div class="card-label"><span>📢 02 // FRONT-OF-PACK MARKETING AUDIT</span><span>OPTIONAL</span></div>', unsafe_allow_html=True)
    marketing_claims = st.text_input(
        "📢 Front-of-Box Marketing Claims (Optional)",
        placeholder="e.g., 'All-Natural', 'Heart Healthy', 'Zero Sugar', 'High Fiber'..."
    )
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Locked Target: Scanned Label", width="stretch")
    else:
        st.markdown("""
        <div class="bento-card" style="text-align:center; padding:32px 20px; border-style:dashed;">
            <div style="font-size:32px; margin-bottom:8px;">🧬</div>
            <div style="font-weight:700; color:#e2e8f0; font-size:15px;">Awaiting Ingredient Label Telemetry</div>
            <div style="font-size:13px; color:#64748b; margin-top:4px;">
                Upload a food label photo on the left to run NOVA classification, allergen detection, and greenwashing audits.
            </div>
        </div>
        """, unsafe_allow_html=True)

# Trigger Analysis
if uploaded_file and st.button("⚡ INITIALIZE MOLECULAR SCAN", type="primary", width="stretch"):
    with st.spinner("🧬 Decoding chemical compounds, cross-referencing NOVA database..."):
        try:
            result = analyze_image(image, demographic_mode, selected_allergens, marketing_claims)
            st.divider()

            # 1. Flashing Allergen Alert
            if result.detected_allergens:
                pills_html = "".join(f'<span class="allergen-pill">⚠️ {html.escape(a)}</span>' for a in result.detected_allergens)
                st.markdown(f"""
                <div class="allergen-hud">
                    <div style="font-family:'Space Grotesk',sans-serif; font-size:20px; font-weight:800; color:#ffe4e6;">
                        🚨 CRITICAL ALLERGEN ALERT: DETECTED TARGET ALLERGENS
                    </div>
                    <div style="display:flex; gap:8px; flex-wrap:wrap;">{pills_html}</div>
                </div>
                """, unsafe_allow_html=True)

            # 2. Core Bento Results Grid
            c1, c2, c3 = st.columns([1.15, 1.15, 1.35], gap="medium")

            color_map = {
                "Red": {"hex": "#f43f5e", "label": "HIGH RISK // AVOID"},
                "Yellow": {"hex": "#f59e0b", "label": "MODERATE // CAUTION"},
                "Green": {"hex": "#10b981", "label": "CLEAN // APPROVED"},
            }
            theme = color_map.get(result.card_color, color_map["Yellow"])
            dash_offset = 283 - (283 * max(0, min(100, result.score * 10)) / 100)

            with c1:
                st.markdown(f"""
                <div class="bento-card" style="text-align:center; border-color:{theme['hex']};">
                    <div style="color:{theme['hex']}; font-family:'JetBrains Mono',monospace; font-size:12px; font-weight:700; margin-bottom:8px;">
                        {html.escape(result.card_color.upper())} CARD • {theme['label']}
                    </div>
                    <svg width="145" height="145" viewBox="0 0 110 110">
                        <circle cx="55" cy="55" r="45" fill="none" stroke="rgba(255,255,255,0.08)" stroke-width="9"/>
                        <circle cx="55" cy="55" r="45" fill="none" stroke="{theme['hex']}" stroke-width="9"
                            stroke-linecap="round" stroke-dasharray="283" stroke-dashoffset="{dash_offset}"
                            transform="rotate(-90 55 55)"/>
                        <text x="50%" y="50%" text-anchor="middle" dy=".32em" fill="#ffffff"
                            font-family="Space Grotesk, sans-serif" font-size="28" font-weight="800">
                            {result.score}/10
                        </text>
                    </svg>
                    <div style="font-size:13px; color:#94a3b8; margin-top:6px;">Overall Safety & Quality Score</div>
                </div>
                """, unsafe_allow_html=True)

                swaps_html = ""
                if result.score <= 8 and result.swaps:
                    for swap in result.swaps:
                        swaps_html += f'<div class="item-pill swap-item"><span>🌱</span><div><strong>Try:</strong> {html.escape(swap)}</div></div>'
                else:
                    st.balloons()
                    swaps_html = '<div class="item-pill swap-item"><span>🌟</span><div><strong>Top-tier clean product!</strong> No swap needed.</div></div>'

                st.markdown(f"""
                <div class="bento-card">
                    <div class="card-label"><span>🔄 BETTER ALTERNATES (SWAPS)</span><span>CLEAN UPGRADES</span></div>
                    {swaps_html}
                </div>
                """, unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                <div class="bento-card">
                    <div class="card-label"><span>🔥 CALORIC ESTIMATE</span><span>PER SERVING</span></div>
                    <div style="font-family:'Space Grotesk',sans-serif; font-size:28px; font-weight:800; color:#fff;">
                        {html.escape(result.calories_estimate)}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                nova_labels = {
                    1: "Group 1: Unprocessed / Minimally Processed",
                    2: "Group 2: Processed Culinary Ingredients",
                    3: "Group 3: Processed Foods",
                    4: "Group 4: Ultra-Processed Food (UPF)"
                }
                nova_bars = "".join(
                    f'<div class="nova-step {"nova-active-" + str(g) if result.nova_group == g else ""}">G{g}</div>'
                    for g in range(1, 5)
                )
                st.markdown(f"""
                <div class="bento-card">
                    <div class="card-label"><span>🧪 NOVA SCALE</span><span>GROUP {result.nova_group}/4</span></div>
                    <div style="font-weight:700; font-size:14px; color:#f8fafc;">{html.escape(nova_labels.get(result.nova_group, ''))}</div>
                    <div class="nova-grid">{nova_bars}</div>
                    <div style="font-size:13px; color:#94a3b8;">{html.escape(result.nova_explanation)}</div>
                </div>
                """, unsafe_allow_html=True)

                gw_status = "⚠️ DECEPTIVE MARKETING" if result.greenwashing_detected else "✅ TRANSPARENT LABEL"
                gw_color = "#f43f5e" if result.greenwashing_detected else "#10b981"
                st.markdown(f"""
                <div class="bento-card" style="border-color:{gw_color};">
                    <div class="card-label"><span>🕵️‍♂️ GREENWASHING CHECK</span><span style="color:{gw_color};">{gw_status}</span></div>
                    <div style="font-size:13.5px; color:#e2e8f0;">{html.escape(result.greenwashing_analysis)}</div>
                </div>
                """, unsafe_allow_html=True)

            with c3:
                if result.demographic_warnings:
                    warns_html = "".join(
                        f'<div class="item-pill warn-item"><span>⚠️</span><div><strong>Flagged:</strong> {html.escape(w)}</div></div>'
                        for w in result.demographic_warnings
                    )
                else:
                    warns_html = f'<div class="item-pill swap-item"><span>✅</span><div>No specific triggers found for {html.escape(demographic_mode)}!</div></div>'

                st.markdown(f"""
                <div class="bento-card">
                    <div class="card-label"><span>🎯 DEMOGRAPHIC ANALYSIS</span><span>{html.escape(demographic_mode.upper())}</span></div>
                    {warns_html}
                </div>
                """, unsafe_allow_html=True)

                impacts_html = "".join(
                    f'<div class="item-pill impact-item"><span>🧬</span><div>{html.escape(impact)}</div></div>'
                    for impact in result.health_impacts
                )
                st.markdown(f"""
                <div class="bento-card">
                    <div class="card-label"><span>🩺 PHYSIOLOGICAL IMPACTS</span><span>BIOMARKERS</span></div>
                    {impacts_html}
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error analyzing image: {e}")

# === END OF FILE ===