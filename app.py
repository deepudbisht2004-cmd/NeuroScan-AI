
import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image
from datetime import datetime
from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧠",
    layout="centered"
)

# ---------------- MOBILE DESIGN ----------------

st.markdown("""
<style>
.stApp {
    background: #081426;
    color: white;
}
header[data-testid="stHeader"] {
    background: transparent;
}
.block-container {
    max-width: 850px;
    padding: 1.5rem 1rem 4rem;
}
.hero {
    background: linear-gradient(135deg,#142d50,#0c1b32);
    padding: 25px 20px;
    border: 1px solid #29496f;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 20px;
}
.hero h1 {
    color: #ffffff;
    font-size: 30px;
    margin-bottom: 5px;
}
.hero p {
    color: #a9c4e6;
}
.section {
    color: #ffffff;
    font-size: 21px;
    font-weight: 700;
    margin: 22px 0 12px;
}
div[data-testid="stFileUploader"] {
    background: #10233d;
    border: 1px dashed #5799dc;
    padding: 12px;
    border-radius: 15px;
}
div.stButton > button,
div.stDownloadButton > button {
    width: 100%;
    min-height: 48px;
    border-radius: 12px;
    font-weight: 700;
}

/* Blue PDF Download Button */
div.stDownloadButton > button {
    background-color: #1677e8;
    color: white;
    border: none;
}

div.stDownloadButton > button:hover {
    background-color: #095bb5;
    color: white;
}
.result-card {
    background: #142d50;
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #31577f;
    text-align: center;
}
.result-card h2 {
    color: #70b8ff;
    font-size: 25px;
}
.small-note {
    color: #a9c4e6;
    font-size: 13px;
}
footer {visibility: hidden;}
@media (max-width: 600px) {
    .block-container {
        padding: 1rem 0.7rem 3rem;
    }
    .hero h1 {
        font-size: 25px;
    }
    .section {
        font-size: 19px;
    }
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------

MODEL_PATH = "brain_tumor_model (5).keras"
CLASS_NAMES_PATH = "class_names (5).json"

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    with open(CLASS_NAMES_PATH, "r") as file:
        class_names = json.load(file)

    return model, class_names


try:
    model, class_names = load_model()
except Exception as e:
    st.error("Could not load the model or class names.")
    st.code(str(e))
    st.stop()

# ---------------- PDF REPORT ----------------

def create_pdf(result, image_name):

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("NEUROSCAN AI", styles["Title"]))
    content.append(Paragraph(
        "Brain MRI Classification Report",
        styles["Heading2"]
    ))
    content.append(Spacer(1, 20))

    data = [
        ["Parameter", "Result"],
        ["Image", image_name],
        ["Predicted category", result["class"].upper()],
        ["Model score", f'{result["confidence"]:.2f}%'],
        ["Date", datetime.now().strftime("%d-%m-%Y %H:%M")]
    ]

    table = Table(data, colWidths=[180, 300])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.7, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 10),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
    ]))

    content.append(table)
    content.append(Spacer(1, 20))
    content.append(Paragraph(
        "Class-wise prediction scores",
        styles["Heading2"]
    ))

    scores = [["Category", "Score"]]
    for name, score in result["scores"].items():
        scores.append([name.upper(), f"{score:.2f}%"])

    score_table = Table(scores, colWidths=[180, 300])
    score_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.7, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 10)
    ]))

    content.append(score_table)
    content.append(Spacer(1, 25))
    content.append(Paragraph(
        "<b>Disclaimer:</b> This is an educational AI research prototype. "
        "It is not a medical diagnosis and must not be used for clinical "
        "decisions. Consult a qualified medical professional.",
        styles["Normal"]
    ))

    doc.build(content)
    buffer.seek(0)
    return buffer.getvalue()

# ---------------- HEADER ----------------

st.markdown("""
<div class="hero">
    <h1>🧠 NeuroScan AI</h1>
    <p>AI-Powered Brain MRI Classification</p>
    <p>Upload • Analyze • View Results</p>
</div>
""", unsafe_allow_html=True)

# ---------------- NAVIGATION ----------------

page = st.radio(
    "Navigation",
    ["Dashboard", "Prediction History", "About"],
    horizontal=True,
    label_visibility="collapsed"
)

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- DASHBOARD ----------------

if page == "Dashboard":

    st.markdown(
        '<div class="section">📤 Upload MRI Image</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Choose a brain MRI image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded MRI Image",
            use_container_width=True
        )

        if st.button("🔍 Analyze MRI Image", type="primary"):

            with st.spinner("Analyzing image..."):

                img = image.resize((224, 224))
                img_array = np.asarray(img, dtype=np.float32)
                img_array = np.expand_dims(img_array, axis=0)

                predictions = model.predict(
                    img_array,
                    verbose=0
                )[0]

                # Convert outputs to probabilities if necessary
                if (
                    np.all(predictions >= 0)
                    and np.isclose(np.sum(predictions), 1.0, atol=0.01)
                ):
                    probabilities = predictions
                else:
                    shifted = predictions - np.max(predictions)
                    exp_values = np.exp(shifted)
                    probabilities = exp_values / np.sum(exp_values)

                predicted_index = int(np.argmax(probabilities))
                predicted_class = class_names[predicted_index]
                confidence = float(probabilities[predicted_index] * 100)

                result = {
                    "class": predicted_class,
                    "confidence": confidence,
                    "scores": {
                        name: float(prob * 100)
                        for name, prob in zip(
                            class_names, probabilities
                        )
                    }
                }

                st.session_state.last_result = result
                st.session_state.last_image_name = uploaded_file.name

                st.session_state.history.insert(0, {
                    "image": uploaded_file.name,
                    "class": predicted_class,
                    "confidence": confidence,
                    "time": datetime.now().strftime("%d-%m-%Y %H:%M")
                })

        if "last_result" in st.session_state:

            result = st.session_state.last_result

            st.markdown(
                '<div class="section">📊 Prediction Result</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="result-card">
                    <p>Predicted Category</p>
                    <h2>{result["class"].upper()}</h2>
                    <h3>{result["confidence"]:.2f}%</h3>
                    <p>Model Score</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("### Class-wise Scores")

            for name, score in result["scores"].items():
                st.write(f"**{name.upper()} — {score:.2f}%**")
                st.progress(min(score / 100, 1.0))

            pdf = create_pdf(
                result,
                st.session_state.last_image_name
            )

            st.download_button(
                "📄 Download PDF Report",
                data=pdf,
                file_name="NeuroScan_AI_Report.pdf",
                mime="application/pdf"
            )

    st.markdown("---")
    st.markdown(
        '<p class="small-note">Educational prototype only. '
        'Not intended for medical diagnosis.</p>',
        unsafe_allow_html=True
    )

# ---------------- HISTORY ----------------

elif page == "Prediction History":

    st.markdown("## 📋 Prediction History")

    if st.session_state.history:

        for item in st.session_state.history:

            with st.container(border=True):
                st.write(f"**Image:** {item['image']}")
                st.write(f"**Prediction:** {item['class'].upper()}")
                st.write(f"**Score:** {item['confidence']:.2f}%")
                st.caption(item["time"])

        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()

    else:
        st.info("No predictions available yet.")

# ---------------- ABOUT ----------------

elif page == "About":

    st.markdown("## 🧠 About NeuroScan AI")

    st.write("""
    NeuroScan AI is an educational brain MRI image classification
    project built using deep learning and EfficientNetB0.

    The model classifies images into four dataset categories:
    """)

    for name in class_names:
        st.write(f"• {name.upper()}")

    st.markdown("### Technology Used")
    st.write("""
    - Python
    - TensorFlow / Keras
    - EfficientNetB0
    - Streamlit
    - NumPy
    - Pillow
    - ReportLab
    """)

    st.warning(
        "This application is an educational prototype, "
        "not a clinically validated diagnostic tool."
    )
