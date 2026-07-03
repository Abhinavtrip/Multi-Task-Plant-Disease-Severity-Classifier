import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# ── Page setup ────────────────────────────────────────────────
st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿")
st.title("🌿 Plant Disease and Severity Detector")
st.write("Upload a leaf photo — the model will tell you the disease and how serious it is.")
st.markdown("---")


@st.cache_resource
def load_model():
    model      = tf.keras.models.load_model("model/plant_multitask_model.keras")
    names      = json.load(open("model/class_names.json"))
    return model, names

# Upload image 
photo = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if photo:
    # Show the image
    leaf = Image.open(photo).convert("RGB")
    st.image(leaf, caption="Your leaf", width=300)

    # Prep img for model
    leaf_resized  = leaf.resize((224, 224))
    leaf_array    = np.array(leaf_resized, dtype=np.float32)
    leaf_array    = np.expand_dims(leaf_array, axis=0)   # shape: (1, 224, 224, 3)

    # Predict
    with st.spinner("Analyzing..."):
        model, class_names = load_model()
        disease_pred, severity_pred = model.predict(leaf_array, verbose=0)

    #Disease result 
    disease_index      = int(np.argmax(disease_pred[0]))
    disease_confidence = float(np.max(disease_pred[0])) * 100
    disease_raw        = class_names[disease_index]

    # Clean up the name 
    parts      = disease_raw.split("___")
    plant_name = parts[0].replace("_", " ")
    condition  = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
    is_healthy = "healthy" in condition.lower()

    # Severity result 
    severity_index = int(np.argmax(severity_pred[0]))
    severity_confidence = float(np.max(severity_pred[0])) * 100

    severity_label = {
        0: "✅ Healthy",
        1: "⚠️ Early Stage",
        2: "🚨 Critical"
    }

    severity_advice = {
        0: "The plant looks completely healthy. No action needed.",
        1: "Disease spotted early. Consider treatment soon.",
        2: "Severe infection. Immediate treatment required."
    }

    # Show theresults
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔬 Disease")
        st.markdown(f"**Plant:** {plant_name}")
        if is_healthy:
            st.success(f"✅ {condition}")
        else:
            st.error(f"🦠 {condition}")
        st.markdown(f"Confidence: **{disease_confidence:.1f}%**")
        st.progress(disease_confidence / 100)

    with col2:
        st.markdown("### 📊 Severity")
        label = severity_label[severity_index]
        advice = severity_advice[severity_index]

        if severity_index == 0:
            st.success(label)
        elif severity_index == 1:
            st.warning(label)
        else:
            st.error(label)

        st.markdown(f"_{advice}_")
        st.markdown(f"Confidence: **{severity_confidence:.1f}%**")
        st.progress(severity_confidence / 100)

    #Top 3 guesses
    st.markdown("---")
    st.markdown("**Top 3 Predictions:**")
    top3 = np.argsort(disease_pred[0])[::-1][:3]
    for i in top3:
        p, c   = class_names[i].split("___")
        score  = disease_pred[0][i] * 100
        st.markdown(f"- **{p.replace('_',' ')}** → {c.replace('_',' ')}  `{score:.1f}%`")

st.markdown("---")
st.caption("EfficientNetB0 · Multi-Task Learning · PlantVillage Dataset · TensorFlow/Keras")
