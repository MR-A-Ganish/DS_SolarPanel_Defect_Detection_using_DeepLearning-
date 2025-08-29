import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="Solar Panel Defect Detection", page_icon="🔍", layout="wide")
st.title("🔍 Solar Panel Defect Detection")
st.markdown("Upload a solar panel image to detect defects with **interactive visualizations**.")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("solar_panel_defect_model.h5")

model = load_model()

# Define class labels
class_labels = [
    "clean",
    "snow-damage",
    "electrical-damage",
    "dusty",
    "bird-damage",
    "physical-damage"
]

# Upload images
uploaded_files = st.file_uploader(
    "📷 Upload solar panel image(s)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# Store results for dashboard
results = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.divider()
        st.markdown(f"### 📂 `{uploaded_file.name}`")

        try:
            # Open & preprocess image
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)

            img_resized = image.resize((128, 128))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            prediction = model.predict(img_array, verbose=0)

            if prediction.shape[1] > len(class_labels):
                prediction = prediction[:, :len(class_labels)]

            prediction = tf.nn.softmax(prediction).numpy()

            # Get best prediction
            predicted_class = class_labels[np.argmax(prediction)]
            confidence = np.max(prediction)

            # Save result for dashboard
            results.append((uploaded_file.name, predicted_class, confidence, prediction[0]))

            st.markdown(f"**🧠 Predicted Class:** `{predicted_class}`")
            st.markdown(f"**📊 Confidence:** `{confidence:.2f}`")
            st.progress(float(confidence))

            # --- Interactive Visualizations ---
            st.subheader("Interactive Confidence Visualizations")

            # 1. Interactive Bar Chart
            fig_bar = px.bar(
                x=class_labels,
                y=prediction[0],
                labels={"x": "Class", "y": "Confidence"},
                title="Confidence per Class",
                text=[f"{p:.2f}" for p in prediction[0]]
            )
            fig_bar.update_traces(textposition="outside")
            st.plotly_chart(fig_bar, use_container_width=True)

            # 2. Interactive Pie Chart
            fig_pie = px.pie(
                names=class_labels,
                values=prediction[0],
                title="Class Probability Distribution",
                hole=0.3
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            # 3. Interactive Line Chart
            fig_line = px.line(
                x=class_labels,
                y=prediction[0],
                markers=True,
                title="Confidence Distribution Across Classes"
            )
            st.plotly_chart(fig_line, use_container_width=True)

            # 4. Interactive Radar Chart
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=prediction[0].tolist() + [prediction[0][0]],
                theta=class_labels + [class_labels[0]],
                fill="toself",
                name="Confidence"
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                showlegend=False,
                title="Radar Chart of Class Confidence"
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            # 5. Top-3 Predictions
            top_indices = np.argsort(prediction[0])[::-1][:3]
            top_labels = [class_labels[i] for i in top_indices]
            top_values = prediction[0][top_indices]

            fig_top3 = px.bar(
                x=top_labels,
                y=top_values,
                text=[f"{v:.2f}" for v in top_values],
                color=top_labels,
                title="Top-3 Predictions"
            )
            fig_top3.update_traces(textposition="outside")
            st.plotly_chart(fig_top3, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error: {e}")

    # --- Multi-image comparison dashboard ---
    if len(results) > 1:
        st.subheader("📊 Multi-Image Comparison Dashboard")

        for filename, predicted_class, confidence, probs in results:
            st.markdown(f"**📂 {filename} → 🧠 `{predicted_class}` ({confidence:.2f})**")
            fig_multi = px.bar(
                x=class_labels,
                y=probs,
                title=f"Confidence Breakdown - {filename}",
                labels={"x": "Class", "y": "Confidence"},
                color=class_labels
            )
            st.plotly_chart(fig_multi, use_container_width=True)

else:
    st.info("Upload one or more solar panel images to begin.")
