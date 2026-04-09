import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import joblib
import numpy as np
import os

# ---------------- PATH ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------- PAGE ----------------
st.set_page_config(page_title="AI Breast Cancer Diagnosis", page_icon="🩺", layout="wide")
st.title("🩺 AI Breast Cancer Diagnosis System")

# ---------------- DEBUG ----------------
# st.write("📂 Current Directory:", BASE_DIR)
# st.write("📁 Files:", os.listdir(BASE_DIR))

# ---------------- MODEL ----------------
@st.cache_resource
def load_pytorch_model():

    class AttentionFusion(nn.Module):
        def __init__(self, radiomic_feature_count=42):
            super().__init__()
            self.radiomic_projection = nn.Linear(radiomic_feature_count, 2048)
            self.attention_network = nn.Sequential(
                nn.Linear(4096, 256),
                nn.ReLU(),
                nn.Linear(256, 2)
            )

        def forward(self, v, r_raw):
            r = self.radiomic_projection(r_raw)
            concat = torch.cat((v, r), dim=1)
            attn = self.attention_network(concat)
            weights = torch.softmax(attn, dim=1)
            return v * weights[:, 0:1] + r * weights[:, 1:2]

    class HybridBreastCancerModel(nn.Module):
        def __init__(self, radiomic_feature_count=42):
            super().__init__()

            resnet = models.resnet50(weights=None)
            self.deep_extractor = nn.Sequential(*list(resnet.children())[:-1])

            self.fusion = AttentionFusion(radiomic_feature_count)

            # ✅ CORRECT CLASSIFIER (512)
            self.classifier = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Linear(512, 1)
            )

        def forward(self, img, radiomics):
            v = self.deep_extractor(img).squeeze()

            if len(v.shape) == 1:
                v = v.unsqueeze(0)

            fused = self.fusion(v, radiomics)
            return self.classifier(fused)

    try:
        model_path = os.path.join(BASE_DIR, "adaptive_hybrid_breast_cancer_model.pth")
        model = HybridBreastCancerModel()
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model, True
    except Exception as e:
        st.error(f"❌ PyTorch Load Error: {e}")
        return None, False


@st.cache_resource
def load_rf_model():
    try:
        rf_path = os.path.join(BASE_DIR, "breast_model.pkl")
        return joblib.load(rf_path), True
    except Exception as e:
        st.error(f"❌ Random Forest Load Error: {e}")
        return None, False


pytorch_model, pytorch_loaded = load_pytorch_model()
rf_model, rf_loaded = load_rf_model()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if pytorch_loaded:
    pytorch_model = pytorch_model.to(device)

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------------- FEATURE NAMES ----------------
feature_names = [
    "Mean Radius", "Mean Texture", "Mean Perimeter", "Mean Area",
    "Mean Smoothness", "Mean Compactness", "Mean Concavity",
    "Mean Concave Points", "Mean Symmetry", "Mean Fractal Dimension",
    "Radius Error", "Texture Error", "Perimeter Error", "Area Error",
    "Smoothness Error", "Compactness Error", "Concavity Error",
    "Concave Points Error", "Symmetry Error", "Fractal Dimension Error",
    "Worst Radius", "Worst Texture", "Worst Perimeter", "Worst Area",
    "Worst Smoothness", "Worst Compactness", "Worst Concavity",
    "Worst Concave Points", "Worst Symmetry", "Worst Fractal Dimension"
]

# ---------------- UI ----------------
st.write("### Select Data Input Method")
tab1, tab2, tab3 = st.tabs([
    "🖼️ Image Scan (PyTorch)",
    "📝 Clinical Data (Random Forest)",
    "🤝 Hybrid Mode"
])

# ---------------- TAB 1 ----------------
with tab1:
    st.info("Upload a medical scan.")

    if not pytorch_loaded:
        st.error("PyTorch model not loaded.")
    else:
        file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

        if file:
            image = Image.open(file).convert("RGB")
            st.image(image, width=400)

            if st.button("Run Image Analysis"):
                img = transform(image).unsqueeze(0).to(device)
                dummy = torch.zeros((1, 42)).to(device)

                with torch.no_grad():
                    prob = torch.sigmoid(pytorch_model(img, dummy)).item()

                if prob > 0.5:
                    st.error(f"⚠️ Malignant ({prob*100:.2f}%)")
                else:
                    st.success(f"✅ Benign ({(1-prob)*100:.2f}%)")

# ---------------- TAB 2 ----------------
with tab2:
    st.write("### Enter Clinical Data")

    if not rf_loaded:
        st.error("RF model not loaded.")
    else:
        features = []
        cols = st.columns(3)

        for i, name in enumerate(feature_names):
            with cols[i % 3]:
                val = st.number_input(name, value=0.0, format="%.4f")
                features.append(val)

        if st.button("Run Clinical Analysis"):
            data = np.array(features).reshape(1, -1)
            prob = rf_model.predict_proba(data)[0][0]

            if prob > 0.85:
                st.error(f"⚠️ Malignant ({prob*100:.2f}%)")
            else:
                st.success(f"✅ Benign ({prob*100:.2f}%)")

# ---------------- TAB 3 ----------------
with tab3:
    st.write("### 🤝 Hybrid Diagnosis")

    if not pytorch_loaded:
        st.error("Model not loaded.")
    else:
        file = st.file_uploader("Upload Image", key="hybrid")

        if file:
            image = Image.open(file).convert("RGB")
            st.image(image, width=400)

        st.write("### Clinical Features")

        features = []
        cols = st.columns(3)

        for i, name in enumerate(feature_names):
            with cols[i % 3]:
                val = st.number_input(name, value=0.0, key=f"h{i}")
                features.append(val)

        if st.button("Run Hybrid Analysis"):
            if file is None:
                st.error("Upload image first.")
            else:
                img = transform(image).unsqueeze(0).to(device)

                # pad to 42
                full = features + [0.0] * (42 - len(features))
                rad = torch.tensor(full, dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    prob = torch.sigmoid(pytorch_model(img, rad)).item()

                if prob > 0.5:
                    st.error(f"⚠️ Malignant ({prob*100:.2f}%)")
                else:
                    st.success(f"✅ Benign ({(1-prob)*100:.2f}%)")