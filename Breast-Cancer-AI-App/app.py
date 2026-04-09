import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import joblib
import numpy as np

# --- 1. SET UP THE PAGE ---
st.set_page_config(page_title="AI Breast Cancer Diagnosis", page_icon="🩺", layout="wide")
st.title("🩺 AI Breast Cancer Diagnosis System")

# --- 2. DEFINE & LOAD MODELS ---

# Load the PyTorch Hybrid Model
@st.cache_resource 
def load_pytorch_model():
    # Define the architecture exactly as it was trained
    class AttentionFusion(nn.Module):
        def __init__(self, radiomic_feature_count=42):
            super(AttentionFusion, self).__init__()
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
            attn_weights = torch.softmax(attn, dim=1)
            return v * attn_weights[:, 0:1] + r * attn_weights[:, 1:2]

    class HybridBreastCancerModel(nn.Module):
        def __init__(self, radiomic_feature_count=42): 
            super(HybridBreastCancerModel, self).__init__()
            resnet = models.resnet50(weights=None)
            self.visual_extractor = nn.Sequential(*list(resnet.children())[:-1])
            self.fusion_module = AttentionFusion(radiomic_feature_count)
            self.classifier = nn.Linear(2048, 1)

        def forward(self, img, radiomics):
            v_features = self.visual_extractor(img).squeeze()
            if len(v_features.shape) == 1:
                v_features = v_features.unsqueeze(0)
            fused_features = self.fusion_module(v_features, radiomics)
            return self.classifier(fused_features)

    # Initialize and load weights
    model = HybridBreastCancerModel()
    try:
        model.load_state_dict(torch.load('adaptive_hybrid_breast_cancer_model.pth', map_location='cpu'))
        model.eval()
        return model, True
    except Exception as e:
        return None, False

# Load the Random Forest Text Model
@st.cache_resource
def load_rf_model():
    try:
        return joblib.load('breast_model.pkl'), True
    except Exception as e:
        return None, False

# Initialize models
pytorch_model, pytorch_loaded = load_pytorch_model()
rf_model, rf_loaded = load_rf_model()

# Image Transformation Pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if pytorch_loaded:
    pytorch_model = pytorch_model.to(device)


# --- 3. UI TABS ---
st.write("### Select Data Input Method")
tab1, tab2, tab3 = st.tabs(["🖼️ Image Scan (PyTorch)", "📝 Clinical Data (Random Forest)", "🤝 Hybrid Mode"])

# ----------------------------------------
# TAB 1: IMAGE ONLY (PyTorch)
# ----------------------------------------
with tab1:
    st.info("Upload a medical scan to analyze visual features.")
    
    if not pytorch_loaded:
        st.error("PyTorch model file (`adaptive_hybrid_breast_cancer_model.pth`) not found. Please ensure it is in the same directory as this script.")
    else:
        uploaded_file = st.file_uploader("Upload a mammogram or ultrasound image...", type=["jpg", "jpeg", "png"], key="img_only")
        
        if uploaded_file is not None:
            col1, col2 = st.columns([2, 1], gap="medium")
            image = Image.open(uploaded_file).convert('RGB')
            
            with col1:
                st.image(image, caption='Uploaded Scan', width=500)
            
            with col2:
                if st.button("Run Image Analysis"):
                    # Prepare Image
                    img_tensor = transform(image).unsqueeze(0).to(device)
                    dummy_radiomics = torch.zeros((1, 42)).to(device)
                    
                    with torch.no_grad():
                        output = pytorch_model(img_tensor, dummy_radiomics)
                        prob = torch.sigmoid(output).item()
                    
                    st.write("---")
                    st.write("### Analysis Results")
                    
                    if prob > 0.5:
                        st.error("⚠️ HIGH RISK (Malignant)")
                        st.write(f"**Confidence:** {prob * 100:.2f}%")
                    else:
                        st.success("✅ Low Risk (Benign)")
                        st.write(f"**Confidence:** {(1 - prob) * 100:.2f}%")

# ----------------------------------------
# TAB 2: TEXT / CLINICAL DATA (Random Forest)
# ----------------------------------------
with tab2:
    st.write("### Enter Clinical Measurements")
    st.info("Input the 30 standard features from the Wisconsin Breast Cancer Dataset.")
    
    if not rf_loaded:
        st.error("Random Forest model file (`breast_model.pkl`) not found. Please ensure you downloaded it from the notebook and placed it in this folder.")
    else:
        # The complete list of 30 features matching your Random Forest model
        features = [
            ("Mean Radius", 14.00, "%.2f"),
            ("Mean Texture", 19.00, "%.2f"),
            ("Mean Perimeter", 90.00, "%.2f"),
            ("Mean Area", 600.00, "%.2f"),
            ("Mean Smoothness", 0.1000, "%.4f"),
            ("Mean Compactness", 0.1000, "%.4f"),
            ("Mean Concavity", 0.1000, "%.4f"),
            ("Mean Concave Points", 0.0500, "%.4f"),
            ("Mean Symmetry", 0.1800, "%.4f"),
            ("Mean Fractal Dimension", 0.0600, "%.4f"),
            ("Radius Error", 0.4000, "%.4f"),
            ("Texture Error", 1.2000, "%.4f"),
            ("Perimeter Error", 2.8000, "%.4f"),
            ("Area Error", 40.00, "%.2f"),
            ("Smoothness Error", 0.0070, "%.4f"),
            ("Compactness Error", 0.0250, "%.4f"),
            ("Concavity Error", 0.0300, "%.4f"),
            ("Concave Points Error", 0.0100, "%.4f"),
            ("Symmetry Error", 0.0200, "%.4f"),
            ("Fractal Dimension Error", 0.0040, "%.4f"),
            ("Worst Radius", 16.00, "%.2f"),
            ("Worst Texture", 25.00, "%.2f"),
            ("Worst Perimeter", 107.00, "%.2f"),
            ("Worst Area", 880.00, "%.2f"),
            ("Worst Smoothness", 0.1300, "%.4f"),
            ("Worst Compactness", 0.2500, "%.4f"),
            ("Worst Concavity", 0.2700, "%.4f"),
            ("Worst Concave Points", 0.1100, "%.4f"),
            ("Worst Symmetry", 0.2900, "%.4f"),
            ("Worst Fractal Dimension", 0.0800, "%.4f"),
        ]
        
        # Create 3 columns
        cols = st.columns(3)
        user_inputs = []
        
        # Loop through all 30 features and place them in the grid
        for i, (name, default_val, fmt) in enumerate(features):
            col_index = i % 3
            with cols[col_index]:
                val = st.number_input(name, value=default_val, format=fmt)
                user_inputs.append(val)
                
        if st.button("Run Clinical Analysis"):
            # Gather all 30 inputs into the shape the model expects
            patient_data = np.array(user_inputs).reshape(1, -1)
            
            # Index 0 is Malignant in the sklearn Wisconsin dataset
            probability = rf_model.predict_proba(patient_data)[0, 0]
            
            st.write("---")
            st.write("### Patient Diagnosis Report")
            
            # Apply the strict 85% clinic threshold
            if probability > 0.85:
                st.error("⚠️ HIGH RISK (Malignant)")
                st.write(f"**Cancer Probability:** {probability * 100:.2f}%")
                st.write("**ACTION:** Schedule biopsy immediately.")
            else:
                st.success("✅ Low Risk (Benign)")
                st.write(f"**Cancer Probability:** {probability * 100:.2f}%")
                st.write("**ACTION:** Routine checkup recommended.")

# ----------------------------------------
# TAB 3: HYBRID MODE
# ----------------------------------------
with tab3:
    st.warning("Hybrid mode is currently under construction. Future updates will combine the visual scan data with the 30 clinical features for maximum accuracy.")