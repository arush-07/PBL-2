import os
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image

# --- 1. SET UP THE PAGE ---
st.set_page_config(page_title="Breast Cancer AI Assistant", page_icon="🩺", layout="centered")
st.title("🩺 AI Breast Cancer Diagnosis")
st.write("Upload a mammogram or ultrasound image to get an instant AI prediction.")

# --- 2. DEFINE THE ADVANCED AI ARCHITECTURE ---
@st.cache_resource 
def load_model():
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
            self.deep_extractor = nn.Sequential(*list(resnet.children())[:-1])
            self.fusion_module = AttentionFusion(radiomic_feature_count)
            self.classifier = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 1)
            )
            
        def forward(self, image, radiomics):
            visual_features = self.deep_extractor(image)
            visual_features = torch.flatten(visual_features, 1)
            fused_features = self.fusion_module(visual_features, radiomics)
            return self.classifier(fused_features)

    # Force CPU for stable cloud deployment
    device = torch.device("cpu") 
    model = HybridBreastCancerModel(radiomic_feature_count=42).to(device)
    
    # --- BULLETPROOF PATH FIX ---
    # This forces Streamlit to look in the exact same folder as this app.py script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, 'adaptive_hybrid_breast_cancer_model.pth')
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, device

# Load model safely with error handling displayed on the web page
try:
    model, device = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("Please ensure 'adaptive_hybrid_breast_cancer_model.pth' is uploaded to the same GitHub folder as this app.py file.")
    model_loaded = False

# --- 3. UI: IMAGE UPLOAD ---
uploaded_file = st.file_uploader("Choose a medical image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model_loaded:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Scan', use_container_width=True)
    st.write("🔍 Analyzing image features...")
    
    # --- 4. PREPARE IMAGE ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Dummy radiomics tensor to satisfy the 42-feature requirement without pyradiomics
    dummy_radiomics = torch.zeros((1, 42)).to(device) 
    
    # --- 5. MAKE PREDICTION ---
    with torch.no_grad():
        output = model(img_tensor, dummy_radiomics)
        prob = torch.sigmoid(output).item()
        
    # --- 6. DISPLAY RESULTS ---
    st.markdown("---")
    confidence = max(prob, 1 - prob) * 100
    
    if prob >= 0.5:
        st.error(f"### ⚠️ Diagnosis: Malignant (High Probability)")
        st.write(f"**Confidence:** {confidence:.2f}%")
        st.progress(int(prob * 100))
    else:
        st.success(f"### ✅ Diagnosis: Benign / Normal")
        st.write(f"**Confidence:** {confidence:.2f}%")
        st.progress(int((1 - prob) * 100))
        
    st.caption("Disclaimer: This is an educational AI tool and not a substitute for professional medical advice.")