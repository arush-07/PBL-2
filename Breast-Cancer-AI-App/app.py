import os
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

# --- 1. SET UP THE PAGE ---
st.set_page_config(page_title="Breast Cancer AI Assistant", page_icon="🩺", layout="centered")
st.title("🩺 AI Breast Cancer Diagnosis")
st.write("Upload a mammogram or ultrasound image to get an instant AI prediction and Tissue Activation Map.")

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

    device = torch.device("cpu") 
    model = HybridBreastCancerModel(radiomic_feature_count=42).to(device)
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, 'adaptive_hybrid_breast_cancer_model.pth')
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, device

try:
    model, device = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("Ensure 'adaptive_hybrid_breast_cancer_model.pth' is uploaded to the same folder.")
    model_loaded = False

# --- 3. UI: IMAGE UPLOAD ---
uploaded_file = st.file_uploader("Choose a medical image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model_loaded:
    col1, col2 = st.columns(2)
    
    image = Image.open(uploaded_file).convert('RGB')
    with col1:
        st.image(image, caption='Original Scan', use_container_width=True)
    
    st.write("🔍 Analyzing image features and generating Activation Map...")
    
    # --- 4. PREPARE IMAGE ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    dummy_radiomics = torch.zeros((1, 42)).to(device) 
    
    # --- 5. THE FOOLPROOF GRAD-CAM (Using Direct Tensor Gradients) ---
    model.zero_grad()
    
    # Step A: Forward pass through the visual backbone ONLY
    visual_features = model.deep_extractor(img_tensor)
    
    # Step B: Tell PyTorch to save the gradients for this specific 3D feature map
    visual_features.retain_grad()
    
    # Step C: Finish the forward pass manually
    flattened_features = torch.flatten(visual_features, 1)
    fused_features = model.fusion_module(flattened_features, dummy_radiomics)
    output = model.classifier(fused_features)
    
    prob = torch.sigmoid(output).item()
    
    # Step D: Backward pass (with polarity fix)
    if prob >= 0.5:
        output.backward() # Malignant
    else:
        (-output).backward() # Benign
        
    # Step E: Extract the exact activations and gradients directly from the tensor
    activations = visual_features.detach()[0] # Shape: [2048, 7, 7]
    grads = visual_features.grad.detach()[0]  # Shape: [2048, 7, 7]
    
    # --- 6. GENERATE HEATMAP ---
    # Calculate weights by averaging gradients spatially
    weights = torch.mean(grads, dim=[1, 2], keepdim=True)
    
    # Multiply activations by weights and sum across channels
    cam = torch.sum(weights * activations, dim=0).cpu().numpy()
    
    # ReLU: Keep only positive influences
    cam = np.maximum(cam, 0)
    
    # Normalize safely
    cam_min, cam_max = np.min(cam), np.max(cam)
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam)

    # --- 7. CREATE HIGH-RES OVERLAY ---
    orig_width, orig_height = image.size
    cam = cv2.resize(cam, (orig_width, orig_height), interpolation=cv2.INTER_CUBIC)
    
    # Clean up low-level noise
    cam[cam < 0.2] = 0.0  
    
    orig_img_array = np.array(image)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Dynamic alpha mask
    alpha_mask = np.expand_dims(cam ** 1.5, axis=2) * 0.65 
    overlay = (heatmap * alpha_mask) + (orig_img_array * (1 - alpha_mask))
    overlay = np.uint8(overlay)

    with col2:
        st.image(overlay, caption='Direct Tensor Grad-CAM', use_container_width=True)

    # --- 8. DISPLAY RESULTS ---
    st.markdown("---")
    confidence = max(prob, 1 - prob) * 100
    
    if prob >= 0.5:
        st.error(f"### ⚠️ Diagnosis: Malignant")
        st.write(f"**Confidence:** {confidence:.2f}%")
        st.progress(int(prob * 100))
    else:
        st.success(f"### ✅ Diagnosis: Benign / Normal")
        st.write(f"**Confidence:** {confidence:.2f}%")
        st.progress(int((1 - prob) * 100))
        
    st.caption("Disclaimer: This is an educational AI tool and not a substitute for professional medical advice.")