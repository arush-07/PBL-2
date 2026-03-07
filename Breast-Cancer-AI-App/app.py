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
st.write("Upload a mammogram or ultrasound image to get an instant AI prediction and Grad-CAM visualization.")

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
    
    st.write("🔍 Analyzing image features and generating Grad-CAM...")
    
    # --- 4. PREPARE IMAGE ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # IMPORTANT: Enable gradients on the input image for CAM calculation
    img_tensor = transform(image).unsqueeze(0).to(device).requires_grad_(True)
    dummy_radiomics = torch.zeros((1, 42)).to(device) 
    
    # --- 5. GRAD-CAM HOOKS ---
    feature_blobs = []
    backward_blobs = []

    def forward_hook(module, input, output):
        feature_blobs.append(output)

    def backward_hook(module, grad_input, grad_output):
        backward_blobs.append(grad_output[0])

    target_layer = model.deep_extractor[7]
    handle_fw = target_layer.register_forward_hook(forward_hook)
    handle_bw = target_layer.register_full_backward_hook(backward_hook)
    
   # --- 6. MAKE PREDICTION & POLARITY ---
    model.zero_grad()
    output = model(img_tensor, dummy_radiomics)
    prob = torch.sigmoid(output).item()
    
    # Ask the model what features correlate with its specific decision
    if prob >= 0.5:
        output.backward() # Calculates gradients for Malignant
    else:
        (-output).backward() # Calculates gradients for Benign
    
    # --- 7. GENERATE CLINICAL-GRADE HEATMAP ---
    activations = feature_blobs[0][0].cpu().detach().numpy()
    grads = backward_blobs[0][0].cpu().detach().numpy()
    
    # Standard Grad-CAM math
    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(activations.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * activations[i, :, :]
        
    cam = np.maximum(cam, 0) # The ReLU filter: Kills all negative signals instantly
    
    # Normalize safely
    cam_min, cam_max = np.min(cam), np.max(cam)
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam)

    # Stretch to match original image using smooth Cubic Interpolation
    orig_width, orig_height = image.size
    cam = cv2.resize(cam, (orig_width, orig_height), interpolation=cv2.INTER_CUBIC)
    
    # THE NOISE FILTER: Erase all low-level background heat completely
    cam[cam < 0.3] = 0.0  
    
    handle_fw.remove()
    handle_bw.remove()
    
    # Apply colormap
    orig_img_array = np.array(image)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # EXPONENTIAL ALPHA MASK: Makes the core bright red, and completely hides the edges
    alpha_mask = np.expand_dims(cam ** 1.5, axis=2) * 0.65 
    
    overlay = (heatmap * alpha_mask) + (orig_img_array * (1 - alpha_mask))
    overlay = np.uint8(overlay)

    with col2:
        st.image(overlay, caption='Focused Grad-CAM Attention Map', use_container_width=True)

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