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
st.set_page_config(page_title="AI Breast Cancer Diagnosis", page_icon="🩺", layout="wide")
st.title("🩺 AI Breast Cancer Diagnosis with Localization")

# --- 2. DEFINE THE AI ARCHITECTURE ---
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
    # Ensure this file exists in the same directory as this script
    MODEL_PATH = os.path.join(BASE_DIR, 'adaptive_hybrid_breast_cancer_model.pth')
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    
    model.eval()
    return model, device

try:
    model, device = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {e}")
    model_loaded = False

# --- 3. UI: IMAGE UPLOAD ---
uploaded_file = st.file_uploader("Upload a mammogram or ultrasound image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model_loaded:
    # Use three columns: Left scan, Right Grad-CAM, Far Right Results
    col1, col2, col3 = st.columns([2.5, 2.5, 1.5], gap="large")
    
    image = Image.open(uploaded_file).convert('RGB')
    orig_img_array = np.array(image)
    
    # --- 4. PREPARE IMAGE ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    # Enable gradients specifically for Grad-CAM visualization
    img_tensor.requires_grad = True 
    dummy_radiomics = torch.zeros((1, 42)).to(device) 
    
    # --- 5. CALCULATE GRAD-CAM (Visualizing features of the malignant class) ---
    model.zero_grad()
    
    x = img_tensor
    for i in range(8): # Loop through ResNet's sequential blocks
        x = model.deep_extractor[i](x)
    
    features = x
    features.retain_grad()
    
    x = model.deep_extractor[8](x) # The final AvgPool layer
    flattened = torch.flatten(x, 1)
    fused = model.fusion_module(flattened, dummy_radiomics)
    output = model.classifier(fused)
    
    prob = torch.sigmoid(output).item()
    
    # Calculate gradients with respect to the output score
    output.backward() 
    
    grads = features.grad[0]
    acts = features.detach()[0]
    weights = torch.mean(grads, dim=(1, 2), keepdim=True)
    cam = torch.sum(weights * acts, dim=0).cpu().numpy()
    
    # Apply ReLU to retain only positive contributions to the decision
    cam = np.maximum(cam, 0) 
    # Normalize between 0 and 1
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-10)
    
    # --- 6. PROCESS IMAGE & CREATE CIRCLE MASK ---
    cam_resized = cv2.resize(cam, (image.width, image.height), interpolation=cv2.INTER_CUBIC)
    
    # We create a binary mask from the normalized activation map to isolate the "tumor" region
    _, circle_mask = cv2.threshold(cam_resized, 0.5, 1, cv2.THRESH_BINARY)
    circle_mask = np.uint8(circle_mask * 255)
    
    # Find contours within this high-activation mask to determine the circle's parameters
    contours, _ = cv2.findContours(circle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detection_viz_left = orig_img_array.copy()
    detection_viz_right = orig_img_array.copy()
    
    # Define circle color and thickness (White, matching reference)
    circle_color = (255, 255, 255) 
    circle_thickness = 4

    if contours:
        # Get the largest activation region
        c = max(contours, key=cv2.contourArea)
        # Find the center and radius of the minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(c)
        center = (int(x), int(y))
        
        # Enforce a small size relative to the scan if needed, or stick strictly to Grad-CAM area.
        # Here we use the calculated radius, similar to the reference image size.
        radius = int(radius * 0.9) # Slightly smaller than the max area, similar to reference look.
        
        # Draw the circle on the Left Image visualization
        cv2.circle(detection_viz_left, center, radius, circle_color, circle_thickness)
        # Draw the circle on the Right Image visualization (before heatmap is overlayed)
        cv2.circle(detection_viz_right, center, radius, circle_color, circle_thickness)
    else:
        st.warning("Could not clearly localize a focused activation region for circle annotation.")

    # --- 7. TISSUE-FOCUSED GRAD-CAM MAP ---
    gray = cv2.cvtColor(orig_img_array, cv2.COLOR_RGB2GRAY)
    _, tissue_mask = cv2.threshold(gray, 15, 1, cv2.THRESH_BINARY)
    kernel = np.ones((5,5), np.uint8)
    tissue_mask = cv2.morphologyEx(tissue_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    cam_final = cam_resized * tissue_mask
    
    # --- 8. RENDER OVERLAY (Using the image that already has the circle drawn) ---
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_final), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Apply alpha blending: Only visible activation regions blend the heatmap
    alpha = np.expand_dims(cam_final, axis=2) * 0.5
    overlay = (detection_viz_right * (1 - alpha) + heatmap * alpha).astype(np.uint8)

    # --- 9. UI: DISPLAY IMAGES ---
    with col1:
        st.image(detection_viz_left, caption='Original Scan', width=500)
    with col2:
        st.image(overlay, caption='Focused Grad-CAM Map', width=500)

    # --- 10. RESULTS (In the far right column) ---
    with col3:
        st.subheader("Analysis Results")
        confidence = max(prob, 1 - prob) * 100
        
        if prob >= 0.5:
            st.error("### ⚠️ Malignant")
        else:
            st.success("### ✅ Benign")
            
        st.write(f"**AI Confidence:** {confidence:.2f}%")
        st.progress(int(confidence))
        
        st.info("Note: The white circle indicates the focus area driving the Malignant classification. This is an AI tool and must be verified by a professional.")