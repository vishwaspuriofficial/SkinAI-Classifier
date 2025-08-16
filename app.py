import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="AI Skin Cancer Classifier",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Simple black background design
st.markdown("""
<style>
    .stApp {
        background-color: #0F0F23;
        color: white;
    }
    .main > div {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 800px;
        margin: 0 auto;
    }
    .title {
        text-align: center;
        color: #ffffff;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 3rem;
    }
    .upload-section {
        background: #1a1a1a;
        border: 2px dashed #666;
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        text-align: center;
    }
    .image-container {
        background: #1a1a1a;
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        text-align: center;
    }
    .result-container {
        background: #1a1a1a;
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        text-align: center;
    }
    .prediction-card {
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        text-align: center;
        color: white;
        font-weight: bold;
    }
    .success-card {
        background: linear-gradient(135deg, #28a745, #20c997);
    }
    .warning-card {
        background: linear-gradient(135deg, #ffc107, #fd7e14);
    }
    .danger-card {
        background: linear-gradient(135deg, #dc3545, #e83e8c);
    }
    .stButton > button {
        background: linear-gradient(135deg, #007bff, #6f42c1);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 1rem 3rem;
        font-size: 1.2rem;
        font-weight: bold;
        width: 100%;
        margin: 1rem 0;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #0056b3, #5a32a3);
    }
    .chart-container {
        background: #1a1a1a;
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stFileUploader > div > div {
        background-color: #1a1a1a;
        border: 2px dashed #666;
        border-radius: 15px;
    }
    .uploadedFile {
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Model architecture classes
class EncoderDecoderCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EncoderDecoderCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),      # 0
            nn.BatchNorm2d(32),                              # 1
            nn.ReLU(inplace=True),                           # 2
            nn.MaxPool2d(2, 2),                              # 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),     # 4
            nn.BatchNorm2d(64),                              # 5
            nn.ReLU(inplace=True),                           # 6
            nn.MaxPool2d(2, 2),                              # 7
            nn.Conv2d(64, 128, kernel_size=3, padding=1),    # 8
            nn.BatchNorm2d(128),                             # 9
            nn.ReLU(inplace=True),                           # 10
            nn.MaxPool2d(2, 2)                               # 11 (removed AdaptiveAvgPool2d)
        )
        self.decoder = nn.Sequential(
            nn.Flatten(),                                    # 0
            nn.Linear(100352, 256),                          # 1 (matches saved model)
            nn.ReLU(inplace=True),                           # 2
            nn.Linear(256, num_classes)                      # 3
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class ResNet18APS(nn.Module):
    def __init__(self, num_classes=7):
        super(ResNet18APS, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.resnet.fc.in_features, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)

# Class names and risk levels
class_names = {
    0: "Actinic keratoses (akiec)",
    1: "Basal cell carcinoma (bcc)",
    2: "Benign keratosis-like lesions (bkl)",
    3: "Dermatofibroma (df)",
    4: "Melanoma (mel)",
    5: "Melanocytic nevi (nv)",
    6: "Vascular lesions (vasc)"
}

risk_levels = {
    0: ("MODERATE", "Pre-cancerous condition requiring medical attention"),
    1: ("HIGH", "Cancer requiring immediate medical treatment"),
    2: ("LOW", "Benign condition, monitor for changes"),
    3: ("LOW", "Benign fibrous tissue tumor"),
    4: ("CRITICAL", "Dangerous skin cancer requiring urgent medical care"),
    5: ("LOW", "Common benign moles, generally harmless"),
    6: ("LOW", "Benign vascular lesions")
}

# Load model function
@st.cache_resource
def load_model():
    try:
        # Load the state dict first
        checkpoint = torch.load('skin_cancer_model.pth', map_location='cpu')
        
        # Check if it's a state dict or full model
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            state_dict = checkpoint.state_dict()
        
        # Determine model architecture based on keys and structure
        if any('resnet' in key for key in state_dict.keys()):
            model = ResNet18APS(num_classes=7)
        else:
            # For EncoderDecoderCNN, check the actual structure
            model = EncoderDecoderCNN(num_classes=7)
        
        # Load with strict=False to handle minor mismatches
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Prediction function
def predict_image(image, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    
    return probabilities.numpy()

# Confidence chart function
def create_confidence_chart(probabilities):
    fig = go.Figure(data=go.Bar(
        x=probabilities,
        y=[class_names[i] for i in range(len(probabilities))],
        orientation='h',
        marker=dict(
            color=probabilities,
            colorscale='Blues',
            colorbar=dict(title="Confidence")
        )
    ))
    
    fig.update_layout(
        title="Prediction Confidence",
        xaxis_title="Confidence Score",
        yaxis_title="Skin Lesion Type",
        height=400,
        plot_bgcolor='#1a1a1a',
        paper_bgcolor='#1a1a1a',
        font=dict(color='white')
    )
    
    fig.update_xaxes(gridcolor='#444', color='white')
    fig.update_yaxes(gridcolor='#444', color='white')
    
    return fig

# Main app function
def main():
    # Title
    st.markdown('<div class="title">🔬 AI Skin Cancer Classifier</div>', unsafe_allow_html=True)
    
    st.markdown("### Upload Skin Lesion Image")
    uploaded_file = st.file_uploader("Drag and drop or click to browse", type=['png', 'jpg', 'jpeg'], label_visibility="collapsed")
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        #st.image(image, caption="Uploaded Image", width=400)
        
        # Analyze button
        if st.button("🧠 Analyze with AI", use_container_width=True):
            model = load_model()
            
            if model is not None:
                with st.spinner("Analyzing skin lesion..."):
                    # Make prediction
                    probabilities = predict_image(image, model)
                    predicted_class = np.argmax(probabilities)
                    confidence = probabilities[predicted_class]
                    
                    # Prediction card
                    risk_level, description = risk_levels[predicted_class]
                    if risk_level == "CRITICAL":
                        card_class = "danger-card"
                    elif risk_level == "HIGH":
                        card_class = "danger-card"
                    elif risk_level == "MODERATE":
                        card_class = "warning-card"
                    else:
                        card_class = "success-card"
                    
                    st.markdown(f'''
                    <div class="prediction-card {card_class}">
                        <h2>🎯 Prediction: {class_names[predicted_class]}</h2>
                        <h3>⚠️ Risk Level: {risk_level}</h3>
                        <p>{description}</p>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    # # Confidence chart
                    # fig = create_confidence_chart(probabilities)
                    # st.plotly_chart(fig, use_container_width=True)
                    # st.markdown('</div>', unsafe_allow_html=True)
                    
                    # # Medical disclaimer
                    # st.markdown("""
                    # <div style="background: #2c3e50; padding: 1rem; border-radius: 10px; margin-top: 2rem; text-align: center;">
                    #     <p style="color: #ecf0f1; margin: 0;">
                    #         ⚠️ <strong>Medical Disclaimer:</strong> This AI tool is for educational purposes only. 
                    #         Always consult a qualified dermatologist for proper medical diagnosis and treatment.
                    #     </p>
                    # </div>
                    # """, unsafe_allow_html=True)
            else:
                st.error("Failed to load the AI model. Please try again.")

if __name__ == "__main__":
    main()