import os
import numpy as np
import torch
import cv2
from flask import Flask, request, render_template, jsonify, url_for
from PIL import Image
from torchvision import transforms, models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

#Configuration
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = 'inception_full_model.pth'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading model on {device}...")

try:
    # Try loading the entire model as saved in the notebook
    model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'inception_full_model.pth' is in the project directory.")
    # Fallback (placeholder structure if file is missing, purely for debug)
    model = models.inception_v3(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)

model.to(device)
model.eval()

# Preprocessing
# Values taken from your notebook snippet
MEAN = [0.4822, 0.4822, 0.4822]
STD = [0.2362, 0.2362, 0.2362]

val_transforms = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

# Helper Functions
def process_image(image_path):
    """Opens and prepares image for the model."""
    img = Image.open(image_path).convert('RGB')
    img_tensor = val_transforms(img).unsqueeze(0).to(device)
    return img, img_tensor

def generate_gradcam(model, input_tensor, original_image_pil, target_class_idx):
    """Generates a Grad-CAM heatmap overlay."""
    # InceptionV3 specific target layer (Mixed_7c is the last conv layer)
    target_layers = [model.Mixed_7c]
    
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(target_class_idx)]
    
    # Generate CAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    # Prepare original image for visualization (resize to 299x299 and normalize 0-1)
    img_resized = original_image_pil.resize((299, 299))
    rgb_img = np.float32(img_resized) / 255
    
    # Overlay heatmap on image
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return visualization

# Routes
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    if file:
        # Save original file
        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Preprocess
        try:
            pil_img, img_tensor = process_image(filepath)
            
            # Inference
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probs, 1)
                
            # Mapping (Assuming 0: NORMAL, 1: PNEUMONIA based on alphabetical order)
            classes = ['NORMAL', 'PNEUMONIA']
            result_class = classes[predicted_class.item()]
            result_conf = round(confidence.item() * 100, 2)
            
            # Generate Grad-CAM
            # Create a unique name for the heatmap
            cam_filename = f"gradcam_{filename}"
            cam_path = os.path.join(UPLOAD_FOLDER, cam_filename)
            
            visualization = generate_gradcam(model, img_tensor, pil_img, predicted_class.item())
            # Save visualization using OpenCV
            cv2.imwrite(cam_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
            
            return jsonify({
                'class': result_class,
                'confidence': result_conf,
                'original_url': url_for('static', filename=f'uploads/{filename}'),
                'gradcam_url': url_for('static', filename=f'uploads/{cam_filename}')
            })
            
        except Exception as e:
            return jsonify({'error': str(e)})

# VALIDATION ROUTES (Remove this section if not needed)

@app.route('/validate', methods=['GET'])
def validate_page():
    """Serves the validation/testing page."""
    return render_template('validate.html')

@app.route('/validate', methods=['POST'])
def validate_prediction():
    """Tests a prediction against a known label."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    true_label = request.form.get('true_label', '').upper()
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if true_label not in ['NORMAL', 'PNEUMONIA']:
        return jsonify({'error': 'Invalid true label'})

    if file:
        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        try:
            pil_img, img_tensor = process_image(filepath)
            
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probs, 1)
            
            classes = ['NORMAL', 'PNEUMONIA']
            prediction = classes[predicted_class.item()]
            conf_percent = round(confidence.item() * 100, 2)
            is_correct = (prediction == true_label)
            
            return jsonify({
                'prediction': prediction,
                'confidence': conf_percent,
                'true_label': true_label,
                'is_correct': is_correct
            })
            
        except Exception as e:
            return jsonify({'error': str(e)})

# END OF VALIDATION ROUTES

if __name__ == '__main__':
    app.run(debug=True)