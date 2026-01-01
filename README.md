
# 🫁 Pneumonia Detection AI

A Deep Learning Web Application that detects **Pneumonia** from Chest X-Ray images. This project uses a trained **InceptionV3** model for classification and **Grad-CAM** (Gradient-weighted Class Activation Mapping) to visualize exactly *where* the model is looking to make its decision.

---

## 🚀 Features
* **AI-Powered Diagnosis:** Utilizes Transfer Learning (InceptionV3) fine-tuned on medical imaging data.
* **Explainable AI (XAI):** Generates heatmap overlays (Grad-CAM) to highlight infected lung regions, making the "Black Box" transparent.
* **Web Interface:** Clean, responsive frontend built with HTML5, CSS3, and Bootstrap.
* **Secure Processing:** Backend logic handles image preprocessing and inference securely using Flask.

---

## 🛠️ Installation & Setup

Follow these steps to get the project running on your local machine.

### 1. Clone the Repository
```bash
https://github.com/SatvikO7/Pneumonia-Detect.git
cd pneumonia-detection
```

### 2. Install Dependencies

Ensure you have Python installed, then install the required libraries:
```bash
pip install -r requirements.txt
```

### 3. ⚠️ Download the Model (Crucial Step)

The trained model file is too large for GitHub, so it is hosted externally.

Download the inception_full_model.pth file from this link: 👉 [https://drive.google.com/file/d/1gzPEKJCRqZ_Iu0mqxyfqRz-vr5tTwM5S/view?usp=sharing]

## Move the downloaded file into the root folder of this project (the same folder where app.py is located).

### 🏃‍♂️ How to Run

Start the Flask Server Run the following command in your terminal:

```
python app.py
```
