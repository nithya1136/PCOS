
# 🧠 PCOS Detection from Ultrasound Images using Deep Learning and Grad-CAM

This project is a web application that predicts the presence of **Polycystic Ovary Syndrome (PCOS)** from ultrasound images using a **pre-trained MobileNetV2 model**. It also provides a **Grad-CAM visualization** to highlight important regions that contributed to the prediction.

---


## 🛠 Features

- 📤 Upload ultrasound image
- 🤖 Predicts whether PCOS is detected or normal
- 🔍 Grad-CAM visualization to highlight critical image regions
- 🧠 Powered by **Transfer Learning** using MobileNetV2
- 📱 Streamlit web app interface

---

## 🧰 Tech Stack

- Python 🐍
- TensorFlow / Keras
- OpenCV
- NumPy
- Streamlit

---

## 📁 Folder Structure

```
├── pcos_mobilenetv2_model.h5         # Trained model
├── app.py                            # Main Streamlit app code
├── README.md                         # Project documentation
├── Visualisation.ipynb               # GRAD CAM image
├── EDA.ipynb                         # EDA steps
├── Model.ipynb                       # MOdel development
├── requirements.txt                  # Python dependencies
```

---

## ⚙️ Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/pcos-detection-app.git
   cd pcos-detection-app
   ```

2. **Create Virtual Environment (Optional but Recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Linux/macOS
   venv\Scripts\activate     # For Windows
   ```

3. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the App**
   ```bash
   streamlit run app.py
   ```

---

## 🧪 Sample Usage

1. Upload a `.jpg` or `.png` ultrasound image.
2. The app will process the image and predict:
   - ✅ **Normal**
   - ❗ **PCOS Detected**
3. Grad-CAM heatmap will be shown to visualize critical regions.

---

## 🤖 Why MobileNetV2?

- Lightweight and fast
- Suitable for deployment on devices with limited resources
- Leverages **transfer learning**, allowing training on small datasets with good accuracy

---

## 📚 What is Grad-CAM?

Grad-CAM (Gradient-weighted Class Activation Mapping) is a visualization technique to understand **which parts of an image contributed to a CNN's decision**.

---

## 📦 requirements.txt

```txt
streamlit
tensorflow
opencv-python
Pillow
numpy
matplotlib
```

---

## 📌 Future Improvements

- Add multi-class classification (e.g., PCOS stages)
- Add report generation functionality
- Personalised recommendation 
---

## 👩‍⚕️ Disclaimer

This app is built for **educational and research purposes only**. It is **not a substitute for professional medical diagnosis**.

---

## 🧑‍💻 Author

- Nithya Shrree S M, Saisree Anusha B, Loshini S
- B.Tech AI & Data Science  

---

## 🌟 Show your Support

If you like this project, feel free to ⭐️ this repository and share it!
