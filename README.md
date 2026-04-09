🚨 Anomaly Detection System (ML + API + Dashboard)

A full end-to-end Machine Learning system for detecting anomalies in transactional data using a combination of:

Isolation Forest
KMeans Clustering
Autoencoder (PyTorch)
Hybrid Prophet (per-customer + statistical fallback)
Rule-based checks
📌 Features   

✅ Multi-model anomaly detection (ensemble)
✅ Customer-level behavior modeling
✅ Hybrid Prophet optimization (fast + scalable)
✅ REST API (FastAPI)
✅ Interactive dashboard (Gradio)
✅ Deployable on Hugging Face Spaces

🧠 Architecture
User → Dashboard (Gradio UI)
        ↓
     FastAPI (Backend)
        ↓
   ML Models (Saved)
        ↓
 Anomaly Scores (Output)
📂 Project Structure
.
├── app.py                # Gradio dashboard
├── api.py                # FastAPI backend
├── train.py              # Train models
├── predict.py            # Inference pipeline
├── trains.py             # Feature engineering + models
├── requirements.txt
├── models/
    ├── iso.pkl
    ├── kmeans.pkl
    ├── scaler.pkl
    ├── encoders.pkl
    ├── prophet_per_customer.json
    ├── fallback_stats.json
    ├── ae.pth
    └── models.db
⚙️ Installation
git clone https://github.com/loony1028/anomaly-detector.git
cd anomaly-detector

python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

pip install -r requirements.txt
🏋️ Train Models
python train.py

This will:

Train ML models
Train per-customer Prophet models
Compute fallback statistics
Save everything into models/
🔮 Run Prediction (CLI)
from predict import predict

df = predict("data/raw_dataset.csv")
print(df.head())
🌐 Run API (FastAPI)
uvicorn api:app --reload

Open:

👉 http://127.0.0.1:8000/docs

📊 Run Dashboard (Gradio)
python app.py

Open:

👉 http://127.0.0.1:7860

🚀 Deployment (Hugging Face)
Create a Space (Gradio)
Upload:
app.py
predict.py
trains.py
requirements.txt
models/
Wait for build → Done
🔌 API Usage
Python Example
import requests

url = "http://127.0.0.1:8000/detect"

files = {"file": open("data/raw_dataset.csv", "rb")}

response = requests.post(url, files=files)

print(response.json())
⚡ Model Details
1. Isolation Forest

Detects global outliers

2. KMeans

Measures distance from cluster centers

3. Autoencoder

Detects reconstruction error

4. Prophet (Hybrid)
High-volume customers → Prophet
Low-volume customers → Z-score fallback
5. Rule-Based System
Duplicate detection
Price spikes
Quantity anomalies
Geo mismatch
📈 Output
Column	Description
order_id	Transaction ID
customer_id	Customer
final_score	Anomaly score
is_anomaly	True/False
scores	Detailed breakdown
⚠️ Limitations
Prophet is slower for large datasets
Hugging Face has CPU constraints
Cold start latency possible
🔥 Future Improvements
Real-time streaming (Kafka)
Model monitoring (drift detection)
GPU acceleration
API authentication
Advanced dashboard (charts)
👨‍💻 Author

Luca

⭐ If you like this project

Give it a star ⭐ and share it!