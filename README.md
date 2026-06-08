# ♻️ Waste Classifier API

<p align="center">
  <img src="https://img.shields.io/badge/Framework-TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/API-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/Container-Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white"/>
  <img src="https://img.shields.io/badge/Orchestration-Kubernetes-326CE5?style=for-the-badge&logo=kubernetes&logoColor=white"/>
</p>

> An end-to-end production machine learning API for automated waste classification, from model training through to containerised Kubernetes deployment.

---

## 🎯 Project Overview

This project demonstrates the **complete ML engineering lifecycle**: data preparation → model training → REST API → Docker containerisation → Kubernetes orchestration.

The classifier identifies waste type from images across **6 categories**, enabling automated sorting and recycling guidance in real-world applications.

---

## 🧠 Model

| Parameter | Value |
|-----------|-------|
| Architecture | 6-class Convolutional Neural Network (CNN) |
| Dataset | Kaggle Garbage Classification |
| Train/Val Split | 80/20 |
| Training Epochs | 30 |
| Batch Size | 32 |
| **Validation Accuracy** | **~90%** |

**Classes:** Cardboard · Glass · Metal · Paper · Plastic · Trash

---

## 🚀 API

Built with **FastAPI**, serving image classification predictions with sub-second latency.

**Endpoint:** `POST /predict`

```bash
curl -X POST http://localhost:30080/predict \
  -F "file=@your_image.jpg"
```

**Response:**
```json
{
  "class": "plastic",
  "confidence": 0.943,
  "all_probabilities": {
    "cardboard": 0.012,
    "glass": 0.008,
    "metal": 0.021,
    "paper": 0.009,
    "plastic": 0.943,
    "trash": 0.007
  }
}
```

---

## 🐳 Deployment

### Docker

```bash
# Build
docker build -t waste-classifier-api .

# Run
docker run -p 8080:8080 waste-classifier-api
```

| Property | Value |
|----------|-------|
| Docker Image Size | 2.47 GB |
| Exposed Port | 8080 |

### Kubernetes

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

| Property | Value |
|----------|-------|
| Service Type | NodePort |
| External Port | 30080 |

---

## 🗂️ Repository Structure

```
waste-classifier-api/
├── model/
│   ├── train.py                  # CNN training script
│   └── waste_classifier_model.h5 # Saved model weights
├── api/
│   └── main.py                   # FastAPI application
├── k8s/
│   ├── deployment.yaml           # Kubernetes deployment spec
│   └── service.yaml              # Kubernetes NodePort service
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

`TensorFlow` · `FastAPI` · `Docker` · `Kubernetes` · `Python` · `NumPy`

---

## 👩‍💻 Author

**Ramya Subramanian Porselva Bharathi**  
M.S. Web and Data Science · Universität Koblenz, Germany  
[LinkedIn](https://www.linkedin.com/in/ramya_sp) · [GitHub](https://github.com/ramyasp64)
