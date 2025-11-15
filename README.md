# 🤖 Smart CS AI

<div align="center">


*AI-Powered Customer Service dengan Akurasi 96% • Deteksi Multi-Intent • Pemrosesan Real-time*

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Accuracy](https://img.shields.io/badge/Accuracy-96%25-brightgreen)](https://github.com/chelbapolandaa/Smart-CS-AI)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**Revolusi Customer Service** dengan kecerdasan buatan untuk klasifikasi intent dan resolusi multi-masalah dalam satu interaksi.

[🚀 **Demo**](#-live-demo) • [📦 **Installasi**](#-quick-start) • [✨ **Fitur**](#-features) • [🔗 **API**](#-api-documentation)

</div>

## 📋 Tentang Proyek

Smart CS AI adalah sistem customer service otomatis yang menggunakan **machine learning** untuk memahami dan merespons pertanyaan pelanggan dengan **akurasi 96%**. Sistem ini mampu mendeteksi **2-4 masalah sekaligus** dalam satu percakapan dan memberikan respons yang natural dan profesional.

### 🎯 Keunggulan Utama

| Fitur | Deskripsi |
|-------|-----------|
| **🎯 Akurasi Tinggi** | 96% akurasi klasifikasi intent - terbaik di industri |
| **🔗 Multi-Intent** | Deteksi 2-4 masalah berbeda dalam satu kalimat |
| **🔍 Entity Recognition** | Ekstraksi otomatis: invoice, produk, tingkat urgensi |
| **⚡ Real-time** | Respons dalam <1 detik |
| **💬 Natural Language** | Respons dalam Bahasa Indonesia yang natural |

## 🚀 Mulai Cepat

### Prasyarat
- **Python 3.9** atau lebih tinggi
- **pip** package manager
- **Virtual environment** (direkomendasikan)

### 📥 Installasi

```bash
# 1. Clone repository
git clone https://github.com/chelbapolandaa/Smart-CS-AI.git
cd Smart-CS-AI

# 2. Buat virtual environment
python -m venv smartcs_env

# 3. Aktivasi virtual environment
# Untuk Windows:
smartcs_env\Scripts\activate
# Untuk Linux/Mac:
source smartcs_env/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Jalankan aplikasi
```
## 🎮 Menjalankan Aplikasi
### Opsi 1: Streamlit UI (Rekomendasi untuk Demo)
```bash
streamlit run streamlit_app.py
```
➡️ Akses di: http://localhost:8501
### Opsi 2: FastAPI Backend
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```
➡️ API Docs: http://localhost:8000/docs

## 💡 Contoh Penggunaan
### Single Intent
```bash
"status pesanan INV123456 belum update"
"barang datang rusak parah" 
"cara return produk yang cacat"
"kurir telat 3 jam"
"ada diskon untuk member baru?"
```
### Multi-Intent (Advanced)
```bash
"pesanan telat dan barang rusak mau refund"
→ Deteksi: tanya_status_pesanan, komplain_produk, minta_refund

"produk cacat + kurir kasar + minta duit tambahan"  
→ Deteksi: komplain_produk, komplain_kurir, minta_kompensasi

"status order gak update dan ada diskon ga?"
→ Deteksi: tanya_status_pesanan, tanya_promo
```

## 🔌 API Documentation
### 📤 Endpoint Predict
URL: POST /predict
Request:
```
{
  "text": "pesanan telat dan barang rusak parah"
}
```
Response:
```
{
  "status": "success",
  "result": {
    "text": "pesanan telat dan barang rusak parah",
    "intents": ["tanya_status_pesanan", "komplain_produk"],
    "confidence": 0.85,
    "source": "multi_intent",
    "entities": {
      "invoice_numbers": [],
      "products": [],
      "urgency_indicators": ["parah"]
    },
    "response": "🔴 PRIORITAS: Mohon maaf atas keterlambatan...",
    "is_multi_intent": true
  }
}
```
Contoh Penggunaan dengan cURL
```
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "pesanan saya telat dan barangnya rusak"}'
```

## 🏗️ Arsitektur Sistem
```
User Query (Input)
        ↓
Streamlit Frontend (UI)
        ↓
FastAPI Backend (REST API)
        ↓
Intent Classification Model (96% Accuracy)
        ↓
Multi-Intent Detection Engine  
        ↓
Entity Extraction Module
        ↓
Natural Language Generation
        ↓
Professional Response (Output)
```

## 🛠️ Tech Stack
### 🤖 Backend & Machine Learning
FastAPI - Framework API high-performance

Scikit-learn - Algoritma machine learning

TF-IDF + Logistic Regression - Model dengan akurasi 96%

Pandas & NumPy - Data processing dan manipulasi

Joblib - Model serialization

### 🎨 Frontend
Streamlit - Web interface yang interaktif

Plotly - Visualisasi data dan metrics

### 🚀 Deployment
Docker - Containerization

Hugging Face Spaces - Free hosting platform

## 🤝 Contribute
Contribute are Very Welcome

Fork project ini

Buat feature branch (git checkout -b feature/FiturBaru)

Commit changes (git commit -m 'Menambah fitur baru')

Push ke branch (git push origin feature/FiturBaru)

Buat Pull Request

## 👨‍💻 Author
Chelba Polanda

GitHub: @chelbapolandaa

Project: Smart-CS-AI

<div align="center">
⭐ Don't forget to star this repository if you find it useful!
Built with ❤️ using Python, FastAPI, Streamlit, and Scikit-learn

"Mengubah customer service dengan AI yang cerdas dan efisien"

</div>
