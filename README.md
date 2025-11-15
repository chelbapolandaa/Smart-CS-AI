🤖 Smart CS AI
<div align="center">
*AI-Powered Customer Service dengan 96% Accuracy • Multi-Intent Detection • Real-time Processing*

https://img.shields.io/badge/Python-3.9+-blue?logo=python
https://img.shields.io/badge/FastAPI-0.104+-green?logo=fastapi
https://img.shields.io/badge/Streamlit-1.28+-red?logo=streamlit
https://img.shields.io/badge/Accuracy-96%2525-brightgreen
https://img.shields.io/badge/License-MIT-yellow

Revolutionizing customer service dengan AI canggih untuk klasifikasi intent dan resolusi multi-masalah dalam satu interaksi.

Demo • Installation • Features • API

</div>

🎯 Overview
Smart CS AI adalah sistem customer service otomatis yang menggunakan machine learning untuk memahami dan merespons pertanyaan pelanggan dengan akurasi 96%. Sistem ini mampu mendeteksi 2-4 masalah sekaligus dan memberikan respons alami yang profesional.

✨ Highlights
🎯 96% Accuracy - Klasifikasi intent terbaik di industri

🔗 Multi-Intent Detection - Handle beberapa masalah sekaligus

🔍 Entity Extraction - Deteksi invoice, produk, urgency

⚡ Real-time - Response <1 detik

💬 Natural Responses - Bahasa Indonesia yang natural

🚀 Quick Start
Prerequisites
Python 3.9+
pip

Installation & Run
# 1. Clone repository
git clone https://github.com/chelbapolandaa/Smart-CS-AI.git
cd Smart-CS-AI

# 2. Setup virtual environment
python -m venv smart_cs_env
source smart_cs_env/bin/activate  # Windows: .\smart_cs_env\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Jalankan Streamlit UI (Recommended)
streamlit run streamlit_app.py
# ➡️ Buka: http://localhost:8501

# 5. Atau jalankan API saja
uvicorn app:app --reload
# ➡️ API Docs: http://localhost:8000/docs

🎮 Usage Examples
Single Intent
"status pesanan INV123456 belum update"
"barang datang rusak parah"
"cara return produk yang cacat"
"kurir telat 3 jam"
"ada diskon untuk member baru?"

Multi-Intent (Advanced)
"pesanan telat dan barang rusak mau refund" → 3 intents
"produk cacat + kurir kasar + minta duit tambahan" → 3 intents  
"status order gak update dan ada diskon ga?" → 2 intents
"barang elektronik mati total mau klaim garansi" → 2 intents

📊 API Documentation
Predict Endpoint
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"pesanan telat dan barang rusak parah\"}"

Response:
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

🏗️ System Architecture
User Query 
    ↓
Streamlit UI (Frontend)
    ↓
FastAPI Backend
    ↓
Intent Classification (96% Accuracy)
    ↓
Multi-Intent Detection (2-4 Issues) 
    ↓
Entity Extraction (Invoice, Products, Urgency)
    ↓
Natural Response Generation
    ↓
Professional Response

🛠️ Tech Stack
Backend & ML

FastAPI - High-performance API framework

Scikit-learn - Machine learning algorithms

TF-IDF + Logistic Regression - 96% accuracy model

Pandas & NumPy - Data processing

Frontend

Streamlit - Interactive web interface

Deployment

Docker - Containerization

Hugging Face Spaces - Free hosting

🌐 Live Demo
Experience the AI in action:

🔗 Hugging Face Spaces Demo

Coba query kompleks seperti: "pesanan telat barang rusak mau refund dan kurir kasar"

🤝 Contributing
Contributions are welcome!
Fork the project
Create your feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request

📄 License
Distributed under the MIT License. See LICENSE for more information.

👨‍💻 Author
Chelbapolandaa

GitHub: @chelbapolandaa

Project: Smart-CS-AI

🙏 Acknowledgments
Scikit-learn team for excellent ML tools

FastAPI for high-performance web framework

Streamlit for rapid web app development

Hugging Face for free model hosting

<div align="center">
⭐ Don't forget to star this repository if you find it useful!
Built with ❤️ using Python, FastAPI, Streamlit, and Scikit-learn

"Mengubah customer service dengan AI yang cerdas dan efisien"

</div>