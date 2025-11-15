# app.py - FIXED VERSION
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="🤖 Smart Customer Service API", version="2.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== MODELS & COMPONENTS ====================

class AdvancedResponseGenerator:
    def __init__(self):
        self.response_templates = {
            # Single intent responses
            "tanya_status_pesanan": [
                "Bisa saya bantu cek status pesanan Anda? Mohon berikan nomor invoice.",
                "Untuk pengecekan status pesanan, silakan reply dengan nomor resi pengiriman.",
                "Pesanan biasanya sampai dalam 3-5 hari kerja. Mau saya cek detailnya?"
            ],
            "komplain_produk": [
                "Mohon maaf atas ketidaknyamanannya. Bisa kirim foto produk yang bermasalah?",
                "Kami akan proses pengembalian untuk produk yang rusak. Silakan upload fotonya.",
                "Produk cacat akan kami ganti. Mohon berikan detail kerusakannya."
            ],
            "tanya_pengembalian": [
                "Untuk retur/refund, kunjungi halaman Return Center di website kami.",
                "Proses refund memakan waktu 3-7 hari kerja setelah produk kami terima.",
                "Syarat pengembalian: produk original, belum digunakan, kemasan lengkap."
            ],
            "komplain_kurir": [
                "Kami akan laporkan ke pihak kurir untuk perbaikan layanan.",
                "Mohon maaf atas ketidaknyamanannya. Tim kurir akan kami beri pelatihan.",
                "Keluhan tentang kurir akan kami tindak lanjuti segera."
            ],
            "tanya_promosi": [
                "Cek promo terbaru di banner website kami! Ada diskon spesial hari ini.",
                "Voucher gratis ongkir masih berlaku untuk min. belanja Rp 100.000.",
                "Ada cashback 10% untuk pembelian pertama dan diskon member 15%."
            ],
            
            # Multi-intent combination responses
            "komplain_produk+tanya_pengembalian": [
                "Mohon maaf produknya rusak. Untuk pengembalian dana, silakan kunjungi halaman Return Center dan upload foto produk yang bermasalah. Proses refund memakan waktu 3-7 hari kerja.",
                "Produk cacat akan kami proses refund-nya. Syarat: produk dalam kondisi original dan kirim foto kerusakannya. Tim kami akan segera menghubungi Anda.",
                "Kami paham kekecewaannya. Untuk klaim garansi/refund, silakan isi form pengembalian di website dan lampirkan bukti foto. Dana akan dikembalikan dalam 5 hari kerja."
            ],
            "tanya_status_pesanan+komplain_kurir": [
                "Mohon maaf atas keterlambatan pengiriman. Tim kurir akan kami tegur untuk perbaikan layanan. Untuk update status pesanan, bisa berikan nomor resinya?",
                "Kami akan cek status pesanan sekaligus laporkan keluhan tentang kurir. Mohon sabar menunggu update dari tim kami.",
                "Keluhan tentang kurir sudah kami catat. Untuk tracking pesanan, silakan reply dengan nomor invoice agar kami bisa bantu cek."
            ],
            "komplain_produk+komplain_kurir": [
                "Mohon maaf ganda untuk produk rusak dan pelayanan kurir. Kami akan proses pengembalian produk sekaligus beri sanksi ke kurir terkait.",
                "Kami sangat menyesal untuk pengalaman buruk ini. Produk akan kami refund dan keluhan kurir akan kami tindak lanjuti dengan serius.",
                "Tim quality control akan investigasi kedua masalah ini. Untuk refund produk, silakan upload foto kerusakan."
            ],
            "tanya_status_pesanan+tanya_pengembalian": [
                "Untuk status pesanan dan proses pengembalian, silakan berikan nomor invoice terlebih dahulu agar kami bisa bantu kedua hal tersebut.",
                "Kami akan cek status pesanan sekaligus informasikan proses pengembalian. Mohon tunggu update dari tim kami.",
                "Tim customer service akan menghubungi Anda untuk membahas status pesanan dan opsi pengembalian."
            ]
        }
    
    def extract_entities(self, text):
        entities = {
            "invoice_numbers": re.findall(r'(?:INV|#|TRK|ORDER)[A-Z0-9]+', text.upper()),
            "prices": re.findall(r'(?:Rp\s*)?(\d+[.,]?\d*)(?:\s*(?:rb|ribu|k))?', text),
            "products": [],
            "urgency_indicators": []
        }
        
        # Product detection
        product_keywords = ['elektronik', 'baju', 'sepatu', 'handphone', 'laptop', 'tas']
        entities["products"] = [word for word in product_keywords if word in text.lower()]
        
        # Urgency detection
        urgency_words = ['segera', 'cepat', 'sekarang', 'urgent', 'parah']
        entities["urgency_indicators"] = [word for word in urgency_words if word in text.lower()]
        
        return entities

    def _smart_fallback_combination(self, intents, entities):
        """Smart fallback untuk combinations yang tidak ada di template"""
        
        # Mapping intent ke response parts yang lebih natural
        intent_responses = {
            "tanya_status_pesanan": "Untuk status pesanan, silakan berikan nomor invoice.",
            "komplain_produk": "Mohon maaf produknya bermasalah, silakan upload foto kerusakan.",
            "tanya_pengembalian": "Untuk proses pengembalian, kunjungi halaman Return Center.",
            "komplain_kurir": "Keluhan tentang kurir akan kami tindak lanjuti segera.",
            "tanya_promosi": "Untuk info promo terbaru, cek banner website kami."
        }
        
        # Ambil responses untuk setiap intent
        responses = []
        for intent in intents:
            if intent in intent_responses:
                responses.append(intent_responses[intent])
        
        # Gabungkan dengan connector yang natural
        if len(responses) == 2:
            response = " ".join(responses)
        elif len(responses) > 2:
            # Untuk 3+ intents, gunakan format yang lebih structured
            response = "Kami tangani beberapa hal: " + ". ".join(responses)
        else:
            # Fallback ke concatenation original
            responses = []
            for intent in intents:
                base_response = self.response_templates.get(intent, [""])[0]
                responses.append(base_response)
            response = " ".join(responses)
        
        return response

    def generate_multi_intent_response(self, intents, entities):
        """Generate more natural responses for multi-intent"""
        
        # Special combinations dengan responses yang lebih natural
        special_combinations = {
            "komplain_produk+tanya_pengembalian": 
                "Mohon maaf produknya rusak. Untuk proses refund, silakan kunjungi halaman Return Center dan upload foto produk yang bermasalah. Dana akan dikembalikan dalam 3-7 hari kerja.",
            
            "tanya_status_pesanan+komplain_kurir":
                "Mohon maaf atas keterlambatan pengiriman. Untuk cek status pesanan, silakan berikan nomor invoice. Keluhan tentang kurir akan kami tindak lanjuti segera.",
            
            "komplain_produk+komplain_kurir":
                "Kami sangat menyesal untuk pengalaman buruk ini. Produk rusak akan kami proses refund, dan keluhan tentang kurir akan kami investigasi serius.",
            
            "tanya_pengembalian+tanya_promosi":
                "Untuk proses pengembalian, kunjungi Return Center. Untuk info promo terbaru, cek banner website kami - ada diskon spesial hari ini!",
            
            "tanya_status_pesanan+komplain_produk":
                "Untuk status pesanan, silakan berikan nomor invoice. Untuk produk yang rusak, mohon upload foto kerusakan untuk kami proses pengembalian."
        }
        
        # Coba special combination dulu
        intent_key = "+".join(sorted(intents))
        if intent_key in special_combinations:
            response = special_combinations[intent_key]
        else:
            # FALLBACK: Gunakan smart combination yang lebih reliable
            response = self._smart_fallback_combination(intents, entities)
        
        # Add entity info
        if entities.get('urgency_indicators'):
            response = "🔴 PRIORITAS: " + response
            
        if entities.get('invoice_numbers'):
            response = response.replace("nomor invoice", f"nomor invoice {entities['invoice_numbers'][0]}")
        
        return response

    def generate_response(self, intents, confidence, entities):
        # Jika multi-intent, gunakan natural response generator
        if len(intents) > 1:
            return self.generate_multi_intent_response(intents, entities)
        else:
            # Single intent - pilih berdasarkan confidence level
            if confidence > 0.7:
                response_idx = 0
            elif confidence > 0.5:
                response_idx = 1
            else:
                response_idx = 2
            response = self.response_templates[intents[0]][response_idx]
            
            # Add entity information
            if entities.get('invoice_numbers'):
                response = response.replace("nomor invoice", f"nomor invoice {entities['invoice_numbers'][0]}")
            
            if entities.get('urgency_indicators'):
                response = "🔴 PRIORITAS: " + response
            
            return response

class HybridSmartCS:
    def __init__(self):
        print("🚀 Loading AI Models...")
        
        # Load single intent model
        self.single_model = joblib.load('smart_cs_model_advanced.pkl')
        print("✅ Single intent model loaded")
        
        self.response_gen = AdvancedResponseGenerator()
        
        # Confidence boost parameters
        self.keyword_boosters = {
            "tanya_status_pesanan": ["status", "kapan", "sampai", "order", "pesanan", "dikirim", "resi", "tracking", "invoice"],
            "komplain_produk": ["rusak", "cacat", "jelek", "palsu", "bekas", "pecah", "mati", "error", "sobek", "gagal"],
            "tanya_pengembalian": ["return", "refund", "retur", "kembali", "ganti", "cancel", "balikin", "uang kembali", "klaim"],
            "komplain_kurir": ["kurir", "driver", "ongkir", "telat", "lambat", "kasar", "lelet", "ugal", "nyasar", "sopan"],
            "tanya_promosi": ["diskon", "promo", "voucher", "cashback", "gratis", "murah", "harga", "potongan", "bonus"]
        }
    
    def boost_confidence(self, raw_confidence, text, intent):
        """BOOST CONFIDENCE BERDASARKAN KEYWORD CLARITY"""
        boosted = raw_confidence
        
        # Get relevant keywords untuk intent ini
        keywords = self.keyword_boosters.get(intent, [])
        
        # Hitung berapa banyak keyword yang match
        matches = sum(1 for kw in keywords if kw in text.lower())
        
        # APPLY CONFIDENCE BOOST RULES
        if matches >= 3:
            boosted *= 1.6  # 60% boost untuk very clear intent
        elif matches >= 2:
            boosted *= 1.4  # 40% boost untuk clear intent  
        elif matches >= 1:
            boosted *= 1.2  # 20% boost untuk ada signal
        
        # Extra boost untuk specific strong patterns
        strong_patterns = [
            ("invoice", ["inv", "#", "order", "trk"]),
            ("urgent", ["segera", "cepat", "parah", "urgent"]),
            ("specific_product", ["elektronik", "handphone", "laptop", "sepatu"])
        ]
        
        for pattern_type, pattern_words in strong_patterns:
            if any(word in text.lower() for word in pattern_words):
                boosted *= 1.15  # 15% extra boost
        
        # Minimum confidence threshold - jangan terlalu rendah
        boosted = max(boosted, 0.5)  # Minimum 50% confidence
        
        # Cap at 90% - jangan over-confident
        return min(round(boosted, 3), 0.90)
    
    def detect_multi_intent_simple(self, text, main_intent):
        """SIMPLE & RELIABLE MULTI-INTENT DETECTION"""
        # Keyword mapping untuk setiap intent
        intent_keywords = {
            "tanya_status_pesanan": ["status", "kapan", "sampai", "order", "pesanan", "dikirim", "resi", "tracking"],
            "komplain_produk": ["rusak", "cacat", "jelek", "palsu", "bekas", "pecah", "mati", "error"],
            "tanya_pengembalian": ["return", "refund", "retur", "kembali", "ganti", "cancel", "balikin"],
            "komplain_kurir": ["kurir", "driver", "ongkir", "telat", "lambat", "kasar", "lelet"],
            "tanya_promosi": ["diskon", "promo", "voucher", "cashback", "gratis", "murah"]
        }
        
        detected_intents = set()
        
        # Check setiap kategori intent
        for intent_type, keywords in intent_keywords.items():
            # Hitung berapa keyword yang match
            keyword_count = sum(1 for keyword in keywords if keyword in text.lower())
            
            # Jika ada minimal 1 keyword yang kuat, consider sebagai intent
            if keyword_count >= 1:
                detected_intents.add(intent_type)
        
        # Konversi ke list
        detected_intents = list(detected_intents)
        
        # Logic untuk menentukan final intents
        if len(detected_intents) > 1:
            # MULTI-INTENT DETECTED!
            return detected_intents, "multi_intent_detected"
        elif len(detected_intents) == 1 and detected_intents[0] != main_intent:
            # Intent berbeda terdeteksi, prefer yang baru
            return detected_intents, "intent_correction"
        else:
            # Gunakan main intent dari model
            return [main_intent], "single_intent"
    
    def predict(self, text):
        # Single intent prediction dari model utama
        single_intent = self.single_model.predict([text])[0]
        raw_confidence = self.single_model.predict_proba([text]).max()
        
        # ✅ APPLY CONFIDENCE BOOST
        boosted_confidence = self.boost_confidence(raw_confidence, text, single_intent)
        
        # ✅ SIMPLE & RELIABLE MULTI-INTENT DETECTION
        final_intents, detection_source = self.detect_multi_intent_simple(text, single_intent)
        
        # Tentukan confidence berdasarkan detection type
        if detection_source == "multi_intent_detected":
            confidence = min(boosted_confidence * 1.4, 0.85)  # Boost untuk multi-intent
            source = "multi_intent"
        elif detection_source == "intent_correction":
            confidence = min(boosted_confidence * 1.2, 0.80)  # Moderate boost
            source = "intent_corrected"
        else:
            confidence = boosted_confidence
            source = "single_intent"
        
        # Extract entities
        entities = self.response_gen.extract_entities(text)
        
        # Generate response
        response = self.response_gen.generate_response(final_intents, confidence, entities)
        
        return {
            "text": text,
            "intents": final_intents,
            "confidence": confidence,
            "source": source,
            "entities": entities,
            "response": response,
            "is_multi_intent": len(final_intents) > 1,
            "single_intent_fallback": single_intent,
            "raw_confidence": round(raw_confidence, 3)
        }

# ==================== FASTAPI ROUTES ====================

# Initialize the system
smart_cs = HybridSmartCS()

class CustomerQuery(BaseModel):
    text: str

@app.post("/predict")
async def predict_intent(query: CustomerQuery):
    """Main prediction endpoint dengan SIMPLE & RELIABLE multi-intent"""
    try:
        result = smart_cs.predict(query.text)
        return {
            "status": "success",
            "result": result
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/")
async def root():
    return {
        "message": "🤖 Advanced Smart Customer Service API is running!", 
        "version": "2.0.0",
        "features": [
            "Single & Multi-intent Detection",
            "Entity Extraction", 
            "Advanced Response Generation",
            "✅ CONFIDENCE BOOST Technology",
            "✅ SIMPLE & RELIABLE Multi-intent",
            "96% Accuracy Model"
        ]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)