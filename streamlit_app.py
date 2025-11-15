# streamlit_app.py - FIXED VERSION
import streamlit as st
import requests

st.set_page_config(page_title="🤖 Smart Customer Service AI", page_icon="🤖")

st.title("🤖 Smart Customer Service AI v2.0")
st.markdown("""
**Advanced Features:**
- 🎯 Single & Multi-intent Detection (2-4 intents sekaligus!)
- 🔍 Entity Extraction (Invoice, Products, Urgency)
- 💬 Natural Response Generation 
- ⚡ 96% Accuracy Model dengan Confidence Boost
- 🔴 Auto Priority Detection
""")

# Initialize session state untuk input
if 'query_text' not in st.session_state:
    st.session_state.query_text = ""

# Input section - HANYA SATU text area
user_input = st.text_area(
    "💬 Tanyakan masalah e-commerce Anda:", 
    value=st.session_state.query_text,
    height=100,
    key="main_input"
)

# Contoh queries advanced yang KOMPLEKS
st.sidebar.title("💡 Contoh Query Kompleks")
st.sidebar.markdown("Klik contoh untuk test:")

complex_examples = [
    "pesanan telat barang rusak mau refund dan kurir kasar",  # 4 INTENTS
    "produk elektronik mati total mau klaim garansi tapi gak tau caranya",  # 2 INTENTS + urgency
    "status INV123456 belum update dan kurirnya gak bisa dihubungi sama sekali",  # 2 INTENTS + entity
    "barang datang rusak parah mau return segera!",  # 2 INTENTS + urgency
    "ada diskon buat sepatu nike dan gratis ongkir ga?",  # 2 INTENTS
    "kurir minta duit tambahan padahal ongkir sudah mahal dan barang cacat",  # 3 INTENTS
    "mau cancel order #TRK789 karena produk jelek kualitas",  # 2 INTENTS + entity
    "driver telat 3 jam dan kasar banget minta uang tambahan",  # 2 INTENTS + urgency
]

# Button untuk contoh query
for i, example in enumerate(complex_examples):
    if st.sidebar.button(example, key=f"btn_{i}"):
        st.session_state.query_text = example
        st.rerun()

# Tombol analisis
if st.button("🚀 Analisis dengan AI", type="primary"):
    if st.session_state.query_text:
        with st.spinner("🔄 AI menganalisis query kompleks..."):
            try:
                response = requests.post(
                    "http://localhost:8000/predict",
                    json={"text": st.session_state.query_text},
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if result["status"] == "success":
                        data = result["result"]
                        
                        st.success("✅ **Hasil Analisis AI Lanjutan**")
                        
                        # Display intents
                        col1, col2 = st.columns(2)
                        with col1:
                            st.info(f"**🎯 Intents:** {', '.join(data['intents'])}")
                        with col2:
                            st.info(f"**📊 Confidence:** {data['confidence']}")
                        
                        # Multi-intent badge
                        if data['is_multi_intent']:
                            st.warning(f"🔗 **MULTI-INTENT DETECTED** - {len(data['intents'])} intents handled!")
                        
                        # Source info
                        st.info(f"**🔧 Source:** {data['source']}")
                        
                        # Entities
                        if data['entities'] and any(data['entities'].values()):
                            with st.expander("🔍 **Entities Extracted**"):
                                st.json(data['entities'])
                        
                        # Response
                        st.success("💡 **Smart Response:**")
                        st.write(data['response'])
                        
                    else:
                        st.error(f"Error: {result['message']}")
                else:
                    st.error("API Error: Pastikan server FastAPI sedang berjalan!")
                    
            except Exception as e:
                st.error(f"Koneksi error: {e}")
    else:
        st.warning("⚠️ Silakan masukkan pertanyaan atau pilih contoh query!")

# Quick stats
st.sidebar.markdown("---")
st.sidebar.markdown("**📈 System Stats:**")
st.sidebar.info("""
- Model Accuracy: 96%
- Multi-intent: 2-4 queries
- Response Time: <1s
- Entities: Invoice, Products, Urgency
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🤖 <b>Advanced Smart Customer Service AI</b> | 96% Accuracy | Multi-intent Detection</p>
    <p>Dibuat dengan FastAPI + Scikit-learn + Streamlit</p>
</div>
""", unsafe_allow_html=True)