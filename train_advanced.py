# hybrid_smart_cs_fixed.py
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer

class HybridSmartCS:
    def __init__(self):
        print("🚀 Initializing Hybrid Smart CS...")
        
        # Load single intent model
        try:
            self.single_model = joblib.load('smart_cs_model_advanced.pkl')
            print("✅ Single intent model loaded")
        except:
            print("❌ Single intent model not found")
            return
        
        # Load atau train multi-intent model
        try:
            self.multi_model = joblib.load('multi_intent_model.pkl')
            self.multi_vectorizer = joblib.load('multi_intent_vectorizer.pkl')
            self.mlb = joblib.load('multi_intent_mlb.pkl')
            print("✅ Multi-intent model loaded")
        except:
            print("❌ Multi-intent model not found, training new one...")
            self.train_multi_intent()
    
    def train_multi_intent(self):
        try:
            # Load dataset
            df = pd.read_csv('smart_cs_dataset_expanded.csv')
            print(f"📊 Loaded dataset with {len(df)} examples")
        except:
            print("❌ Dataset not found. Please run create_expanded_dataset.py first")
            return
        
        # Process untuk multi-intent
        texts = []
        all_intents = []
        
        for _, row in df.iterrows():
            if '+' in str(row['intent']):
                intents = row['intent'].split('+')
            else:
                intents = [row['intent']]
            
            texts.append(row['text'])
            all_intents.append(intents)
        
        print(f"📈 Total texts: {len(texts)}")
        print(f"🎯 Sample intents: {all_intents[:5]}")
        
        # Train multi-intent model
        self.mlb = MultiLabelBinarizer()
        y_multi = self.mlb.fit_transform(all_intents)
        
        print(f"🔢 Multi-label classes: {self.mlb.classes_}")
        
        self.multi_vectorizer = TfidfVectorizer(max_features=1500, ngram_range=(1, 2))
        X_multi = self.multi_vectorizer.fit_transform(texts)
        
        self.multi_model = MultiOutputClassifier(
            LogisticRegression(random_state=42, max_iter=1000)
        )
        self.multi_model.fit(X_multi, y_multi)
        
        # Save models
        joblib.dump(self.multi_model, 'multi_intent_model.pkl')
        joblib.dump(self.multi_vectorizer, 'multi_intent_vectorizer.pkl')
        joblib.dump(self.mlb, 'multi_intent_mlb.pkl')
        print("💾 Multi-intent models saved!")
    
    def predict_hybrid(self, text, multi_threshold=0.3):
        # Single intent prediction
        single_intent = self.single_model.predict([text])[0]
        single_confidence = self.single_model.predict_proba([text]).max()
        
        # Multi-intent prediction
        multi_input = self.multi_vectorizer.transform([text])
        multi_pred = self.multi_model.predict(multi_input)[0]
        multi_proba = self.multi_model.predict_proba(multi_input)
        
        # Get intents dengan probability above threshold
        multi_intents = []
        confidences = []
        
        for i, (pred, proba_arr) in enumerate(zip(multi_pred, multi_proba)):
            proba = proba_arr[0][1]  # Probability untuk class positive
            if proba >= multi_threshold:  # Adjust threshold di sini
                intent_name = self.mlb.classes_[i]
                multi_intents.append(intent_name)
                confidences.append(proba)
        
        # Hybrid decision logic
        if len(multi_intents) > 1:
            final_intents = multi_intents
            confidence = np.mean(confidences)
            source = "multi_intent"
        elif len(multi_intents) == 1 and multi_intents[0] != single_intent:
            # Jika multi intent berbeda dengan single intent, prefer multi
            final_intents = multi_intents
            confidence = confidences[0]
            source = "multi_intent_preferred"
        else:
            final_intents = [single_intent]
            confidence = single_confidence
            source = "single_intent"
        
        return {
            "text": text,
            "intents": final_intents,
            "confidence": round(confidence, 3),
            "source": source,
            "single_intent": single_intent,
            "single_confidence": round(single_confidence, 3),
            "multi_intents": multi_intents
        }

# Test hybrid system
if __name__ == "__main__":
    hybrid_cs = HybridSmartCS()
    
    test_queries = [
        "pesanan INV123 telat sampai dan barang rusak parah minta refund segera",
        "kurir kasar banget dan minta duit tambahan padahal ongkir sudah mahal", 
        "ada diskon buat sepatu nike dan gratis ongkir ga?",
        "produk elektronik mati total dan mau klaim garansi tapi gak tau caranya",
        "status order #TRK789 gak update dan kurirnya gak bisa dihubungi sama sekali"
    ]
    
    print("\n🧪 HYBRID SMART CS TESTING:")
    for query in test_queries:
        try:
            result = hybrid_cs.predict_hybrid(query)
            print(f"\n📝 Query: {query}")
            print(f"🎯 Final Intents: {result['intents']} (conf: {result['confidence']})")
            print(f"🔧 Source: {result['source']}")
            print(f"📊 Single: {result['single_intent']} (conf: {result['single_confidence']})")
            print(f"🔗 Multi: {result['multi_intents']}")
            print("─" * 60)
        except Exception as e:
            print(f"❌ Error processing '{query}': {e}")