import torch
import re
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nort

class ComplaintsClassifier():
    def __init__(self, model_path):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.label_encoder = joblib.load("C:/Users/user/Desktop/Codes/patients/saved_model/label_encoder.joblib")
        self.model.eval()

    def _clean_text(self, text: str):
        text = text.lower()
        text = re.sub(r'x{2,}/x{2,}/x{4,}', '', text)
        text = re.sub(r'xx/xx/xxxx', '', text)
        text = re.sub(r'\bx{2,}\b', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        return re.sub(r'\s+', ' ', text).strip()
    
    def _process_complaint(self, raw_complaint: str):
        clean_text = self._clean_text(raw_complaint)

        tokeinzed_text = self.tokenizer(clean_text, return_tensors='pt', truncation=True, padding=True)

        with torch.no_grad():
            output = self.model(**tokeinzed_text)
            prob = torch.nn.functional.softmax(output.logits, dim=-1)
            conf, pred_idx = torch.max(prob, dim=1)

        category = self.label_encoder.inverse_transform([pred_idx.item()])[0]
        score = conf.item()

        action = self._route_an_action(category, score)
        return {
            "status": "success",
            "predicted_category": category,
            "confidence_score": round(score, 4),
            "suggested_action": action
        }
    def _route_an_action(self, category: str, score: float):

        if score < 0.60:
                return "Escalate to Human Agent (Low Confidence)"
            
            # توجيه تلقائي للأقسام بناءً على التصنيف
        routing_map = {
                "Credit card or prepaid card": "Route to Credit Card Operations Team",
                "Mortgage": "Route to Mortgage Support Department",
                "Checking or savings account": "Route to Retail Banking Operations",
                "Credit reporting, credit repair services, or other personal consumer reports": "Route to Credit Bureau Escalations"
            }
        return routing_map.get(category, "Route to General Support Queue")


# تحميل الأيجينت من المجلد الذي حفظت فيه النموذج
# 🟢 استخدام Slashing عادية
MODEL_DIR = "C:/Users/user/Desktop/Codes/patients/saved_model"
agent = ComplaintsClassifier(model_path=MODEL_DIR)

# تجربة الشكوى
complaint_input = """ I am writing to express my extreme frustration regarding my savings account ending in 4092. 
On July 15th, a monthly maintenance fee of $25 was deducted from my account, even though 
my daily balance stayed well above the required $1,000 minimum threshold for the entire month. 
I contacted customer support twice, but the representatives were unhelpful and refused to refund the charge. 
This is an unauthorized deduction and a clear breach of our account agreement. 
I demand an immediate refund of $25 and an update to my account status."""
response = agent._process_complaint(complaint_input)

print(response)