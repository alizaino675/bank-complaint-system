import torch
import re
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .gibberish import detect_gibberish

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
        
        try:
            if detect_gibberish(raw_complaint) == 'clean':
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
            else:
                 return {
                        "status": "rejected",
                        "predicted_category": "Invalid / Nonsense Text",
                        "confidence_score": 0.0,
                        "suggested_action": "Ask user to retype a valid complaint",
                 }
            

        except Exception as e:
            return f"An Error has been accured: {e}."

       
    def _route_an_action(self, category: str, score: float):
        if score < 0.60:
                return "Escalate to Human Agent (Low Confidence)"
            
        routing_map = {
                "Debt collection": "Route to Debt Collection Resolution Team",
                "Checking or savings account": "Route to Retail Banking Operations",
                "Credit reporting, credit repair services, or other personal consumer reports": "Route to Credit Bureau Escalations"
            }
        return routing_map.get(category, "Route to General Support Queue")


#MODEL_DIR = "C:/Users/user/Desktop/Codes/patients/saved_model"
#agent = ComplaintsClassifier(model_path=MODEL_DIR)

#complaint_input = """ I opened a new savings account after seeing an advertisement where the bank promised a $50 bonus upon account opening. However, after opening the account, I never received the bonus. When I contacted them, they claimed there was a requirement to maintain a minimum balance for 3 months—a condition that was never disclosed before opening the account or in the initial paperwork I received. Does the bank have the right to do this under disclosure regulations?."""
#print (agent._process_complaint(complaint_input))