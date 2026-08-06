import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from models.classifier import ComplaintsClassifier
from tools.rag_tools import get_ansewr
from agents.agent import run_complaint_crew
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComplaintRequest(BaseModel):
    text: str

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_bath = os.path.join(base_dir, "saved_model")
app = FastAPI()

try:
    agent = ComplaintsClassifier(model_path=model_bath)
except Exception as e:
    logger.error(f"Failed to load classifier model: {e}")
    agent = None

@app.get("/", response_class=HTMLResponse)
def home():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Complaint Classifier & Multi-Agent Routing</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f4f6f9;
                margin: 0;
                padding: 40px 20px;
                display: flex;
                justify-content: center;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 12px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                max-width: 650px;
                width: 100%;
            }
            h2 {
                color: #2c3e50;
                margin-bottom: 20px;
                text-align: center;
            }
            textarea {
                width: 100%;
                height: 110px;
                padding: 12px;
                border: 1px solid #ccc;
                border-radius: 8px;
                font-size: 14px;
                resize: vertical;
                box-sizing: border-box;
            }
            button {
                width: 100%;
                background-color: #007bff;
                color: white;
                border: none;
                padding: 12px;
                margin-top: 15px;
                border-radius: 8px;
                font-size: 16px;
                cursor: pointer;
                transition: background 0.3s;
            }
            button:hover {
                background-color: #0056b3;
            }
            .result-card {
                margin-top: 25px;
                padding: 20px;
                border-radius: 8px;
                background-color: #f8f9fa;
                border-left: 5px solid #007bff;
                display: none;
            }
            .result-card.rejected {
                border-left-color: #dc3545;
                background-color: #fff5f5;
            }
            .result-item {
                margin-bottom: 12px;
                line-height: 1.5;
            }
            .result-item strong {
                color: #333;
            }
            .routing-badge {
                background-color: #e3f2fd;
                color: #0d47a1;
                padding: 10px 15px;
                border-radius: 6px;
                font-weight: bold;
                border: 1px solid #bbdefb;
                margin-bottom: 15px;
                display: none;
            }
            .chat-box {
                margin-top: 15px;
                background: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 15px;
            }
            .user-msg {
                background-color: #e9ecef;
                padding: 10px;
                border-radius: 6px;
                margin-bottom: 10px;
            }
            .agent-msg {
                background-color: #d1e7dd;
                color: #0f5132;
                padding: 12px;
                border-radius: 6px;
                white-space: pre-line;
            }
            .loading {
                text-align: center;
                display: none;
                margin-top: 15px;
                color: #666;
            }
        </style>
    </head>
    <body>

    <div class="container">
        <h2>Complaint Classification Portal</h2>
        <label for="complaintText"><strong>Enter Complaint Text:</strong></label>
        <textarea id="complaintText" placeholder="Type your complaint here..."></textarea>
        
        <button onclick="analyzeComplaint()">Analyze Complaint</button>
        
        <div id="loading" class="loading">Analyzing complaint and contacting specialized agent...</div>

        <div id="resultCard" class="result-card">
            <div id="routingNotice" class="routing-badge"></div>

            <div class="result-item"><strong>Status:</strong> <span id="status">-</span></div>
            <div class="result-item"><strong>Category:</strong> <span id="category">-</span></div>
            <div class="result-item"><strong>Confidence:</strong> <span id="confidence">-</span></div>
            <div class="result-item"><strong>Action:</strong> <span id="action">-</span></div>

            <!-- Agent Resolution Chat Box -->
            <div id="agentChat" class="chat-box" style="display: none;">
                <div class="result-item"><strong>💬 User Complaint:</strong></div>
                <div id="userQuery" class="user-msg"></div>

                <div class="result-item"><strong>🤖 Agent Solution:</strong></div>
                <div id="ragAnswer" class="agent-msg"></div>
            </div>
        </div>
    </div>

    <script>
        async function analyzeComplaint() {
            const text = document.getElementById("complaintText").value.trim();
            if (!text) {
                alert("Please type your complaint first.");
                return;
            }

            document.getElementById("loading").style.display = "block";
            const resultCard = document.getElementById("resultCard");
            const agentChat = document.getElementById("agentChat");
            const routingNotice = document.getElementById("routingNotice");

            resultCard.style.display = "none";
            agentChat.style.display = "none";
            routingNotice.style.display = "none";
            resultCard.classList.remove("rejected");

            try {
                const response = await fetch('/classify', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text: text })
                });

                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({}));
                    throw new Error(errorData.detail || `Server returned status ${response.status}`);
                }

                const data = await response.json();

                if (data.status === "rejected") {
                    resultCard.classList.add("rejected");
                    document.getElementById("status").innerText = "Rejected ❌";
                    document.getElementById("category").innerText = data.predicted_category || "N/A";
                    document.getElementById("confidence").innerText = "0%";
                    document.getElementById("action").innerText = data.suggested_action || "None";
                } else {
                    document.getElementById("status").innerText = "Success ✅";
                    document.getElementById("category").innerText = data.predicted_category || "N/A";
                    document.getElementById("confidence").innerText = data.confidence_score ? (data.confidence_score * 100).toFixed(2) + "%" : "N/A";
                    document.getElementById("action").innerText = data.suggested_action || "None";

                    // تعديل الشرط ليشمل أي فئة تم توجيهها ولديها إجابة RAG
                    if (data.routed_to && data.rag_answer) {
                        routingNotice.innerText = `🔄 Notice: You have been routed to the ${data.routed_to} Specialist Agent.`;
                        routingNotice.style.display = "block";

                        document.getElementById("userQuery").innerText = text;
                        document.getElementById("ragAnswer").innerText = data.rag_answer;
                        agentChat.style.display = "block";
                    }
                }
                resultCard.style.display = "block";

            } catch (error) {
                console.error("Error details:", error);
                alert(`Error: ${error.message}`);
            } finally {
                document.getElementById("loading").style.display = "none";
            }
        }
    </script>

    </body>
    </html>
    """
    return html_content


CATEGORY_ROUTING_MAP = {
    "Debt collection": "Route to Debt Collection Resolution Team",
    "Checking or savings account": "Route to Retail Banking Operations",
    "Credit reporting, credit repair services, or other personal consumer reports": "Route to Credit Bureau Escalations"
}

# --- 2. Classification & Agent Routing Endpoint ---
@app.post('/classify')
def classify_compalint(data: ComplaintRequest):
    if agent is None:
        raise HTTPException(status_code=500, detail="Classifier model is not initialized properly.")

    try:
        # 1. Classification
        result = agent._process_complaint(data.text)
        category = result.get('predicted_category', '')

        if category in CATEGORY_ROUTING_MAP.keys():
            action_team = CATEGORY_ROUTING_MAP[category]
            rag_answer = run_complaint_crew(complaint_text=data.text, category=category, action_team=action_team)
            
            result['routed_to'] = category
            result['rag_answer'] = rag_answer
            result['suggested_action'] = action_team
        else:
            result['routed_to'] = None
            result['rag_answer'] = None

        return result

    except Exception as e:
        logger.error(f"Backend processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Error: {str(e)}")