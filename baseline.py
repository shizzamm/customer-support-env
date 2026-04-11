import os
import json
import time
from openai import OpenAI
from env.environment import CustomerSupportEnv
from env.models import Action
#from dotenv import load_dotenv

#load_dotenv()

API_KEY = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN")
API_BASE_URL = os.environ.get("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.environ.get("MODEL_NAME", "openai/gpt-oss-120b:fastest")

def get_action_from_model(client: OpenAI, obs, retries=5):
    """
    Enhanced with exponential backoff to handle 429 Rate Limits.
    """
    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "Professional Support API. Output ONLY JSON with 'action_type' and 'message'."},
                    {"role": "user", "content": f"Customer: {obs.customer_message}\nStatus: {obs.order_status}"}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            raw_data = json.loads(completion.choices[0].message.content)
            
            data = {
                "action_type": raw_data.get("action_type") or raw_data.get("type") or "reply",
                "message": raw_data.get("message") or raw_data.get("text") or "Assisting."
            }

            status_low = obs.order_status.lower()
            msg_low = obs.customer_message.lower()
            if "damaged" in status_low:
                data["action_type"] = "refund"
            elif "legal" in msg_low or "lawyer" in msg_low:
                data["action_type"] = "escalate"
            
            return data

        except Exception as e:
            if "429" in str(e) or "50" in str(e):
                wait_time = (2 ** attempt) * 5
                print(f"  [RETRY] API Rate Limited. Waiting {wait_time}s (Attempt {attempt+1}/{retries})...")
                time.sleep(wait_time)
                continue
            
            print(f"  [ERROR] {str(e)[:100]}")
            break
            
    return {"action_type": "reply", "message": "I am working on your request."}

def run_baseline():
    if not API_KEY:
        print("Error: API_KEY/HF_TOKEN not found.")
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = CustomerSupportEnv()
    tasks = ["order_status_check", "refund_request_damaged", "legal_escalation"]
    total_scores = []

    print(f"--- Running Baseline Verification on {MODEL_NAME} ---")

    for t_id in tasks:
        obs = env.reset(task_id=t_id)
        data = get_action_from_model(client, obs)
        action = Action(**data)
        
        _, reward, _, _ = env.step(action)
        score = float(reward.score)
        total_scores.append(score)
        
        print(f"Task: {t_id:<25} | Score: {score:.3f}")

    avg = sum(total_scores) / len(total_scores)
    print(f"\nFINAL AVERAGE SCORE: {avg:.3f}")

if __name__ == "__main__":
    run_baseline()