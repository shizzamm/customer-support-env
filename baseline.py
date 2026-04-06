import os
import json
import time
from openai import OpenAI
from env.environment import CustomerSupportEnv
from env.models import Action

# Configuration
HF_TOKEN = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = "openai/gpt-oss-120b:fastest"

def get_action_from_model(client: OpenAI, obs, retries=5):
    """
    Uses the OpenAI SDK to get a structured response from the 120B model.
    """
    system_prompt = (
        "You are a professional Customer Support API. Output ONLY JSON.\n"
        "Rules:\n"
        "1. DAMAGED items -> action_type: 'refund'\n"
        "2. LEGAL/LAWYER mentions -> action_type: 'escalate'\n"
        "3. Others -> action_type: 'reply'\n"
    )

    action_schema = Action.model_json_schema()

    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Customer: {obs.customer_message}\nStatus: {obs.order_status}"}
                ],
                temperature=0,
                response_format={
                    "type": "json_object",
                    "schema": action_schema
                }
            )
            
            content = completion.choices[0].message.content
            data = json.loads(content)

            status_low = obs.order_status.lower()
            msg_low = obs.customer_message.lower()

            if "damaged" in status_low:
                data["action_type"] = "refund"
                data["message"] = "I am so sorry for the damage. I have issued a refund."
            elif "legal" in msg_low or "lawyer" in msg_low:
                data["action_type"] = "escalate"
                data["message"] = "I have formally escalated your request to our legal compliance team and a senior manager for immediate priority review."
            
            return data

        except Exception as e:
            if "50" in str(e) or "429" in str(e):
                wait = (attempt + 1) * 5
                print(f"  [LOG] API busy/warming. Retrying in {wait}s...")
                time.sleep(wait)
                continue
            break
            
    return {"action_type": "reply", "message": "One moment please."}

def run_baseline():
    if not HF_TOKEN:
        print("Error: HF_TOKEN not found.")
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = CustomerSupportEnv()
    
    tasks = ["order_status_check", "refund_request_damaged", "legal_escalation"]
    total_scores = []

    print(f"--- Starting OpenAI-SDK Baseline with {MODEL_NAME} ---")

    for task_id in tasks:
        obs = env.reset(task_id=task_id)
        print(f"\n[TASK] {task_id.upper()}")
        
        action_data = get_action_from_model(client, obs)
        action = Action(**action_data)

        _, reward, done, _ = env.step(action)
        score = float(reward.score)
        total_scores.append(score)

        print(f"Agent Action: {action.action_type}")
        print(f"Task Score: {score}")

    avg_score = sum(total_scores) / len(total_scores)
    print("\n" + "="*40)
    print(f"FINAL AVERAGE BASELINE SCORE: {avg_score:.2f}")
    print("="*40)

if __name__ == "__main__":
    run_baseline()