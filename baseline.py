import os
import json
import requests
import time
from env.environment import CustomerSupportEnv
from env.models import Action

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_URL = "https://router.huggingface.co/v1/chat/completions"

def get_action_from_model(obs, task_id, retries=3):
    """
    Connects to the HF router using the OPENAI_API_KEY.
    """
    system_prompt = (
        "You are a professional Customer Support API. You must output ONLY JSON.\n"
        "Rules:\n"
        "1. If item is DAMAGED/BROKEN: action_type='refund'.\n"
        "2. If customer is ANGRY or mentions LEGAL/COMPLAINTS: action_type='escalate'.\n"
        "3. For status/general questions: action_type='reply'.\n"
        "Output format: {\"action_type\": \"refund/escalate/reply\", \"message\": \"...\"}"
    )

    payload = {
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Task Context: {task_id}\nCustomer: {obs.customer_message}\nStatus: {obs.order_status}"}
        ],
        "temperature": 0,  
        "max_tokens": 150
    }

    try:
        response = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json=payload,
            timeout=20
        )

        if response.status_code == 503 and retries > 0:
            time.sleep(5)
            return get_action_from_model(obs, task_id, retries - 1)

        if response.status_code != 200:
            return {"action_type": "reply", "message": "I am looking into that for you."}

        data = response.json()
        text = data['choices'][0]['message']['content'].strip()
        
        start, end = text.find("{"), text.rfind("}") + 1
        return json.loads(text[start:end])

    except Exception:
        return {"action_type": "reply", "message": "Connecting to a representative..."}

def run_baseline():
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY environment variable not found.")
        return

    print("--- Starting OpenEnv Baseline Evaluation ---")
    env = CustomerSupportEnv()
    
    tasks = ["order_status_check", "refund_request_damaged", "legal_escalation"]
    total_scores = []

    for task_id in tasks:
        obs = env.reset(task_id=task_id)
        
        print(f"\n[TASK] {task_id.upper()}")
        print(f"Customer: {obs.customer_message}")

        action_data = get_action_from_model(obs, task_id)

        action = Action(
            action_type=action_data.get("action_type", "reply"),
            message=action_data.get("message", "One moment please.")
        )

        _, reward, done, _ = env.step(action)
        
        score = reward.score if hasattr(reward, 'score') else reward
        total_scores.append(score)

        print(f"Agent Action: {action.action_type}")
        print(f"Task Score: {score}")

    avg_score = sum(total_scores) / len(total_scores)
    print("\n" + "="*40)
    print(f"FINAL AVERAGE BASELINE SCORE: {avg_score:.2f}")
    print("="*40)

if __name__ == "__main__":
    run_baseline()