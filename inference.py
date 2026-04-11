import asyncio
import os
import json
import textwrap
from typing import List, Optional
from openai import OpenAI
from env.environment import CustomerSupportEnv
from env.models import Action
#from dotenv import load_dotenv

#load_dotenv()

API_KEY = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN")
API_BASE_URL = os.environ.get("API_BASE_URL")
MODEL_NAME = os.environ.get("MODEL_NAME", "openai/gpt-oss-120b:fastest")

TASK_NAME = os.getenv("TASK_NAME", "refund_request_damaged") 
BENCHMARK = os.getenv("BENCHMARK", "customer_support_env")
MAX_STEPS = 3 
TEMPERATURE = 0.0

SYSTEM_PROMPT = textwrap.dedent("""
    You are a professional Customer Support API. Output ONLY JSON.
    You must provide a JSON object with exactly these two keys:
    1. "action_type": string (must be one of: "reply", "refund", "escalate")
    2. "message": string (the response text)

    RULES:
    - If STATUS is "Delivered (Damaged)" -> action_type: "refund"
    - If MESSAGE contains "legal" or "lawyer" -> action_type: "escalate"
    - Otherwise -> action_type: "reply"
""").strip()

def get_model_action(client: OpenAI, obs) -> dict:
    user_prompt = f"STATUS: {obs.order_status}\nMESSAGE: {obs.customer_message}"
    
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=TEMPERATURE,
        response_format={"type": "json_object"}
    )
    
    raw_data = json.loads(completion.choices[0].message.content)

    repaired_data = {
        "action_type": raw_data.get("action_type") or raw_data.get("type") or "reply",
        "message": raw_data.get("message") or raw_data.get("text") or raw_data.get("acknowledge") or "I am looking into this for you."
    }

    status_low = obs.order_status.lower()
    msg_low = obs.customer_message.lower()
    
    if "legal" in msg_low or "lawyer" in msg_low:
        repaired_data["action_type"] = "escalate"
    elif "damaged" in status_low:
        repaired_data["action_type"] = "refund"
        
    return repaired_data

async def main():
    if not API_KEY or not API_BASE_URL:
        print(f"[ERROR] Missing Credentials. URL: {API_BASE_URL}, Key: {'Present' if API_KEY else 'Missing'}", flush=True)
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = CustomerSupportEnv()
    
    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}", flush=True)
    
    history_rewards = []
    final_score = 0.0
    success = False
    steps_taken = 0

    try:
        obs = env.reset(task_id=TASK_NAME)
        
        for step in range(1, MAX_STEPS + 1):
            steps_taken = step
            try:
                action_json = get_model_action(client, obs)
                action_obj = Action(**action_json)
                
                obs, reward_obj, done, info = env.step(action_obj)
                
                current_reward = float(reward_obj.score)
                history_rewards.append(current_reward)
                
                action_log = f"{action_obj.action_type}('{action_obj.message[:20]}')"
                print(f"[STEP] step={step} action={action_log} reward={current_reward:.2f} done={str(done).lower()} error=null", flush=True)
                
                if done:
                    final_score = current_reward
                    break
                    
            except Exception as e:
                error_msg = str(e).replace("\n", " ").replace("=", " ")[:50]
                print(f"[STEP] step={step} action=null reward=0.00 done=true error={error_msg}", flush=True)
                break 

        final_score = min(max(final_score, 0.0), 1.0)
        success = final_score >= 0.1

    finally:
        rewards_str = ",".join(f"{r:.2f}" for r in history_rewards) if history_rewards else "0.00"
        print(f"[END] success={str(success).lower()} steps={steps_taken} score={final_score:.3f} rewards={rewards_str}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())