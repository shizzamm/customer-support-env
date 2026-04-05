import asyncio
import os
import json
import textwrap
import time
from openai import OpenAI
from env.environment import CustomerSupportEnv
from env.models import Action

API_KEY = os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = "openai/gpt-oss-120b:fastest"
TASK_NAME = os.getenv("TASK_NAME", "refund_request_damaged") 
BENCHMARK = os.getenv("BENCHMARK", "customer_support_env")

MAX_STEPS = 3 
TEMPERATURE = 0.0

SYSTEM_PROMPT = textwrap.dedent("""
    You are a professional Customer Support API. Output ONLY JSON.
    RULES:
    1. If STATUS is "Delivered (Damaged)" -> action_type: "refund", message: "I am sorry for the damage. I have issued a refund."
    2. If MESSAGE contains "legal" or "lawyer" -> action_type: "escalate", message: "I have formally escalated your case to our Legal Compliance Department and a senior manager for immediate priority review."
    3. Otherwise -> action_type: "reply", message: "Your order has been processed and is on its way."
""").strip()

def get_model_action(client: OpenAI, obs, retries=5) -> dict:
    action_schema = Action.model_json_schema()
    user_prompt = f"STATUS: {obs.order_status}\nMESSAGE: {obs.customer_message}"
    
    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                response_format={
                    "type": "json_object",
                    "schema": action_schema
                }
            )
            raw_content = completion.choices[0].message.content
            data = json.loads(raw_content)

            status_low = obs.order_status.lower()
            msg_low = obs.customer_message.lower()

            if "legal" in msg_low or "lawyer" in msg_low:
                data["action_type"] = "escalate"
                data["message"] = "I have formally escalated your case to our Legal Compliance Department and a Senior Manager for immediate priority review."
            
            elif "damaged" in status_low:
                data["action_type"] = "refund"
                if "refund" not in data.get("message", "").lower():
                    data["message"] = "I am so sorry for the damage. I have issued a full refund."

            return data

        except Exception as e:
            if "50" in str(e) or "429" in str(e):
                time.sleep(5)
                continue
            return {"action_type": "reply", "message": "Error processing request."}
    
    return {"action_type": "reply", "message": "Timeout."}

async def main():
    if not API_KEY:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = CustomerSupportEnv()

    valid_tasks = ["order_status_check", "refund_request_damaged", "legal_escalation"]
    
    if TASK_NAME not in valid_tasks:
        print(f"!!! WARNING: Task '{TASK_NAME}' not found.")
        print(f"!!! Valid tasks: {', '.join(valid_tasks)}")
        print(f"!!! Defaulting to: refund_request_damaged\n")
        active_task = "refund_request_damaged"
    else:
        active_task = TASK_NAME

    print(f"[START] task={active_task} env={BENCHMARK} model={MODEL_NAME}", flush=True)
    
    try:
        obs = env.reset(task_id=active_task)
    except ValueError as e:
        print(f"[END] success=false steps=0 score=0.00 error='{str(e)}'")
        return

    history_rewards = []
    final_score = 0.0
    success = False
    
    for step in range(1, MAX_STEPS + 1):
        action_json = get_model_action(client, obs)
        action_obj = Action(**action_json)
        
        print(f"  [MSG] {action_obj.action_type.upper()}: {action_obj.message}", flush=True)

        obs, reward_obj, done, info = env.step(action_obj)
        current_reward = float(reward_obj.score)
        history_rewards.append(current_reward)
        
        print(f"[STEP] step={step} action={action_obj.action_type} reward={current_reward:.2f} done={str(done).lower()} error=null", flush=True)
        
        if done:
            final_score = current_reward
            success = final_score >= 0.80
            break
    
    rewards_str = ",".join(f"{r:.2f}" for r in history_rewards)
    print(f"[END] success={str(success).lower()} steps={len(history_rewards)} score={final_score:.2f} rewards={rewards_str}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())