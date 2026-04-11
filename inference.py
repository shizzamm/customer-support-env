import os
import json
import textwrap
from typing import List, Optional
from openai import OpenAI

from env.environment import CustomerSupportEnv
from env.models import Action

#from dotenv import load_dotenv
#load_dotenv()

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

TASK_NAME = os.getenv("TASK_NAME", "refund_request_damaged") 
BENCHMARK = "customer_support_env"
MAX_STEPS = 5
TEMPERATURE = 0.0

SYSTEM_PROMPT = textwrap.dedent("""
    You are a professional Customer Support API. Output ONLY JSON.
    You must provide a JSON object with exactly these two keys:
    1. "action_type": string (must be one of: "reply", "refund", "escalate", "ask_clarification")
    2. "message": string (the response text to the customer)

    LOGIC:
    - If status is 'Delivered (Damaged)', action_type MUST be 'refund'.
    - If customer mentions 'legal' or 'lawyer', action_type MUST be 'escalate'.
    - If you need more info, use 'ask_clarification'.
    - Otherwise, use 'reply'.
""").strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

def get_model_action(client: OpenAI, obs) -> dict:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Customer: {obs.customer_message}\nStatus: {obs.order_status}\nHistory: {obs.history}"}
            ],
            temperature=TEMPERATURE,
            response_format={"type": "json_object"}
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        return {"action_type": "reply", "message": "I am looking into this for you."}

def main():
    if not API_KEY:
        print("Error: HF_TOKEN/API_KEY environment variable not set.")
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = CustomerSupportEnv()
    
    rewards_history = []
    steps_taken = 0
    final_score = 0.0
    
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset(task_id=TASK_NAME)
        
        for step in range(1, MAX_STEPS + 1):
            steps_taken = step
            
            action_data = get_model_action(client, obs)
            action_obj = Action(**action_data)
            
            obs, reward_obj, done, info = env.step(action_obj)
            
            current_reward = float(reward_obj.score)
            rewards_history.append(current_reward)
            
            action_summary = f"{action_obj.action_type}('{action_obj.message[:30]}...')"
            log_step(step=step, action=action_summary, reward=current_reward, done=done, error=None)
            
            if done:
                final_score = max(0.0, min(current_reward, 1.0))
                break
    
    except Exception as e:
        log_step(step=steps_taken, action="error", reward=0.0, done=True, error=str(e))
    
    finally:
        success = final_score >= 0.7  
        log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards_history)

if __name__ == "__main__":
    main()