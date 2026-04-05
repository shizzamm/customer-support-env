import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from env.environment import CustomerSupportEnv
from env.models import Action

# Required env variables
API_BASE_URL = os.getenv("API_BASE_URL", "dummy")
MODEL_NAME = os.getenv("MODEL_NAME", "dummy")
HF_TOKEN = os.getenv("HF_TOKEN", "dummy")


# Rule-based agent  
def simple_agent(obs):
    msg = obs.customer_message.lower()

    if "legal" in msg or "complained" in msg:
        return "escalate", "I understand your concern. Escalating to support team."

    elif "damaged" in msg:
        return "refund", "Sorry for the inconvenience. We will refund or replace."

    elif "where is my order" in msg:
        return "reply", f"Your order status is: {obs.order_status}"

    else:
        return "ask_clarification", "Could you please provide more details?"


def run():
    print("[START] Running inference")

    env = CustomerSupportEnv()

    obs = env.reset()

    print(f"[STEP] Observation: {obs.customer_message}")

    action_type, message = simple_agent(obs)

    action = Action(
        action_type=action_type,
        message=message
    )

    print(f"[STEP] Action: {action_type}")

    _, reward, done, _ = env.step(action)

    print(f"[STEP] Reward: {reward.score}")

    print("[END] Done")


if __name__ == "__main__":
    run()