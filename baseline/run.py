import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.environment import CustomerSupportEnv
from env.models import Action


def simple_agent(obs):
    msg = obs.customer_message.lower()

    if "legal" in msg or "complained" in msg:
        return "escalate", "I understand your concern. I am escalating this to our support team immediately."

    elif "damaged" in msg or "broken" in msg:
        return "refund", "I'm really sorry for the inconvenience. We will process a refund or replacement right away."

    elif "where is my order" in msg:
        return "reply", f"Your order status is: {obs.order_status}"

    else:
        return "ask_clarification", "Could you please provide more details about your issue?"


def run_baseline():
    env = CustomerSupportEnv()

    obs = env.reset()

    print("Customer:", obs.customer_message)

    action_type, message = simple_agent(obs)

    action = Action(
        action_type=action_type,
        message=message
    )

    _, reward, _, _ = env.step(action)

    print("Action:", action_type)
    print("Response:", message)
    print("Score:", reward.score)


if __name__ == "__main__":
    run_baseline()