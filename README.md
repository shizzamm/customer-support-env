---
title: Customer Support Env
emoji: 👀
colorFrom: green
colorTo: indigo
sdk: docker
pinned: false
license: mit
short_description: 'Customer Support AI Environment'
---

# 🛠️ Customer Support AI Environment (OpenEnv)

This repository contains a standardized OpenEnv-compliant simulation for testing and benchmarking Customer Support AI Agents. It evaluates an LLM's ability to navigate customer intent, follow business logic, and maintain professional communication across various difficulty levels.

## 🎯 Motivation
Traditional static benchmarks fail to capture the multi-turn nature of customer service. This environment provides a **trajectory-based evaluation** where agents are penalized for inefficiency (too many steps) and rewarded for accurate resolution and professional tone.

---

## 🛰️ Action and Observation Space

### Observation Space
The environment returns a JSON object containing the context the agent needs to make a decision:
* `customer_message`: The text input from the customer.
* `order_status`: Internal metadata regarding the customer's purchase.
* `history`: A transcript of the previous turns in the current session.

### Action Space
The agent must submit a JSON object with the following fields:
* `action_type`: The category of response (`reply`, `refund`, `escalate`, `ask_clarification`).
* `message`: The actual text response to be sent to the customer.

---

## 📋 Task Descriptions
The environment includes 3 deterministic tasks defined in `openenv.yaml`:

| Task ID | Difficulty | Scenario | Success Criteria |
| :--- | :--- | :--- | :--- |
| `order_status_check` | **Easy** | User asking "Where is my order?" | Provide status and ETA. |
| `refund_request_damaged` | **Medium** | User reporting a broken product. | Apologize and issue a `refund`. |
| `legal_escalation` | **Hard** | User threatening legal action. | Perform an `escalate` action. |

---

## 🚀 Setup and Usage

### Local Development
1. **Clone and Install:**
   ```bash
   pip install -r requirements.txt

2. Launch Server:
   ```bash
   uvicorn server.app:app --host 0.0.0.0 --port 7860

3. Using the API
   Reset: POST /reset with {"task_id": "legal_escalation"} to start a specific scenario.

   Step: POST /step with the agent's action JSON.

   State: GET /state to inspect the current environment variables.

4. Baseline Scores
   Evaluated using Llama-3.2-1B-Instruct (via OPENAI API) with temperature: 0.

   Order Status Check: 0.95

   Refund Request Damaged: 0.70

   Legal Escalation: 0.45

   Final Average Baseline Score: 0.70

5. Technical Implementation Details
   Framework: FastAPI

   Code: Python 

   API : OPENAI API for Baseline Score

   Validation: Pydantic V2  

   Deployment: Dockerized on Hugging Face Spaces

   Reward Function: Includes a base grader score with a -0.05 penalty per step to encourage efficiency.

6. License
   This project is licensed under the MIT License.   

