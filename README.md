---
title: Customer Support Env
emoji: 👀
colorFrom: green
colorTo: indigo
sdk: docker
pinned: false
license: mit
short_description: 'Standardized OpenEnv Customer Support Simulation'
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
| `legal_escalation` | **Hard** | User threatening legal action. | Perform an `escalate` action with professional empathy. |

---

## 📊 Validated Scores
Evaluated using **Qwen/Qwen2.5-72B-Instruct** (via OpenEnv Benchmark) with temperature: 0.

| Task | Score | Efficiency | Result |
| :--- | :--- | :--- | :--- |
| Order Status Check | 0.78 | 1 Step | ✅ Success |
| Refund Request Damaged | 0.78 | 1 Step | ✅ Success |
| Legal Escalation | 0.75 | 1 Step | ✅ Success |
| **Final Average Score** | **0.77** | **Optimal** | **Passed** |

## ⚙️ Technical Implementation Details
* **Framework**: FastAPI
* **Agent Logic**: Asynchronous Agent Loop (`inference.py`)
* **Grader Logic**: Attribute-safe keyword matching with support for Pydantic models and dictionaries.
* **Validation**: Pydantic V2
* **Reward Function**: Includes a base grader score (0.0 - 1.0) with a `-0.05` penalty per step to encourage rapid resolution.
* **Success Threshold**: Tasks are marked successful if the base score reaches `0.4` or higher (optimized for professional escalation).

## 🚀 Setup and Usage

### Local Development
1. **Clone and Install:**
   ```bash
   pip install -r requirements.txt

2. Launch Server:
   ```bash
   uvicorn server.app:app --host 0.0.0.0 --port 7860

3. Using the API
	The environment follows a standard RL-style interface via REST API:

	Reset: POST /reset with {"task_id": "legal_escalation"} to start a specific scenario.

	Step: POST /step with the agent's action JSON (e.g., {"action_type": "reply", "message": "Hello!"}).

	State: GET /state to inspect the current environment variables and history.

4. Deployment
   
   This environment is designed to be Dockerized and is fully compatible with Hugging Face Spaces using the sdk: docker configuration.

5. License
   This project is licensed under the MIT License.   

