import threading
import subprocess
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from env.environment import CustomerSupportEnv
from env.models import Action

app = FastAPI()
env = CustomerSupportEnv()

# --- AGENT BACKGROUND EXECUTION ---

def run_inference_agent():
    """
    Runs the inference.py script as a background process.
    This generates the mandatory log tags for Phase 2 validation.
    """
    try:
        print("[SYSTEM] Starting Agent Loop (inference.py)...", flush=True)
        # Using subprocess ensures the agent runs in its own process space
        subprocess.run(["python", "inference.py"], check=True)
    except Exception as e:
        print(f"[SYSTEM ERROR] Agent execution failed: {e}", flush=True)

@app.on_event("startup")
async def startup_event():
    """
    FastAPI startup hook to trigger the agent loop automatically.
    """
    # Start agent in a background thread so it doesn't block the API/Uvicorn
    agent_thread = threading.Thread(target=run_inference_agent, daemon=True)
    agent_thread.start()

# --- EXISTING ENDPOINTS ---

class ResetRequest(BaseModel):
    task_id: Optional[str] = None

@app.post("/reset")
def reset(request: Optional[ResetRequest] = None):
    global env
    task_id = request.task_id if request else None
    try:
        obs = env.reset(task_id=task_id)
        return obs.model_dump()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step")
def step(action: dict):
    global env
    if env.current_task is None:
        env.reset()
    try:
        act = Action(**action)
        obs, reward, done, info = env.step(act)
        return {
            "observation": obs.model_dump(),
            "reward": reward.score, 
            "done": done,
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid action format: {str(e)}")

@app.get("/state")
def state():
    return env.state()

@app.get("/")
def home():
    return {
        "status": "running",
        "env": "customer-support-env",
        "agent_triggered": True
    }

# --- ENTRY POINT ---

def main():
    # If running locally or on HF, this starts the server
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()