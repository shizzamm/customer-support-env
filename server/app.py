import threading
import subprocess
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from typing import Optional
from env.environment import CustomerSupportEnv
from env.models import Action


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events. 
    Replaces the deprecated @app.on_event("startup").
    """
    def run_inference_agent():
        try:
            print("[SYSTEM] Starting Agent Loop (inference.py)...", flush=True)
            subprocess.run(["python", "inference.py"], check=True)
        except Exception as e:
            print(f"[SYSTEM ERROR] Agent execution failed: {e}", flush=True)

    agent_thread = threading.Thread(target=run_inference_agent, daemon=True)
    agent_thread.start()
    
    yield  

    print("[SYSTEM] Application shutting down...", flush=True)

app = FastAPI(lifespan=lifespan)
env = CustomerSupportEnv()


@app.post("/reset")
async def reset(request: Request):
    """
    Resets the environment. Supports optional task_id for validation.
    """
    global env
    try:
        try:
            body = await request.json()
        except Exception:
            body = {}

        task_id = body.get("task_id")

        obs = env.reset(task_id=task_id)
        
        return obs.model_dump()
        
    except Exception as e:
        print(f"[ERROR] Reset failed: {e}", flush=True)
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step")
async def step(action: dict):
    """
    Executes an action in the environment.
    """
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
        print(f"[ERROR] Step failed: {e}", flush=True)
        raise HTTPException(status_code=422, detail=f"Invalid action format: {str(e)}")

@app.get("/state")
def state():
    """
    Returns the current internal state of the environment.
    """
    return env.state()

@app.get("/")
def home():
    """
    Health check endpoint for Hugging Face Spaces.
    """
    return {
        "status": "running",
        "env": "customer-support-env",
        "agent_triggered": True
    }


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()