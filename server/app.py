import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from env.environment import CustomerSupportEnv
from env.models import Action

app = FastAPI()
env = CustomerSupportEnv()

class ResetRequest(BaseModel):
    task_id: Optional[str] = None

@app.post("/reset")
def reset(request: Optional[ResetRequest] = None):
    """
    Resets the environment. 
    If task_id is provided, it initializes that specific scenario.
    """
    global env
    task_id = request.task_id if request else None
    
    try:
        obs = env.reset(task_id=task_id)
        return obs.model_dump()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step")
def step(action: dict):
    """
    Executes a step in the environment using the provided action.
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
        raise HTTPException(status_code=422, detail=f"Invalid action format: {str(e)}")

@app.get("/state")
def state():
    """
    Returns the current state of the environment.
    """
    global env
    return env.state()

@app.get("/")
def home():
    return {
        "status": "running",
        "env": "customer-support-env",
        "version": "1.0.0"
    }

# --- ADDED FOR OPENENV VALIDATION ---

def main():
    """
    The entry point called by the 'customer-support-env' script 
    defined in pyproject.toml.
    """
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()