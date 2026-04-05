from pydantic import BaseModel
from typing import List, Optional


class Observation(BaseModel):
    customer_message: str
    order_status: str
    history: List[str]


class Action(BaseModel):
    action_type: str
    message: str


class Reward(BaseModel):
    score: float
    feedback: Optional[str] = None