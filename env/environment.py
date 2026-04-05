import random
from typing import Tuple, Dict, Any, Optional
from env.models import Observation, Action, Reward
from env.tasks import tasks
from env.grader import grade

class CustomerSupportEnv:

    def __init__(self):
        self.current_task = None
        self.history = []
        self.steps = 0   
        self.max_steps = 5  
        self.task_id = None

    def reset(self, task_id: Optional[str] = None) -> Observation:
        """
        Resets the environment. 
        Requirement: Supports selecting specific tasks (Easy/Medium/Hard).
        """
        if task_id:
            selected_task = next((t for t in tasks if t.get("id") == task_id), None)
            if not selected_task:
                raise ValueError(f"Task {task_id} not found in task definitions.")
            self.current_task = selected_task
            self.task_id = task_id
        else:
            self.current_task = random.choice(tasks)
            self.task_id = self.current_task.get("id", "random")

        self.history = []
        self.steps = 0  

        return Observation(
            customer_message=self.current_task["customer_message"],
            order_status=self.current_task["order_status"],
            history=self.history
        )

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if self.current_task is None:
            raise ValueError("Call /reset before /step")

        self.steps += 1  
        expected = self.current_task.get("expected")
        
        base_score = grade(action, expected)

        step_penalty = -0.05 * self.steps
        
        quality_penalty = -0.2 if len(action.message) < 10 else 0
        
        final_reward_value = max(-1.0, min(1.0, base_score + step_penalty + quality_penalty))

        self.history.append(f"Agent ({action.action_type}): {action.message}")

        reward = Reward(
            score=final_reward_value,
            feedback=(
                "Excellent response" if base_score > 0.8 else
                "Good but can improve" if base_score > 0.5 else
                "Action does not match customer needs"
            )
        )


        done = (base_score > 0.8) or (self.steps >= self.max_steps)

        observation = Observation(
            customer_message=self.current_task["customer_message"],
            order_status=self.current_task["order_status"],
            history=self.history
        )

        info = {
            "task_id": self.task_id,
            "expected_action": expected,
            "step_count": self.steps,
            "base_grader_score": base_score
        }

        return observation, reward, done, info

    def state(self) -> Dict:
        """
        Requirement: Implement state() returning current environment status.
        """
        return {
            "task_id": self.task_id,
            "customer_message": self.current_task["customer_message"] if self.current_task else None,
            "order_status": self.current_task["order_status"] if self.current_task else None,
            "history": self.history,
            "steps": self.steps,
            "done": (self.steps >= self.max_steps)
        }