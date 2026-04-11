import random
from typing import Tuple, Dict, Any, Optional
from env.models import Observation, Action, Reward
from env.tasks import tasks
from env.grader import grade

class CustomerSupportEnv:
    """
    Standardized OpenEnv-compliant simulation for Customer Support AI Agents.
    Evaluates agents on multi-turn trajectories with performance-based rewards.
    """

    def __init__(self):
        self.current_task = None
        self.history = []
        self.steps = 0   
        self.max_steps = 5  
        self.task_id = None
        self.is_done = False

    def reset(self, task_id: Optional[str] = None) -> Observation:
        """
        Resets the environment. Ensures task_id is tracked for validation.
        """
        if task_id:
            selected_task = next((t for t in tasks if t.get("id") == task_id), None)
            if not selected_task:
                # Fallback to random if specific task not found, but log it
                print(f"[WARNING] Task {task_id} not found, choosing random.")
                selected_task = random.choice(tasks)
            self.current_task = selected_task
            self.task_id = self.current_task.get("id")
        else:
            self.current_task = random.choice(tasks)
            self.task_id = self.current_task.get("id")

        self.history = []
        self.steps = 0
        self.is_done = False

        return Observation(
            customer_message=self.current_task["customer_message"],
            order_status=self.current_task["order_status"],
            history=self.history,
            metadata={"task_id": self.task_id} # Validator looks for this
        )

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Executes one time step. FIX: Standardized grader call order.
        """
        if self.current_task is None:
            raise ValueError("Call reset() before step()")

        self.steps += 1  
        
        base_score = grade(self.current_task, action)

        step_penalty = -0.05 * self.steps
        quality_penalty = -0.2 if len(action.message) < 10 else 0
        
        final_reward_value = max(-1.0, min(1.0, base_score + step_penalty + quality_penalty))

        self.history.append(f"Agent ({action.action_type}): {action.message}")

        self.is_done = (base_score >= 0.4) or (self.steps >= self.max_steps)

        reward = Reward(
            score=final_reward_value,
            feedback=(
                "Excellent response" if base_score > 0.8 else
                "Good but can improve" if base_score > 0.5 else
                "Action does not match customer needs"
            )
        )

        observation = Observation(
            customer_message=self.current_task["customer_message"],
            order_status=self.current_task["order_status"],
            history=self.history,
            metadata={"task_id": self.task_id}
        )

        info = {
            "task_id": self.task_id,
            "step_count": self.steps,
            "base_grader_score": base_score,
            "done": self.is_done
        }

        return observation, reward, self.is_done, info

    def state(self) -> Dict:
        """
        Returns the current internal state.
        """
        return {
            "task_id": self.task_id,
            "customer_message": self.current_task["customer_message"] if self.current_task else None,
            "order_status": self.current_task["order_status"] if self.current_task else None,
            "history": self.history,
            "steps": self.steps,
            "done": self.is_done
        }