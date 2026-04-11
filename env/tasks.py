tasks = [
    {
        "id": "order_status_check",
        "customer_message": "Where is my order #12345?",
        "order_status": "Shipped",
        "difficulty": "easy",
        "grader": "env.grader:grade",
        "expected": {
            "type": "reply",
            "keywords": ["shipped", "order", "status"]
        }
    },
    {
        "id": "refund_request_damaged",
        "customer_message": "My item arrived broken. I want a refund.",
        "order_status": "Delivered",
        "difficulty": "medium",
        "grader": "env.grader:grade",
        "expected": {
            "type": "refund",
            "keywords": ["sorry", "damaged", "refund", "processed", "item", "broken"]
        }
    },
    {
        "id": "legal_escalation",
        "customer_message": "This is unacceptable. I am calling my lawyer!",
        "order_status": "Pending",
        "difficulty": "hard",
        "grader": "env.grader:grade",
        "expected": {
            "type": "escalate",
            "keywords": ["apologize", "legal", "manager", "escalate", "inconvenience"]
    }
}
]