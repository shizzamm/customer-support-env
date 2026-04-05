tasks = [
    {
        "id": "order_status_check",
        "customer_message": "Where is my order?",
        "order_status": "Shipped, arriving tomorrow",
        "difficulty": "easy",
        "expected": {
            "type": "reply",
            "keywords": ["tomorrow", "shipped"]
        }
    },
    {
        "id": "refund_request_damaged",
        "customer_message": "My order arrived but the item is completely shattered.",
        "order_status": "Delivered (Damaged)",
        "difficulty": "medium",
        "expected": {
            "type": "refund",
            "keywords": ["refund", "sorry"]
        }
    },
    {
        "id": "legal_escalation",
        "customer_message": "I will take legal action if this is not fixed",
        "order_status": "Delivered",
        "difficulty": "hard",
        "expected": {
            "type": "escalate",
            "keywords": ["escalate", "manager", "supervisor", "support"]
        }
    }
]