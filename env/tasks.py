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
        "customer_message": "My product arrived damaged",
        "order_status": "Delivered",
        "difficulty": "medium",
        "expected": {
            "type": "refund",
            "keywords": ["refund", "replace"]
        }
    },
    {
        "id": "legal_escalation",
        "customer_message": "I will take legal action if this is not fixed",
        "order_status": "Delivered",
        "difficulty": "hard",
        "expected": {
            "type": "escalate",
            "keywords": ["escalate", "support"]
        }
    }
]