def grade(task, action, response=None):
    """
    Standardized grader with deep debugging for Phase 2.
    """  
    score = 0.0

    try:
        if isinstance(task, dict):
            expected = task.get("expected", {})
        else:
            expected = {}

        if not expected:
            return 0.0

        provided_type = ""
        message_content = ""

        if hasattr(action, "action_type"):
            provided_type = str(action.action_type).strip().lower()
        elif isinstance(action, dict):
            provided_type = str(action.get("action_type", "")).strip().lower()

        if hasattr(action, "message"):
            message_content = str(action.message).lower()
        elif isinstance(action, dict):
            message_content = str(action.get("message", "")).lower()

        expected_type = str(expected.get("type", "")).strip().lower()

        if provided_type == expected_type:
            score += 0.5
        else:
            score -= 0.2

        keywords = expected.get("keywords", [])
        if keywords:
            matches = 0
            for word in keywords:
                if word.lower() in message_content:
                    matches += 1
            keyword_score = 0.5 * (matches / len(keywords))
            score += keyword_score

    except Exception as e:
        return 0.0

    final_score = max(0.0, min(float(score), 1.0))
    
    return final_score