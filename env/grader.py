def grade(action, expected):
    score = 0.0

    if expected is None:
        return 0.0

    provided_type = str(action.action_type).strip().lower()
    expected_type = str(expected.get("type", "")).strip().lower()

    if provided_type == expected_type:
        score += 0.5
    else:
        score -= 0.2

    keywords = expected.get("keywords", [])
    message_content = str(action.message).lower()

    if keywords:
        matches = 0
        for word in keywords:
            if word.lower() in message_content:
                matches += 1
        score += 0.5 * (matches / len(keywords))

    score = max(0.0, min(float(score), 1.0))

    return score