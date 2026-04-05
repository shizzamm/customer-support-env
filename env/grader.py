def grade(action, expected):
    score = 0.0

    if expected is None:
        return 0.0

    if str(action.action_type).lower() == str(expected.get("type")).lower():
        score += 0.5
    else:
        score -= 0.2  # penalty

    keywords = expected.get("keywords", [])
    matches = sum(1 for word in keywords if word in action.message.lower())

    if len(keywords) > 0:
        score += 0.5 * (matches / len(keywords))

    score = max(0.0, min(score, 1.0))

    return score