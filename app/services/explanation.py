import re


URL_PATTERN = re.compile(
    r"https?://[^\s<>\"']+|www\.[^\s<>\"']+",
    re.IGNORECASE,
)

INDICATOR_RULES = [
    (
        "Urgency or time pressure",
        re.compile(
            r"\b(urgent|immediately|act now|expires today|final warning|"
            r"within \d+ hours|account will be suspended)\b",
            re.IGNORECASE,
        ),
        "The message uses urgent language or pressures the reader to act quickly.",
    ),
    (
        "Credential or sensitive-information request",
        re.compile(
            r"\b(password|passcode|login credentials|verify your account|"
            r"confirm your identity|social security number|one.time password|"
            r"\botp\b)\b",
            re.IGNORECASE,
        ),
        "The message mentions credentials, identity verification, or sensitive information.",
    ),
    (
        "Payment or financial pressure",
        re.compile(
            r"\b(wire transfer|payment required|pay immediately|"
            r"bank details|gift card|processing fee|claim your refund)\b",
            re.IGNORECASE,
        ),
        "The message mentions a payment or financial action that deserves verification.",
    ),
    (
        "Authority or account-threat language",
        re.compile(
            r"\b(administrator|security department|legal action|"
            r"account suspended|account blocked|law enforcement|"
            r"your account has been compromised)\b",
            re.IGNORECASE,
        ),
        "The message invokes authority, account restrictions, or security threats.",
    ),
]


def explain_text(text: str, prediction: dict) -> dict:
    """Return observable message indicators alongside the model prediction."""
    indicators = []

    for title, pattern, detail in INDICATOR_RULES:
        if pattern.search(text):
            indicators.append({
                "title": title,
                "detail": detail,
            })

    if URL_PATTERN.search(text):
        indicators.append({
            "title": "URL present",
            "detail": (
                "The message contains a web address. Its presence alone "
                "does not establish that the destination is malicious."
            ),
        })

    label = str(prediction.get("label", "UNKNOWN")).upper()

    if indicators:
        summary = (
            f"The model classified this message as {label}. "
            f"{len(indicators)} observable indicator category/categories "
            "were found by simple text rules."
        )
    else:
        summary = (
            f"The model classified this message as {label}. "
            "The configured text rules found no listed indicators; "
            "this does not establish that the message is safe."
        )

    return {
        "summary": summary,
        "indicators": indicators,
        "method": "Rule-based observation of message text",
        "limitations": (
            "These indicators are not proof of phishing and are not a "
            "causal explanation of the DistilBERT prediction. Legitimate "
            "messages can contain these patterns, and phishing messages "
            "may contain none of them."
        ),
    }


def explain_audio(prediction: dict) -> dict:
    """Explain the audio classifier output without inventing acoustic evidence."""
    label = str(prediction.get("prediction", "unknown")).lower()
    probabilities = prediction.get("class_probabilities") or {}

    score = probabilities.get(label)
    if score is not None:
        try:
            score = float(score)
            if not 0.0 <= score <= 1.0:
                score = None
        except (TypeError, ValueError):
            score = None

    if label == "spoof":
        summary = (
            "The audio classifier labeled this recording as potentially "
            "synthetic or spoofed."
        )
    elif label == "genuine":
        summary = (
            "The audio classifier labeled this recording as genuine-like. "
            "This does not guarantee that the recording is authentic."
        )
    else:
        summary = (
            "The audio classifier returned an unrecognized prediction label."
        )

    indicators = []
    if score is not None:
        indicators.append({
            "title": "Classifier output score",
            "detail": (
                f"The classifier assigned a score of {score:.1%} to the "
                f"{label} class. This is a model estimate, not proof of authenticity."
            ),
        })

    return {
        "summary": summary,
        "indicators": indicators,
        "method": (
            "Wav2Vec2 embeddings followed by a StandardScaler and "
            "Logistic Regression classifier"
        ),
        "limitations": (
            "The 768 embedding dimensions are not directly interpretable as "
            "human-readable voice characteristics. This explanation does not "
            "identify a specific acoustic defect or prove that audio was generated."
        ),
    }
