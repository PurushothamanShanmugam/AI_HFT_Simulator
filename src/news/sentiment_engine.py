from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

import torch

MODEL_NAME = "ProsusAI/finbert"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME
)

LABELS = [
    "positive",
    "negative",
    "neutral"
]


def get_sentiment(text: str):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():

        outputs = model(**inputs)

    probs = torch.softmax(
        outputs.logits,
        dim=1
    )[0]

    prediction = LABELS[
        torch.argmax(probs).item()
    ]

    confidence = probs.max().item()

    return {

        "sentiment": prediction,

        "confidence": confidence,

        "positive": probs[0].item(),

        "negative": probs[1].item(),

        "neutral": probs[2].item()
    }