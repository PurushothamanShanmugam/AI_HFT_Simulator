from sentiment_engine import get_sentiment

text = """
Apple beats earnings estimates and raises guidance.
"""

result = get_sentiment(text)

print(result)