from news_fetcher import fetch_news

df = fetch_news("Apple")

print(df.head())