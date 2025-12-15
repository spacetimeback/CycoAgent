import os
import time
import pandas as pd
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from tqdm import tqdm

# 加载环境变量
load_dotenv()

# Alpha Vantage API 配置
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
BASE_URL = "https://www.alphavantage.co/query"

def format_date_for_api(date_str):
    """
    将日期字符串格式化为 Alpha Vantage API 所需的格式 (YYYYMMDDTHHMM)
    
    Parameters:
        date_str (str): 日期字符串，格式为 YYYY-MM-DD
        
    Returns:
        str: 格式化后的日期字符串，格式为 YYYYMMDDTHHMM
    """
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    return date_obj.strftime("%Y%m%dT0000")

def fetch_news_sentiment(ticker, time_from, time_to, limit=1000):
    """
    从 Alpha Vantage API 获取新闻情感数据
    
    Parameters:
        ticker (str): 股票代码
        time_from (str): 开始日期，格式为 YYYYMMDDTHHMM
        time_to (str): 结束日期，格式为 YYYYMMDDTHHMM
        limit (int): 返回的最大结果数，默认为 1000
        
    Returns:
        dict: API 响应数据
    """
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "time_from": time_from,
        "time_to": time_to,
        "sort": "LATEST",
        "limit": limit,
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    
    response = requests.get(BASE_URL, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

def process_news_data(news_data):
    """
    处理从 API 获取的新闻数据，转换为 DataFrame
    
    Parameters:
        news_data (dict): API 返回的新闻数据
        
    Returns:
        pd.DataFrame: 处理后的新闻数据
    """
    if not news_data or "feed" not in news_data:
        print("No news data found or invalid response format")
        return pd.DataFrame()
    
    news_items = news_data["feed"]
    
    # 提取新闻数据
    processed_data = []
    for item in news_items:
        # 提取情感数据
        sentiment_data = item.get("overall_sentiment", {})
        sentiment_score = sentiment_data.get("score", 0)
        sentiment_label = sentiment_data.get("label", "")
        
        # 提取时间数据
        time_published = item.get("time_published", "")
        if time_published:
            try:
                # 将时间字符串转换为 datetime 对象
                dt = datetime.strptime(time_published, "%Y%m%dT%H%M%S")
                date = dt.date()
            except ValueError:
                date = None
        else:
            date = None
        
        # 创建新闻条目
        news_item = {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "summary": item.get("summary", ""),
            "source": item.get("source", ""),
            "category": item.get("category_within_source", ""),
            "sentiment_score": sentiment_score,
            "sentiment_label": sentiment_label,
            "datetime": dt if 'dt' in locals() else None,
            "date": date,
            "ticker": item.get("ticker_sentiment", [{}])[0].get("ticker", ""),
            "ticker_sentiment_score": item.get("ticker_sentiment", [{}])[0].get("sentiment_score", 0),
            "ticker_sentiment_label": item.get("ticker_sentiment", [{}])[0].get("sentiment_label", "")
        }
        
        processed_data.append(news_item)
    
    # 创建 DataFrame
    df = pd.DataFrame(processed_data)
    
    # 按日期排序
    if not df.empty and "date" in df.columns:
        df = df.sort_values(by="date", ascending=False)
    
    return df

def main():
    # 设置参数
    ticker = "AMZN"
    start_date = "2023-08-15"
    end_date = "2025-03-28"
    
    # 格式化日期
    time_from = format_date_for_api(start_date)
    time_to = format_date_for_api(end_date)
    
    print(f"Fetching news for {ticker} from {start_date} to {end_date}...")
    
    # 获取新闻数据
    news_data = fetch_news_sentiment(ticker, time_from, time_to)
    
    if news_data:
        # 处理新闻数据
        df = process_news_data(news_data)
        
        if not df.empty:
            # 保存数据
            output_file = f"data/03_primary/alpha_vantage_{ticker}_news_{start_date}_to_{end_date}.csv"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            df.to_csv(output_file, index=False)
            print(f"Saved {len(df)} news items to {output_file}")
            
            # 显示数据统计
            print(f"\nData Summary:")
            print(f"Total news items: {len(df)}")
            print(f"Date range: {df['date'].min()} to {df['date'].max()}")
            print(f"Sources: {df['source'].nunique()} different sources")
            print(f"Average sentiment score: {df['sentiment_score'].mean():.2f}")
            
            # 显示前几行数据
            print("\nSample data:")
            print(df[["date", "title", "source", "sentiment_score", "sentiment_label"]].head())
        else:
            print("No news data found for the specified period")
    else:
        print("Failed to fetch news data")

if __name__ == "__main__":
    main() 