import os
import time
import httpx
import polars as pl
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Dict, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
from tqdm import tqdm
import logging

# 创建日志目录
os.makedirs('logs', exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/news_fetcher.log'),
        logging.StreamHandler()
    ]
)

# API配置
BASE_URL = "https://data.alpaca.markets/v1beta1/news"
DEFAULT_PARAMS = {
    "limit": 50,
    "include_content": "true"
}
MAX_RETRIES = 3
RETRY_WAIT = 10

class AlpacaNewsFetcher:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("ALPACA_KEY")
        self.secret_key = os.getenv("ALPACA_KEY_SECRET_KEY")
        
        # 验证API密钥
        if not self.api_key or not self.secret_key:
            raise ValueError("ALPACA_KEY and ALPACA_KEY_SECRET_KEY must be set in .env file")
            
        self.headers = {
            "Apca-Api-Key-Id": self.api_key,
            "Apca-Api-Secret-Key": self.secret_key
        }
        self.output_dir = Path("data/03_primary/news")
        self.temp_dir = Path("data/temp/news")
        
        # 初始化目录
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=1, min=4, max=RETRY_WAIT))
    def fetch_news_page(self, symbol: str, start: str, end: str, page_token: str = None) -> Dict:
        """获取单页新闻数据"""
        params = {**DEFAULT_PARAMS, "symbols": symbol, "start": start, "end": end}
        if page_token:
            params["page_token"] = page_token

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(BASE_URL, headers=self.headers, params=params)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                logging.error("Authentication failed. Please check your API keys.")
                raise
            elif e.response.status_code == 429:
                logging.warning("Rate limit exceeded. Waiting before retry...")
                time.sleep(60)  # 等待60秒后重试
                raise
            else:
                logging.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
                raise
        except Exception as e:
            logging.error(f"Request failed: {str(e)}")
            raise

    def process_news_item(self, item: Dict) -> Dict:
        """处理单个新闻条目"""
        # 处理日期时间
        published_at = item.get("created_at")
        if published_at:
            try:
                # 移除Z后缀并解析为datetime
                published_at = published_at.rstrip("Z")
                dt = datetime.fromisoformat(published_at)
            except ValueError:
                dt = None
        else:
            dt = None
            
        return {
            "id": item.get("id"),
            "symbol": item.get("symbols", [""])[0] if item.get("symbols") else "",
            "published_at": dt,
            "headline": item.get("headline"),
            "content": item.get("content", ""),
            "source": item.get("source"),
            "url": item.get("url"),
            "author": item.get("author", "Unknown")
        }

    def fetch_news_for_date_range(self, symbol: str, start_date: date, end_date: date) -> pl.DataFrame:
        """获取日期范围内的新闻"""
        all_news = []
        current_date = start_date
        
        with tqdm(total=(end_date - start_date).days + 1, desc=f"Fetching {symbol} news") as pbar:
            while current_date <= end_date:
                next_date = current_date + timedelta(days=1)
                page_token = None
                
                try:
                    while True:
                        data = self.fetch_news_page(
                            symbol=symbol,
                            start=current_date.strftime("%Y-%m-%d"),
                            end=next_date.strftime("%Y-%m-%d"),
                            page_token=page_token
                        )
                        
                        for item in data.get("news", []):
                            all_news.append(self.process_news_item(item))
                        
                        page_token = data.get("next_page_token")
                        if not page_token:
                            break
                            
                    pbar.update(1)
                    current_date = next_date
                except Exception as e:
                    logging.error(f"Failed for {current_date}: {str(e)}")
                    current_date = next_date
                    continue

        return pl.DataFrame(all_news)

    def save_news_data(self, df: pl.DataFrame, symbol: str):
        """保存新闻数据"""
        if len(df) == 0:
            logging.warning(f"No news data to save for {symbol}")
            return
            
        try:
            # 按日期分区存储
            df = df.with_columns(
                pl.col("published_at").dt.date().alias("date")
            )
            
            # 保存临时文件
            temp_file = self.temp_dir / f"{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}.parquet"
            df.write_parquet(temp_file)
            
            # 合并到主文件
            output_file = self.output_dir / "amzn.parquet"
            if output_file.exists():
                try:
                    existing_df = pl.read_parquet(output_file)
                    df = pl.concat([existing_df, df]).unique(subset=["id"])
                except Exception as e:
                    logging.error(f"Error reading existing parquet file: {str(e)}")
                    logging.info("Creating new file instead of merging")
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # 保存文件
            df.write_parquet(output_file)
            logging.info(f"Successfully saved {len(df)} records to {output_file}")
            
            # 删除临时文件
            if temp_file.exists():
                temp_file.unlink()
                
        except Exception as e:
            logging.error(f"Error saving news data: {str(e)}")
            raise

def main():
    fetcher = AlpacaNewsFetcher()
    
    # 配置参数
    ticker = "AMZN"
    start_date = date(2023, 8, 15)
    end_date = date(2025, 3, 28)
    
    try:
        news_df = fetcher.fetch_news_for_date_range(ticker, start_date, end_date)
        fetcher.save_news_data(news_df, ticker)
        logging.info(f"Successfully saved {len(news_df)} news items for {ticker}")
    except Exception as e:
        logging.error(f"Failed to fetch news: {str(e)}")

if __name__ == "__main__":
    main()