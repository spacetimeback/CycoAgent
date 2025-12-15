import os
import time
import pandas as pd
import requests
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
from tqdm import tqdm

# 加载环境变量
load_dotenv()

# Alpha Vantage API 配置
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
BASE_URL = "https://www.alphavantage.co/query"

def get_quarters_in_range(start_date, end_date):
    """
    获取指定日期范围内的所有季度
    
    Parameters:
        start_date (str): 开始日期，格式为 YYYY-MM-DD
        end_date (str): 结束日期，格式为 YYYY-MM-DD
        
    Returns:
        list: 季度列表，格式为 YYYYQN
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    quarters = []
    current = start
    
    while current <= end:
        # 计算当前季度
        year = current.year
        month = current.month
        quarter = (month - 1) // 3 + 1
        
        # 添加季度到列表
        quarter_str = f"{year}Q{quarter}"
        if quarter_str not in quarters:
            quarters.append(quarter_str)
        
        # 移动到下一个季度
        if quarter == 4:
            current = datetime(year + 1, 1, 1)
        else:
            current = datetime(year, quarter * 3 + 1, 1)
    
    return quarters

def safe_json_decode(response):
    """
    安全地解析JSON响应，处理可能的错误
    
    Parameters:
        response: requests.Response对象
        
    Returns:
        dict: 解析后的JSON数据，如果解析失败则返回None
    """
    try:
        # 打印原始响应内容以便调试
        print(f"Raw response content: {response.text[:500]}...")
        
        # 尝试解析JSON
        return response.json()
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        print(f"响应状态码: {response.status_code}")
        print(f"响应头: {response.headers}")
        return None
    except Exception as e:
        print(f"处理响应时出错: {e}")
        return None

def fetch_earnings_calendar(symbol, start_date, end_date):
    """
    从 Alpha Vantage API 获取财报日历
    
    Parameters:
        symbol (str): 股票代码
        start_date (str): 开始日期，格式为 YYYY-MM-DD
        end_date (str): 结束日期，格式为 YYYY-MM-DD
        
    Returns:
        dict: API 响应数据
    """
    params = {
        "function": "EARNINGS_CALENDAR",
        "symbol": symbol,
        "horizon": "3year",
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    
    print(f"Requesting earnings calendar for {symbol}...")
    print(f"Request URL: {BASE_URL}?{'&'.join([f'{k}={v}' for k, v in params.items()])}")
    
    try:
        response = requests.get(BASE_URL, params=params)
        
        if response.status_code == 200:
            data = safe_json_decode(response)
            if data:
                # 打印完整的API响应以便调试
                print(f"API Response for earnings calendar:")
                print(json.dumps(data, indent=2)[:500] + "..." if len(json.dumps(data)) > 500 else json.dumps(data, indent=2))
            return data
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"请求财报日历时出错: {e}")
        return None

def fetch_earnings_call_transcript(symbol, quarter):
    """
    从 Alpha Vantage API 获取财报电话会议记录
    
    Parameters:
        symbol (str): 股票代码
        quarter (str): 财政季度，格式为 YYYYQM
        
    Returns:
        dict: API 响应数据
    """
    params = {
        "function": "EARNINGS_CALL_TRANSCRIPT",
        "symbol": symbol,
        "quarter": quarter,
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    
    print(f"Requesting data for {symbol} {quarter}...")
    print(f"Request URL: {BASE_URL}?{'&'.join([f'{k}={v}' for k, v in params.items()])}")
    
    try:
        response = requests.get(BASE_URL, params=params)
        
        if response.status_code == 200:
            data = safe_json_decode(response)
            if data:
                # 打印完整的API响应以便调试
                print(f"API Response for {symbol} {quarter}:")
                print(json.dumps(data, indent=2)[:500] + "..." if len(json.dumps(data)) > 500 else json.dumps(data, indent=2))
            return data
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"请求会议记录时出错: {e}")
        return None

def process_transcript_data(transcript_data, quarter):
    """
    处理从 API 获取的会议记录数据
    
    Parameters:
        transcript_data (dict): API 返回的会议记录数据
        quarter (str): 财政季度，格式为 YYYYQM
        
    Returns:
        list: 处理后的会议记录数据列表
    """
    # 检查API响应格式
    if not transcript_data:
        print(f"No data returned for {quarter}")
        return None
    
    # 打印API响应结构以便调试
    print(f"API response structure for {quarter}: {type(transcript_data)}")
    if isinstance(transcript_data, dict):
        print(f"Keys: {list(transcript_data.keys())}")
    
    # 处理不同的API响应格式
    if isinstance(transcript_data, dict) and "transcript" in transcript_data:
        transcript = transcript_data["transcript"]
    elif isinstance(transcript_data, dict) and "transcripts" in transcript_data:
        # 如果API返回的是transcripts列表
        if transcript_data["transcripts"] and len(transcript_data["transcripts"]) > 0:
            transcript = transcript_data["transcripts"][0]  # 取第一个记录
        else:
            print(f"No transcripts found for {quarter}")
            return None
    else:
        print(f"Unexpected API response format for {quarter}")
        return None
    
    # 检查transcript是否为列表
    if not isinstance(transcript, list):
        print(f"Transcript is not a list for {quarter}: {type(transcript)}")
        return None
    
    # 提取会议记录数据
    processed_data = []
    for entry in transcript:
        entry_data = {
            "quarter": quarter,
            "symbol": transcript_data.get("symbol", ""),
            "speaker": entry.get("speaker", ""),
            "title": entry.get("title", ""),
            "content": entry.get("content", ""),
            "sentiment": entry.get("sentiment", "")
        }
        processed_data.append(entry_data)
    
    return processed_data

def extract_quarter_from_date(date_str):
    """
    从日期字符串中提取季度
    
    Parameters:
        date_str (str): 日期字符串，格式为 YYYY-MM-DD
        
    Returns:
        str: 季度字符串，格式为 YYYYQN
    """
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        year = date_obj.year
        month = date_obj.month
        quarter = (month - 1) // 3 + 1
        return f"{year}Q{quarter}"
    except ValueError:
        return None

def check_api_key():
    """
    检查API密钥是否有效
    
    Returns:
        bool: API密钥是否有效
    """
    if not ALPHA_VANTAGE_API_KEY:
        print("错误: 未设置ALPHA_VANTAGE_API_KEY环境变量")
        print("请在.env文件中添加: ALPHA_VANTAGE_API_KEY=your_api_key_here")
        return False
    
    # 使用一个简单的API调用来测试密钥
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": "IBM",
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    
    try:
        print("测试API密钥...")
        response = requests.get(BASE_URL, params=params)
        
        if response.status_code == 200:
            data = safe_json_decode(response)
            if data and "Time Series (Daily)" in data:
                print("API密钥有效")
                return True
            elif data and "Error Message" in data:
                print(f"API错误: {data['Error Message']}")
                return False
            else:
                print("API响应格式不符合预期")
                return False
        else:
            print(f"API请求失败，状态码: {response.status_code}")
            return False
    except Exception as e:
        print(f"测试API密钥时出错: {e}")
        return False

def main():
    # 检查API密钥
    if not check_api_key():
        print("请检查您的API密钥并重试")
        return
    
    # 设置参数
    symbol = "AMZN"
    start_date = "2023-08-15"
    end_date = "2025-03-28"
    
    # 获取财报日历
    earnings_calendar = fetch_earnings_calendar(symbol, start_date, end_date)
    
    if not earnings_calendar or "earningsCalendar" not in earnings_calendar:
        print("Failed to fetch earnings calendar. Trying with quarters...")
        # 获取日期范围内的所有季度
        quarters = get_quarters_in_range(start_date, end_date)
    else:
        # 从财报日历中提取季度
        earnings_data = earnings_calendar["earningsCalendar"]
        quarters = []
        for earning in earnings_data:
            if "reportedDate" in earning:
                quarter = extract_quarter_from_date(earning["reportedDate"])
                if quarter and quarter not in quarters:
                    quarters.append(quarter)
        
        # 如果没有从财报日历中提取到季度，则使用日期范围计算
        if not quarters:
            quarters = get_quarters_in_range(start_date, end_date)
    
    print(f"Fetching earnings call transcripts for {symbol} from {start_date} to {end_date}...")
    print(f"Quarters to fetch: {quarters}")
    
    # 获取会议记录数据
    all_transcripts = []
    for quarter in tqdm(quarters, desc="Fetching transcripts"):
        transcript_data = fetch_earnings_call_transcript(symbol, quarter)
        
        if transcript_data:
            processed_data = process_transcript_data(transcript_data, quarter)
            if processed_data:
                all_transcripts.extend(processed_data)
        
        # 添加延迟以避免 API 限制
        time.sleep(12)  # Alpha Vantage 免费 API 限制为每分钟 5 次调用
    
    # 如果没有找到数据，尝试获取一些历史数据
    if not all_transcripts:
        print("\nNo transcripts found for the specified period. Trying to fetch historical data...")
        
        # 尝试获取最近几个季度的数据
        historical_quarters = ["2023Q2", "2023Q1", "2022Q4", "2022Q3"]
        for quarter in historical_quarters:
            print(f"\nTrying to fetch historical data for {quarter}...")
            transcript_data = fetch_earnings_call_transcript(symbol, quarter)
            
            if transcript_data:
                processed_data = process_transcript_data(transcript_data, quarter)
                if processed_data:
                    all_transcripts.extend(processed_data)
            
            # 添加延迟以避免 API 限制
            time.sleep(12)
    
    if all_transcripts:
        # 创建 DataFrame
        df = pd.DataFrame(all_transcripts)
        
        # 保存数据
        output_file = f"data/03_primary/alpha_vantage_{symbol}_earnings_calls_{start_date}_to_{end_date}.csv"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"Saved {len(df)} earnings call transcript entries to {output_file}")
        
        # 显示数据统计
        print(f"\nData Summary:")
        print(f"Total transcript entries: {len(df)}")
        print(f"Quarters covered: {', '.join(df['quarter'].unique().tolist())}")
        print(f"Speakers: {df['speaker'].nunique()} different speakers")
        print(f"Average sentiment: {df['sentiment'].astype(float).mean():.2f}")
        
        # 显示前几行数据
        print("\nSample data:")
        print(df[["quarter", "speaker", "title", "sentiment"]].head())
    else:
        print("No earnings call transcripts found for the specified period or historical data.")
        print("This could be due to:")
        print("1. API key issues - Check if your API key is valid and has access to this endpoint")
        print("2. Data availability - The company may not have earnings call transcripts for these periods")
        print("3. API limitations - Your API subscription may not include this data")
        print("4. API changes - The API structure may have changed")
        print("\nAlternative approaches:")
        print("1. Try using a different API service for earnings call transcripts")
        print("2. Check if Tesla provides transcripts on their investor relations website")
        print("3. Consider using a paid subscription to Alpha Vantage for access to this data")

if __name__ == "__main__":
    main() 