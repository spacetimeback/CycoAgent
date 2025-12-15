# Arguments
END_POINT_TEMPLATE = "https://data.alpaca.markets/v1beta1/news?start={start_date}&end={end_date}&limit=50&symbols={symbol}"
END_POINT_TEMPLATE_LINK_PAGE = "https://data.alpaca.markets/v1beta1/news?limit=50&symbol={symbol}&page_token={page_token}"
NUM_NEWS_PER_RECORD = 200
MAX_ATTEMPTS = 5
WAIT_TIME = 60
MAX_WORKERS = 30

# dependencies
import os
import time
import shutil
import httpx
import tenacity
import polars as pl
from dotenv import load_dotenv
from rich import print
from tqdm import tqdm
from uuid import uuid4
from datetime import date, timedelta, datetime
from typing import List, Dict, Tuple, Union
from tenacity import retry, stop_after_attempt, wait_fixed
import logging
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

logger = logging.getLogger(__name__)


def round_to_next_day(date: pl.Expr) -> pl.Expr:
    hour = date.dt.hour()
    minute = date.dt.minute()
    second = date.dt.second()
    year = date.dt.year()
    month = date.dt.month()
    day = date.dt.day()
    condition = ((hour >= 16)) & ((second > 0) | (minute > 0))
    new_day = day + condition.cast(pl.UInt32)
    return pl.datetime(year, month, new_day, 9, 0, 0)


class ScraperError(Exception):
    pass


class RecordContainerFull(Exception):
    pass


class ParseRecordContainer:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self.record_counter = 0
        self.author_list = []
        self.content_list = []
        self.date_list = []
        self.source_list = []
        self.summary_list = []
        self.title_list = []
        self.url_list = []

    def add_records(self, records: List[Dict[str, str]]) -> None:
        for cur_record in records:
            self.author_list.append(cur_record["author"])
            self.content_list.append(cur_record["content"])
            date = cur_record["created_at"].rstrip("Z")
            self.date_list.append(datetime.fromisoformat(date))
            self.source_list.append(cur_record["source"])
            self.summary_list.append(cur_record["summary"])
            self.title_list.append(cur_record["headline"])
            self.url_list.append(cur_record["url"])
            self.record_counter += 1
            if self.record_counter == NUM_NEWS_PER_RECORD:
                raise RecordContainerFull

    def pop(self, align_next_date: bool = True) -> Union[pl.DataFrame, None]:
        if self.record_counter == 0:
            return None
        return_df = pl.DataFrame(
            {
                "author": self.author_list,
                "content": self.content_list,
                "datetime": self.date_list,
                "source": self.source_list,
                "summary": self.summary_list,
                "title": self.title_list,
                "url": self.url_list,
            }
        )
        if align_next_date:
            return_df = return_df.with_columns(
                round_to_next_day(return_df["datetime"]).alias("date"),
            )
        else:
            return_df = return_df.with_columns(
                pl.col("datetime").date().alias("date"),
            )
        return return_df.with_columns(pl.lit(self.symbol).alias("equity"))


@retry(stop=stop_after_attempt(MAX_ATTEMPTS), wait=wait_fixed(WAIT_TIME))
def query_one_record(args: Tuple[date, str]) -> None:
    date, symbol = args
    next_date = date + timedelta(days=1)
    request_header = {
        "Apca-Api-Key-Id": os.environ.get("ALPACA_KEY"),
        "Apca-Api-Secret-Key": os.environ.get("ALPACA_KEY_SECRET_KEY"),
    }
    container = ParseRecordContainer(symbol)

    with httpx.Client() as client:
        # first request
        response = client.get(
            END_POINT_TEMPLATE.format(
                start_date=date.strftime("%Y-%m-%d"),
                end_date=next_date.strftime("%Y-%m-%d"),
                symbol=symbol,
            ),
            headers=request_header,
        )
        if response.status_code != 200:
            print("[red]Hit limit[/red]")
            raise ScraperError(response.text)
        result = response.json()
        next_page_token = result["next_page_token"]
        container.add_records(result["news"])

        while next_page_token:
            try:
                response = client.get(
                    END_POINT_TEMPLATE_LINK_PAGE.format(
                        symbol=symbol, page_token=next_page_token
                    ),
                    headers=request_header,
                )
                if response.status_code != 200:
                    raise ScraperError(response.text)
                result = response.json()
                next_page_token = result["next_page_token"]
                container.add_records(result["news"])
            except RecordContainerFull:
                break

    result = container.pop(align_next_date=True)
    if result is not None:
        result.write_parquet(os.path.join("data", "temp", f"{uuid4()}.parquet"))


def main_sync() -> None:
    # 设置输出目录和文件
    output_dir = os.path.join("data", "03_primary")
    output_file = os.path.join(output_dir, "amzn.parquet")
    temp_dir = os.path.join("data", "temp")
    checkpoint_file = os.path.join(temp_dir, "checkpoint.json")
    
    # 确保目录存在
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # 加载检查点
        processed_dates = set()
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                    processed_dates = set(checkpoint.get('processed_dates', []))
                logger.info(f"从检查点恢复，已处理 {len(processed_dates)} 个日期")
            except Exception as e:
                logger.error(f"加载检查点失败: {str(e)}")
        
        # 设置特定的日期范围
        start_date = datetime(2023, 8, 15)
        end_date = datetime(2025, 3, 28)
        
        # 获取所有需要处理的日期
        current_date = start_date
        dates_to_process = []
        while current_date <= end_date:
            if current_date.date() not in processed_dates:
                dates_to_process.append(current_date)
            current_date += timedelta(days=1)
        
        if not dates_to_process:
            logger.info("所有日期都已处理完成")
            return
            
        logger.info(f"需要处理 {len(dates_to_process)} 个日期的数据")
        
        # 批量处理数据
        batch_size = 3  # 每次处理3天的数据，避免API限制
        for i in range(0, len(dates_to_process), batch_size):
            batch_dates = dates_to_process[i:i + batch_size]
            batch_data = []
            
            # 并行处理每个日期的数据
            with ThreadPoolExecutor(max_workers=min(batch_size, 3)) as executor:
                futures = []
                for date in batch_dates:
                    futures.append(executor.submit(
                        query_one_record,
                        (date, 'AMZN')
                    ))
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result is not None:
                            batch_data.append(result)
                    except Exception as e:
                        logger.error(f"处理日期数据时出错: {str(e)}")
            
            # 保存批处理结果
            if batch_data:
                # 合并数据
                combined_df = pl.concat(batch_data)
                
                # 追加到输出文件
                if os.path.exists(output_file):
                    existing_df = pl.read_parquet(output_file)
                    combined_df = pl.concat([existing_df, combined_df])
                
                # 去重并保存
                combined_df = combined_df.unique()
                combined_df.write_parquet(output_file)
                
                # 更新检查点
                processed_dates.update(date.date() for date in batch_dates)
                with open(checkpoint_file, 'w') as f:
                    json.dump({'processed_dates': list(processed_dates)}, f)
                
                logger.info(f"已处理并保存 {len(batch_data)} 天的数据")
            
            # 添加延迟以避免API限制
            time.sleep(2)  # 增加延迟时间，避免API限制
            
    except Exception as e:
        logger.error(f"处理数据时出错: {str(e)}")
    finally:
        # 保留临时目录和检查点文件
        pass


if __name__ == "__main__":
    main_sync()