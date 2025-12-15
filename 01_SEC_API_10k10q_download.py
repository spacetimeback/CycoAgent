# !pip install clean-text

# parameters
NUM_WORKERS = 1
ENDPOINT_URL = "https://api.sec-api.io?token={SEC_KEY}"
EXTRACTOR_URL = "https://api.sec-api.io/extractor?url={url}&token={SEC_KEY}&item={item}"
TEN_K_ITEM_CODE = [
    # "1",
    # "1A",
    # "1B",
    # "2",
    # "3",
    # "4",
    # "5",
    # "6",
    "7",  # target
    # "7A",
    # "8",
    # "9",
    # "9A",
    # "9B",
    # "10",
    # "11",
    # "12",
    # "13",
    # "14",
    # "15",
]
TEN_Q_ITEM_CODE = [
    # "part1item1",
    "part1item2",  # target
    # "part1item3",
    # "part1item4",
    # "part2item1",
    # "part2item1a",
    # "part2item2",
    # "part2item3",
    # "part2item4",
    # "part2item5",
    # "part2item6",
]
EIGHT_K_ITEM_CODE = [
    "1-1",
    "1-2",
    "1-3",
    "1-4",
    "2-1",
    "2-2",
    "2-3",
    "2-4",
    "2-5",
    "2-6",
    "3-1",
    "3-2",
    "3-3",
    "4-1",
    "4-2",
    "5-1",
    "5-2",
    "5-3",
    "5-4",
    "5-5",
    "5-6",
    "5-7",
    "5-8",
    "6-1",
    "6-2",
    "6-3",
    "6-4",
    "6-5",
    "6-6",
    "6-10",
    "7-1",
    "8-1",
    # "9-1",# not target
]
 
SIZE = 50
START_DATE = "2023-08-15"
END_DATE = "2025-03-28"
SLEEP_TIME = 10

# dependencies
import os
import httpx
import pytz
import time
import logging
import itertools
import polars as pl
from cleantext import clean
from tqdm import tqdm
from rich import print
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from httpx import RequestError
from datetime import datetime
from dateutil import parser
from typing import List, Dict, Any
import concurrent.futures
from unidecode import unidecode


# set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler(
    os.path.join("data", "03_primary", "filing_fails.log"), mode="w"
)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


# load environment variables
logger.info("Program starts")

print(load_dotenv(os.path.join(".env")))


# convert time zone
def convert_utc_to_est(utc_dt: datetime) -> datetime:
    utc = pytz.UTC
    utc_dt = utc.localize(utc_dt)
    est = pytz.timezone("US/Eastern")
    return utc_dt.astimezone(est).replace(tzinfo=None)


# get index table
def get_index_single(symbol: str, type: str) -> pl.DataFrame:
    with httpx.Client() as client:
        page_count = 0
        ticker_list = []
        cik_list = []
        timestamp_list = []
        document_url_list = []

        while True:
            query_payload = {
                "query": {
                    "query_string": {"query": f'ticker:{symbol} AND formType:"{type}"'}
                },
                "from": f"{page_count}",
                "size": f"{SIZE}",
                "sort": [{"filedAt": {"order": "desc"}}],
            }
            # whether to break
            response = client.post(
                ENDPOINT_URL.format(SEC_KEY=os.environ.get("SEC_KEY")),
                json=query_payload,
            )
            if response.status_code != 200:
                if response.status_code != 429:
                    raise RequestError(response.text)
                logger.info("[red]Hit limit[/red]")
                time.sleep(SLEEP_TIME)
                response = client.post(
                    ENDPOINT_URL.format(SEC_KEY=os.environ.get("SEC_KEY")),
                    json=query_payload,
                )
            if response.status_code != 200:
                raise RequestError(response.text)
            result = response.json()
            if len(result["filings"]) == 0:
                break
            # parse data
            for cur_record in result["filings"]:
                for cur_document in cur_record["documentFormatFiles"]:
                    if cur_document["type"] == type:
                        document_url_list.append(cur_document["documentUrl"])
                        ticker_list.append(cur_record["ticker"])
                        cik_list.append(cur_record["cik"])
                        timestamp_list.append(
                            parser.parse(cur_record["filedAt"]).replace(tzinfo=None)
                        )
                        break

            # page count
            page_count += 1 * SIZE

        df = pl.DataFrame(
            {
                "ticker": ticker_list,
                "cik": cik_list,
                "utc_timestamp": timestamp_list,
                "document_url": document_url_list,
            }
        )
        utc_times = df["utc_timestamp"].to_list()
        est_times = [convert_utc_to_est(t) for t in utc_times]
        df = df.with_columns(
            pl.Series(est_times).alias("est_timestamp"), pl.lit(type).alias("type")
        )

        return df


def get_index(symbols: List[str], type: str) -> pl.DataFrame:
    index_params = []
    index_params.extend({"symbol": e, "type": type} for e in symbols)
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        index_data = list(
            tqdm(
                executor.map(lambda x: get_index_single(**x), index_params),
                total=len(index_params),
                desc=f"Downloading index data for {type}",
            )
        )
    index_df = pl.concat([f for f in index_data if f.shape[0] > 0])
    index_df = index_df.unique(subset="document_url")
    return index_df


def request_content_single(
    client: httpx.Client,
    cur_extractor_url: str,
    max_retries: int = 3,
    timeout: int = 60
) -> Dict[str, Any]:
    """
    Make a single request to SEC API with retry mechanism
    
    Args:
        client: httpx client
        cur_extractor_url: URL to request
        max_retries: Maximum number of retry attempts
        timeout: Timeout in seconds
        
    Returns:
        Response data
    """
    for attempt in range(max_retries):
        try:
            response = client.get(cur_extractor_url, timeout=timeout)
            
            # Handle rate limiting
            if response.status_code == 429:
                wait_time = (attempt + 1) * 10  # Exponential backoff
                print(f"Rate limit hit, waiting {wait_time} seconds...")
                time.sleep(wait_time)
                continue
                
            response.raise_for_status()
            text = response.text
            return clean(text)
            
        except (httpx.ReadTimeout, httpx.ConnectTimeout) as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5
                print(f"Timeout occurred, retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Max retries reached for URL: {cur_extractor_url}")
                raise
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                print(f"URL not found: {cur_extractor_url}")
                return None
            raise
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            raise


def request_content(
    content_params: List[Dict[str, Any]],
    max_workers: int = 3,  # Reduced from 5 to 3 to avoid rate limiting
    timeout: int = 60
) -> List[Dict[str, Any]]:
    """
    Make concurrent requests to SEC API
    
    Args:
        content_params: List of parameters for requests
        max_workers: Maximum number of concurrent workers
        timeout: Timeout in seconds per request
        
    Returns:
        List of response data
    """
    with httpx.Client(timeout=timeout) as client:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            content_data = list(
                tqdm(
                    executor.map(
                        lambda x: request_content_single(client=client, **x),
                        content_params
                    ),
                    total=len(content_params),
                    desc="Downloading content"
                )
            )
    return content_data


if __name__ == "__main__":
    # load data
    unique_equities = ["AMZN"]#在这里替换股票代码

    # get file index
    ten_k_index_table = get_index(unique_equities, "10-K")
    ten_q_index_table = get_index(unique_equities, "10-Q")
    # eight_k_index_table = get_index(unique_equities, "8-K")

    # request content
    ten_k_df = (
        pl.DataFrame(
            [
                pl.Series("document_url", ten_k_index_table["document_url"].to_list()),
                pl.Series(
                    "content",
                    request_content(
                        content_params=[
                            {
                                "cur_extractor_url": EXTRACTOR_URL.format(
                                    url=url,
                                    SEC_KEY=os.environ.get("SEC_KEY"),
                                    item=item
                                )
                            }
                            for url in ten_k_index_table["document_url"].to_list()
                            for item in TEN_K_ITEM_CODE
                        ]
                    ),
                ),
            ]
        )
        .join(ten_k_index_table, on="document_url")
        .drop_nulls()
    )
    
    # Add delay between requests
    time.sleep(5)
    
    ten_q_df = (
        pl.DataFrame(
            [
                pl.Series("document_url", ten_q_index_table["document_url"].to_list()),
                pl.Series(
                    "content",
                    request_content(
                        content_params=[
                            {
                                "cur_extractor_url": EXTRACTOR_URL.format(
                                    url=url,
                                    SEC_KEY=os.environ.get("SEC_KEY"),
                                    item=item
                                )
                            }
                            for url in ten_q_index_table["document_url"].to_list()
                            for item in TEN_Q_ITEM_CODE
                        ]
                    ),
                ),
            ]
        )
        .join(ten_q_index_table, on="document_url")
        .drop_nulls()
    )
    # eight_k_df = (
    #     pl.DataFrame(
    #         [
    #             pl.Series(
    #                 "document_url", eight_k_index_table["document_url"].to_list()
    #             ),
    #             pl.Series(
    #                 "content",
    #                 request_content(
    #                     filings=eight_k_index_table["document_url"].to_list(),
    #                     sections=EIGHT_K_ITEM_CODE,
    #                 ),
    #             ),
    #         ]
    #     )
    #     .join(eight_k_index_table, on="document_url")
    #     .drop_nulls()
    # )

    # filing_data = pl.concat([ten_k_df, ten_q_df, eight_k_df])
    filing_data = pl.concat([ten_k_df, ten_q_df])
    filing_data.write_parquet(os.path.join("data", "03_primary", "filing_data2.parquet"))
    logger.info("Program ends")
