import os
import shutil
import pickle
from datetime import date
from typing import List, Dict, Tuple, Union, Any
from pydantic import BaseModel, ValidationError
import logging

# type alias
market_info_type = Tuple[
    date,  # cur date
    float,  # cur price
    Union[str, None],  # cur filing_k
    Union[str, None],  # cur filing_q
    List[str],  # cur news
    float,  # cur record
    bool,  # termination flag
]
terminated_market_info_type = Tuple[None, None, None, None, None, None, bool]


# env data structure validation
class NewsItem(BaseModel):
    content: str
    sentiment: Dict[str, float]

class OneDateRecord(BaseModel):
    price: Dict[str, float] = {}  # 包含open, high, low, close等价格数据
    filing_k: str = ""  # 10-K文件
    filing_q: str = ""  # 10-Q文件
    news: List[str] = []  # 新闻列表
    earnings_calls: List[str] = []  # 财报电话会议记录
    record: float = 0.0  # 默认值为0.0
    done: bool = False  # 默认值为False


class MarketEnvironment:
    def __init__(
        self,
        env_data_pkl: Dict[date, Dict[str, Any]],
        start_date: date,
        end_date: date,
        symbol: str,
    ) -> None:
        # validate structure
        first_date = list(env_data_pkl.keys())[0]
        if not isinstance(first_date, date):
            raise TypeError("env_data_pkl keys must be date type")
        try:
            OneDateRecord.model_validate(env_data_pkl[first_date])
        except ValidationError as e:
            raise e
            
        # 包含所有日期，包括周末和节假日
        self.date_series = [
            date for date in env_data_pkl.keys()
            if (date >= start_date) and (date <= end_date)
        ]
        
        if not self.date_series:
            raise ValueError("No dates found in the date range")
            
        self.date_series = sorted(self.date_series)
        self.date_series_keep = self.date_series.copy()
        self.simulation_length = len(self.date_series)
        self.start_date = start_date
        self.end_date = end_date
        self.cur_date = None
        self.env_data = env_data_pkl
        self.symbol = symbol

    def reset(self) -> None:
        self.date_series = [
            i
            for i in self.date_series_keep
            if (i >= self.start_date) and (i <= self.end_date)
        ]
        self.date_series = sorted(self.date_series)
        self.cur_date = None

    def step(self) -> Union[market_info_type, terminated_market_info_type]:
        try:
            # Get current date
            if not self.date_series:
                logging.getLogger(__name__).info("No more dates in series - simulation complete")
                return None, None, None, None, None, None, True
                
            self.cur_date = self.date_series.pop(0)  # type: ignore
            
            # 获取当前日期的数据
            cur_data = self.env_data.get(self.cur_date, {})
            cur_price = cur_data.get("price", {})
            cur_close = cur_price.get("close") if cur_price else None
            
            # 如果没有价格数据，使用前一天的收盘价或默认值
            if cur_close is None:
                logging.getLogger(__name__).info(f"No price data for {self.cur_date}, using previous close price")
                # 尝试获取前一天的收盘价
                prev_date = None
                prev_close = None
                for date in sorted(self.env_data.keys(), reverse=True):
                    if date < self.cur_date:
                        prev_data = self.env_data.get(date, {})
                        prev_price = prev_data.get("price", {})
                        prev_close = prev_price.get("close") if prev_price else None
                        if prev_close is not None:
                            prev_date = date
                            break
                
                # 如果找到前一天的收盘价，使用它
                if prev_close is not None:
                    cur_close = prev_close
                    logging.getLogger(__name__).info(f"Using previous close price from {prev_date}: {cur_close}")
                else:
                    # 如果没有前一天的收盘价，使用默认值
                    cur_close = 0.0
                    logging.getLogger(__name__).info(f"No previous close price found, using default value: {cur_close}")
            
            # 获取未来价格
            future_date = None
            future_price = None
            future_close = None
            
            # 查找下一个有价格数据的日期
            for next_date in self.date_series:
                next_data = self.env_data.get(next_date, {})
                next_price = next_data.get("price", {})
                next_close = next_price.get("close") if next_price else None
                
                if next_close is not None:
                    future_date = next_date
                    future_close = next_close
                    break
                
            # 如果没有找到未来价格，使用当前价格
            if future_close is None:
                future_close = cur_close
                future_date = self.cur_date
            
            cur_record = future_close - cur_close
            
            # 获取其他数据
            cur_filing_k = cur_data.get("filing_k")
            cur_filing_q = cur_data.get("filing_q")
            cur_news = cur_data.get("news", [])
            
            # 检查是否是最后一天
            is_last_day = len(self.date_series) == 0
            if is_last_day:
                logging.getLogger(__name__).info(f"Last trading day reached: {self.cur_date}")
            
            return (
                self.cur_date,
                cur_close,
                cur_filing_k if cur_filing_k else None,
                cur_filing_q if cur_filing_q else None,
                cur_news,
                cur_record,
                is_last_day,
            )
            
        except IndexError:
            logging.getLogger(__name__).info("Index error - no more dates available")
            return None, None, None, None, None, None, True

    def save_checkpoint(self, path: str, force: bool = False) -> None:
        path = os.path.join(path, "env")
        if os.path.exists(path):
            if force:
                shutil.rmtree(path)
            else:
                raise FileExistsError(f"Path {path} already exists")
        os.mkdir(path)
        with open(os.path.join(path, "env.pkl"), "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_checkpoint(cls, path: str) -> "MarketEnvironment":
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path {path} does not exists")
        with open(os.path.join(path, "env.pkl"), "rb") as f:
            env = pickle.load(f)
        # update
        env.simulation_length = len(env.date_series)
        return env

    def cleanup(self) -> None:
        """Clean up any resources used by the environment."""
        # Reset the environment state
        self.reset()
        # Clear any cached data
        self.env_data = {}
        self.date_series = []
        self.date_series_keep = []
        self.cur_date = None