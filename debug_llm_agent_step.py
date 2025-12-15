import datetime
from puppy.agent import LLMAgent
from puppy.run_type import RunMode
import logging

# ========= 配置加载 =========
# 假设你已有一个 config dict，或从 YAML 读取
from puppy.config import load_config  # 你需要换成你自己的加载逻辑
cfg = load_config("config/tsla_gpt_config.toml")      # 替换成你的路径

# ========= 初始化 logger =========
logger = logging.getLogger("debug_llm")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# ========= 初始化 Agent =========
agent = LLMAgent.from_config(cfg)

# ========= 构造 market_info 输入 =========
market_info = {
    "date": datetime.date(2023, 8, 16),
    "price": {
        "open": 250.0,
        "close": 260.0,
        "high": 265.0,
        "low": 245.0,
        "change": 10.0,
        "pct_change": 0.04,
        "vwap": 255.0,
        "volume": 600000,
        "amount": 1.2e8,
        "turnover_ratio": 0.04,
        "total_mv": 900e9,
        "pe": 75.0,
        "pb": 10.0,
    },
    "filing_k": "Tesla announced a new gigafactory in India.",
    "filing_q": "Quarterly earnings show 20% YoY revenue increase.",
    "news": [
        "Tesla collaborates with Samsung on FSD chip development.",
        "Elon Musk hints at Model 2 prototype reveal next quarter."
    ],
    "earnings_calls": [],
}

# ========= 手动触发一个 step =========
logger.info(">>> Triggering agent.step() in Train mode...")
agent.step(market_info=market_info, run_mode=RunMode.Train)

# ========= 打印 portfolio 状态 =========
logger.info(">>> Latest portfolio state:")
logger.info(agent.portfolio.history[-1] if agent.portfolio.history else "No action history recorded.")
