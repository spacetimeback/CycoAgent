import os
import toml
import typer
import logging
import pickle
import warnings
import faulthandler
import sys
from tqdm import tqdm
from dotenv import load_dotenv
from datetime import datetime
from typing import Union, Optional
from puppy import MarketEnvironment, LLMAgent, RunMode
from puppy.agent import LLMAgent
from puppy.environment import MarketEnvironment
from puppy.run_type import RunMode
import time

# 启用 faulthandler
faulthandler.enable()
# 设置 faulthandler 输出文件
faulthandler.dump_traceback_later(timeout=60)

# set up
load_dotenv()
app = typer.Typer(name="puppy")
warnings.filterwarnings("ignore")


@app.command("sim", help="Start Simulation", rich_help_panel="Simulation")
def sim_func(
    market_data_info_path: str = typer.Option(
        os.path.join("data", "03_model_input", "processed_tsla.pkl"),
        "-mdp",
        "--market-data-path",
        help="The environment data pickle path",
    ),
    start_time: str = typer.Option(
        "2023-08-15", "-st", "--start-time", help="The start time"
    ),
    end_time: str = typer.Option(
        "2024-12-31", "-et", "--end-time", help="The end time"
    ),
    run_mode: str = typer.Option(
        "train", "-rm", "--run-model", help="Run mode: train or test"
    ),
    config_path: str = typer.Option(
        os.path.join("config", "tsla_gpt_config.toml"),
        "-cp",
        "--config-path",
        help="config file path",
    ),
    checkpoint_path: str = typer.Option(
        os.path.join("data", "06_train_checkpoint"),
        "-ckp",
        "--checkpoint-path",
        help="The checkpoint path",
    ),
    result_path: str = typer.Option(
        os.path.join("data", "05_train_model_output"),
        "-rp",
        "--result-path",
        help="The result save path",
    ),
    trained_agent_path: Union[str, None] = typer.Option(
        None,
        "-tap",
        "--trained-agent-path",
        help="Only used in test mode, the path of trained agent",
    ),
) -> None:
    import traceback
    import sys
    
    try:
        # Initialize logger first
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # Create log directory if it doesn't exist
        log_dir = os.path.join("data", "04_model_output_log")
        os.makedirs(log_dir, exist_ok=True)
        
        # Add file handler with rotation
        logging_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            os.path.join(log_dir, f"run_{run_mode}.log"),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,  # Keep 5 backup files
            mode="a",
        )
        file_handler.setFormatter(logging_formatter)
        logger.addHandler(file_handler)
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging_formatter)
        logger.addHandler(console_handler)
        
        # Log start of simulation
        logger.info(f"Starting simulation in {run_mode} mode")
        logger.info(f"Config path: {config_path}")
        logger.info(f"Market data path: {market_data_info_path}")
        logger.info(f"Result path: {result_path}")
        logger.info(f"Time range: {start_time} to {end_time}")
        
        # Load configuration
        try:
            with open(config_path, "r") as f:
                config = toml.load(f)
            logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise
            
        # 从配置中获取股票代码
        symbol = config["general"]["trading_symbol"].lower()
        logger.info(f"Trading symbol: {symbol}")
        
        # 如果没有指定检查点路径，使用默认路径
        if checkpoint_path is None:
            checkpoint_path = os.path.join("data", "08_test_checkpoint", f"agent_{symbol}")
        if result_path is None:
            result_path = os.path.join("data", "05_train_model_output", f"agent_{symbol}")
            
        # 确保目录存在
        os.makedirs(checkpoint_path, exist_ok=True)
        os.makedirs(result_path, exist_ok=True)
        
        # verify run mode
        logger.info("Verifying run mode...")
        if run_mode in {"train", "test"}:
            run_mode_var = RunMode.Train if run_mode == "train" else RunMode.Test
            logger.info(f"Run mode set to: {run_mode_var}")
        else:
            raise ValueError("Run mode must be train or test")
            
        # create environment
        logger.info("Creating environment...")
        try:
            # 检查是否存在环境 checkpoint
            env_checkpoint_path = os.path.join(checkpoint_path, "env")
            if os.path.exists(env_checkpoint_path):
                logger.info(f"Loading environment from checkpoint: {env_checkpoint_path}")
                environment = MarketEnvironment.load_checkpoint(path=env_checkpoint_path)
                logger.info("Environment loaded successfully from checkpoint")
            else:
                logger.info(f"Loading market data from: {market_data_info_path}")
                with open(market_data_info_path, "rb") as f:
                    env_data_pkl = pickle.load(f)
                logger.info("Market data loaded successfully")
                
                environment = MarketEnvironment(
                    symbol=config["general"]["trading_symbol"],
                    env_data_pkl=env_data_pkl,
                    start_date=datetime.strptime(start_time, "%Y-%m-%d").date(),
                    end_date=datetime.strptime(end_time, "%Y-%m-%d").date(),
                )
                logger.info("Environment created successfully")
        except Exception as e:
            logger.error(f"Error creating environment: {str(e)}")
            logger.error(f"Market data path: {market_data_info_path}")
            logger.error(f"Current directory contents: {os.listdir('.')}")
            raise
        
        # create agent
        logger.info("Creating agent...")
        try:
            # 构建正确的 agent checkpoint 路径
            agent_checkpoint_path = os.path.join(checkpoint_path, f"agent_{symbol}")
            if run_mode_var == RunMode.Train:
                # 检查是否存在 checkpoint
                if os.path.exists(agent_checkpoint_path):
                    logger.info(f"Loading agent from checkpoint: {agent_checkpoint_path}")
                    # 先创建 agent 实例
                    the_agent = LLMAgent.from_config(config)
                    # 然后加载检查点
                    the_agent.load_checkpoint(path=agent_checkpoint_path)
                    logger.info("Agent loaded successfully from checkpoint")
                else:
                    logger.info("Initializing new agent from config...")
                    logger.info(f"Config keys: {config.keys()}")
                    the_agent = LLMAgent.from_config(config)
                    logger.info("Agent initialized successfully")
            else:  # Test mode
                if trained_agent_path is None:
                    # 如果没有指定训练好的 agent 路径，使用默认路径
                    trained_agent_path = os.path.join("data", "06_train_checkpoint", f"agent_{symbol}")
                    logger.info(f"Using default trained agent path: {trained_agent_path}")
                
                if not os.path.exists(trained_agent_path):
                    raise FileNotFoundError(f"Trained agent not found at: {trained_agent_path}")
                
                logger.info(f"Loading agent from checkpoint: {trained_agent_path}")
                # 先创建 agent 实例
                the_agent = LLMAgent.from_config(config)
                # 然后加载检查点
                the_agent.load_checkpoint(path=trained_agent_path)
                logger.info("Agent loaded successfully")
        except Exception as e:
            logger.error(f"Error creating agent: {str(e)}")
            logger.error("Traceback:")
            traceback.print_exc()
            raise
            
        # start simulation
        logger.info("Starting simulation loop...")
        total_steps = environment.simulation_length
        current_step = 0
        max_retries = 3  # 最大重试次数
        retry_delay = 5  # 重试延迟（秒）
        
        while current_step < total_steps:
            retry_count = 0
            while retry_count < max_retries:
                try:
                    logger.info(f"Step {the_agent.counter}")
                    the_agent.counter += 1
                    
                    logger.info(f"Processing step {current_step + 1}/{total_steps}")
                    market_info = environment.step()
                    logger.info(f"Date {market_info[0]}")
                    logger.info(f"Price {market_info[1]}")
                    
                    if market_info[6]:  # done flag is at index 6
                        logger.info("Simulation completed!")
                        break
                        
                    logger.info("Running agent step...")
                    the_agent.step(market_info=market_info, run_mode=run_mode_var)
                    
                    # 更新进度
                    current_step += 1
                    progress = (current_step / total_steps) * 100
                    # 使用 print 显示进度到控制台
                    print(f"\rProgress: {progress:.1f}% ({current_step}/{total_steps})", end="")
                    
                    # 每100步保存一次检查点
                    if current_step % 100 == 0:
                        logger.info("Saving checkpoint...")
                        the_agent.save_checkpoint(path=checkpoint_path, force=True)
                        environment.save_checkpoint(path=checkpoint_path, force=True)
                    
                    # 如果成功执行，跳出重试循环
                    break
                    
                except Exception as e:
                    retry_count += 1
                    logger.error(f"Error in simulation step: {str(e)}")
                    logger.error("Traceback:")
                    traceback.print_exc()
                    
                    if retry_count < max_retries:
                        logger.info(f"Retrying in {retry_delay} seconds... (Attempt {retry_count + 1}/{max_retries})")
                        time.sleep(retry_delay)
                    else:
                        logger.error("Max retries exceeded. Saving checkpoint and exiting...")
                        # 保存当前状态
                        the_agent.save_checkpoint(path=checkpoint_path, force=True)
                        environment.save_checkpoint(path=checkpoint_path, force=True)
                        raise  # 重新抛出异常
                
        logger.info("Saving final results...")
        the_agent.save_checkpoint(path=result_path, force=True)
        environment.save_checkpoint(path=result_path, force=True)
        logger.info("Simulation completed successfully!")
        
    except Exception as e:
        logger.error(f"Fatal error in simulation: {str(e)}")
        logger.error("Traceback:")
        traceback.print_exc()
        raise
    finally:
        # Cleanup resources
        try:
            if 'the_agent' in locals():
                the_agent.cleanup()
            if 'environment' in locals():
                environment.cleanup()
            if 'logger' in locals():
                logger.info("Simulation completed")
                for handler in logger.handlers[:]:
                    handler.close()
                    logger.removeHandler(handler)
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")


@app.command(
    "sim-checkpoint",
    help="Start Simulation from checkpoint",
    rich_help_panel="Simulation",
)
def sim_checkpoint(
    checkpoint_path: str = typer.Option(
        os.path.join("data", "06_train_checkpoint"),
        "-ckp",
        "--checkpoint-path",
        help="The checkpoint path",
    ),
    result_path: str = typer.Option(
        os.path.join("data", "05_train_model_output"),
        "-rp",
        "--result-path",
        help="The result save path",
    ),
    config_path: str = typer.Option(
        os.path.join("config", "aapl_tgi_config.toml"),
        "-cp",
        "--config-path",
        help="config file path",
    ),
    run_mode: str = typer.Option(
        "train", "-rm", "--run-model", help="Run mode: train or test"
    ),
) -> None:
    # load config
    config = toml.load(config_path)
    # set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logging_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(
        os.path.join(
            "data",
            "04_model_output_log",
            f'{config["general"]["trading_symbol"]}_run.log',
        ),
        mode="a",
    )
    file_handler.setFormatter(logging_formatter)
    logger.addHandler(file_handler)
    # verify run mode
    if run_mode in {"train", "test"}:
        run_mode_var = RunMode.Train if run_mode == "train" else RunMode.Test
    else:
        raise ValueError("Run mode must be train or test")
    # load env & agent from checkpoint
    environment = MarketEnvironment.load_checkpoint(
        path=os.path.join(checkpoint_path, "env")
    )
    the_agent = LLMAgent.load_checkpoint(path=os.path.join(checkpoint_path, "agent_1"))
    pbar = tqdm(total=environment.simulation_length)
    # run simulation
    while True:
        logger.info(f"Step {the_agent.counter}")
        the_agent.counter += 1
        market_info = environment.step()
        if market_info[6]:
            break
        the_agent.step(market_info=market_info, run_mode=run_mode_var)  # type: ignore
        pbar.update(1)
        # save checkpoint every time, openai api is not stable
        the_agent.save_checkpoint(path=checkpoint_path, force=True)
        environment.save_checkpoint(path=checkpoint_path, force=True)
    # save result after finish
    the_agent.save_checkpoint(path=result_path, force=True)
    environment.save_checkpoint(path=result_path, force=True)


if __name__ == "__main__":
    app()