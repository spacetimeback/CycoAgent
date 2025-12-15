from puppy import LLMAgent
import pandas as pd

if __name__ == '__main__':
    agent = LLMAgent.load_checkpoint("./data/06_train_checkpoint/gpt_agent_tsla/agent_tsla/agent")
    df = agent.portfolio.get_action_df()
    
    # 检查数据是否为空
    if df is None or df.empty:
        print("警告：没有获取到数据")
    else:
        print(f"获取到 {len(df)} 行数据")
        # 使用正确的pandas方法保存数据
        df.to_csv("tsla_deepseek.csv", index=False)
        print("数据已保存到 tsla_deepseek.csv")