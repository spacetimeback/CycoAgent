export OPENAI_API_KEY="sk-4eoxi0M6Qi8PkRjBYgwMKefLP2WjQhrltLTQHGsrygs659xO"

# Create logs directory if it doesn't exist
mkdir -p logs

# gpt
# train
#python run.py sim \
#-mdp data/03_model_input/tsla.pkl \
#-st 2023-08-15 \
#-et 2024-12-31 \
#-rm train \
#-cp config/tsla_gpt_config.toml \
#-ckp data/06_train_checkpoint/gemini_agent_tsla/agent_tsla \
#-rp data/05_train_model_output > logs/tsla_gpt_train.log 2>&1

# train-checkpoint
#python run.py sim-checkpoint \
#   -ckp ./data/06_train_checkpoint/gemini_agent_tsla \
#    -rp data/05_train_model_output \
#    -cp config/tsla_gpt_config.toml \
#    -rm train > logs/tsla_gpt_train.log 2>&1

# # test
python run.py sim \
-mdp data/03_model_input/tsla.pkl \
-st 2025-01-01 \
-et 2025-03-28 \
-rm test \
-cp config/tsla_gpt_config.toml \
-tap data/06_train_checkpoint/gemini_agent_tsla/agent_tsla/agent \
-ckp ./data/08_test_checkpoint/gemini_agent_tsla \
-rp data/09_results/gemini_agent_tsla > logs/tsla_gpt_test.log 2>&1

# test-checkpoint
#python run.py sim-checkpoint \
#-rm test \
#-ckp ./data/08_test_checkpoint \
# -rp ./data/09_results > logs/tsla_gpt_test.log 2>&1

# 保存测试结果
#python save_file.py