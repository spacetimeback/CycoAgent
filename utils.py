def build_llama2_prompt(messages):
    """
    Build a prompt string for Llama2 model from a list of messages.
    
    Args:
        messages (list): List of message dictionaries with 'role' and 'content' keys.
        
    Returns:
        str: Formatted prompt string for Llama2 model.
    """
    startPrompt = "<s>[INST] "
    endPrompt = " [/INST]"
    conversation = []
    for index, message in enumerate(messages):
        if message["role"] == "system" and index == 0:
            conversation.append(f"<<SYS>>\n{message['content']}\n<</SYS>>\n\n")
        elif message["role"] == "user":
            conversation.append(message["content"].strip())
        else:
            conversation.append(f" [/INST] {message['content'].strip()}</s><s>[INST] ")

    return startPrompt + "".join(conversation) + endPrompt 

import pickle
import os
from typing import Dict, Any, List

def inspect_state_dict(path: str) -> Dict[str, Any]:
    """
    Inspect the contents of a state_dict.pkl file
    
    Args:
        path: Path to the state_dict.pkl file
        
    Returns:
        Dictionary containing the state information
    """
    with open(path, 'rb') as f:
        state_dict = pickle.load(f)
    
    # Print basic information
    print("\n=== Basic Information ===")
    print(f"Agent Name: {state_dict['agent_name']}")
    print(f"Trading Symbol: {state_dict['trading_symbol']}")
    print(f"Top K: {state_dict['top_k']}")
    print(f"Counter: {state_dict['counter']}")
    
    # Print portfolio information
    portfolio = state_dict['portfolio']
    print("\n=== Portfolio Information ===")
    print(f"Current Cash: ${portfolio.cash:,.2f}")
    print(f"Current Holdings: {portfolio.holding_shares} shares")
    print(f"Transaction Fee Rate: {portfolio.transaction_fee_rate*100}%")
    print(f"Max Position: {portfolio.max_position} shares")
    print(f"Number of Trading Days: {len(portfolio.date_series)}")
    
    # Print model parameters
    print("\n=== Model Parameters ===")
    print(f"Dropout Rate: {state_dict.get('dropout_rate', 0.2)}")
    print(f"Text Weight: {state_dict.get('text_weight', 0.7)}")
    print(f"Price Weight: {state_dict.get('price_weight', 0.3)}")
    print(f"Lookback Window Size: {state_dict.get('look_back_window_size', 7)}")
    
    # Print reflection results
    print("\n=== Recent Reflection Results ===")
    reflection_dict = state_dict['reflection_result_series_dict']
    if reflection_dict:
        latest_date = max(reflection_dict.keys())
        latest_reflection = reflection_dict[latest_date]
        print(f"Latest Date: {latest_date}")
        print(f"Latest Reflection: {latest_reflection}")
    
    return state_dict 

def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sharpe ratio for a series of returns
    
    Args:
        returns: List of daily returns
        risk_free_rate: Annual risk-free rate (default: 2%)
        
    Returns:
        Sharpe ratio
    """
    if not returns:
        return 0.0
    
    # Convert annual risk-free rate to daily
    daily_rf = (1 + risk_free_rate) ** (1/252) - 1
    
    # Calculate excess returns
    excess_returns = [r - daily_rf for r in returns]
    
    # Calculate Sharpe ratio
    mean_excess_return = sum(excess_returns) / len(excess_returns)
    std_excess_return = (sum((r - mean_excess_return) ** 2 for r in excess_returns) / len(excess_returns)) ** 0.5
    
    if std_excess_return == 0:
        return 0.0
        
    # Annualize
    sharpe_ratio = (mean_excess_return / std_excess_return) * (252 ** 0.5)
    return sharpe_ratio

def calculate_max_drawdown(values: List[float]) -> Dict[str, float]:
    """
    Calculate maximum drawdown and related metrics
    
    Args:
        values: List of portfolio values
        
    Returns:
        Dictionary containing drawdown metrics
    """
    if not values:
        return {"max_drawdown": 0.0, "drawdown_duration": 0}
    
    max_value = values[0]
    max_drawdown = 0.0
    current_drawdown = 0.0
    drawdown_start = 0
    max_drawdown_duration = 0
    current_drawdown_duration = 0
    
    for i, value in enumerate(values):
        if value > max_value:
            max_value = value
            current_drawdown = 0.0
            current_drawdown_duration = 0
        else:
            current_drawdown = (max_value - value) / max_value
            current_drawdown_duration += 1
            
            if current_drawdown > max_drawdown:
                max_drawdown = current_drawdown
                max_drawdown_duration = current_drawdown_duration
    
    return {
        "max_drawdown": max_drawdown,
        "drawdown_duration": max_drawdown_duration
    }

def calculate_sortino_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sortino ratio for a series of returns
    
    Args:
        returns: List of daily returns
        risk_free_rate: Annual risk-free rate (default: 2%)
        
    Returns:
        Sortino ratio
    """
    if not returns:
        return 0.0
    
    # Convert annual risk-free rate to daily
    daily_rf = (1 + risk_free_rate) ** (1/252) - 1
    
    # Calculate excess returns
    excess_returns = [r - daily_rf for r in returns]
    
    # Calculate downside returns (negative returns only)
    downside_returns = [r for r in excess_returns if r < 0]
    
    if not downside_returns:
        return 0.0
    
    # Calculate Sortino ratio
    mean_excess_return = sum(excess_returns) / len(excess_returns)
    downside_std = (sum(r ** 2 for r in downside_returns) / len(downside_returns)) ** 0.5
    
    if downside_std == 0:
        return 0.0
        
    # Annualize
    sortino_ratio = (mean_excess_return / downside_std) * (252 ** 0.5)
    return sortino_ratio

def calculate_information_ratio(portfolio_returns: List[float], benchmark_returns: List[float]) -> float:
    """
    Calculate Information ratio
    
    Args:
        portfolio_returns: List of portfolio daily returns
        benchmark_returns: List of benchmark daily returns
        
    Returns:
        Information ratio
    """
    if not portfolio_returns or not benchmark_returns or len(portfolio_returns) != len(benchmark_returns):
        return 0.0
    
    # Calculate excess returns
    excess_returns = [p - b for p, b in zip(portfolio_returns, benchmark_returns)]
    
    # Calculate Information ratio
    mean_excess_return = sum(excess_returns) / len(excess_returns)
    tracking_error = (sum((r - mean_excess_return) ** 2 for r in excess_returns) / len(excess_returns)) ** 0.5
    
    if tracking_error == 0:
        return 0.0
        
    # Annualize
    information_ratio = (mean_excess_return / tracking_error) * (252 ** 0.5)
    return information_ratio

def analyze_trading_frequency(action_series: Dict) -> Dict[str, Any]:
    """
    Analyze trading frequency and patterns
    
    Args:
        action_series: Dictionary of trading actions
        
    Returns:
        Dictionary containing trading frequency metrics
    """
    if not action_series:
        return {
            "trades_per_day": 0,
            "avg_trade_interval": 0,
            "max_trades_per_day": 0,
            "trading_days_ratio": 0
        }
    
    # Group trades by date
    trades_by_date = {}
    for date, action in action_series.items():
        if action.get('direction', 0) != 0:  # If there was a trade
            if date not in trades_by_date:
                trades_by_date[date] = 0
            trades_by_date[date] += 1
    
    # Calculate metrics
    total_trades = len(action_series)
    total_days = len(trades_by_date)
    max_trades_per_day = max(trades_by_date.values()) if trades_by_date else 0
    
    # Calculate average trade interval
    dates = sorted(trades_by_date.keys())
    intervals = []
    for i in range(1, len(dates)):
        interval = (dates[i] - dates[i-1]).days
        intervals.append(interval)
    
    avg_interval = sum(intervals) / len(intervals) if intervals else 0
    
    return {
        "trades_per_day": total_trades / total_days if total_days > 0 else 0,
        "avg_trade_interval": avg_interval,
        "max_trades_per_day": max_trades_per_day,
        "trading_days_ratio": total_days / len(action_series) if action_series else 0
    }

def analyze_holding_periods(action_series: Dict) -> Dict[str, Any]:
    """
    Analyze holding periods for positions
    
    Args:
        action_series: Dictionary of trading actions
        
    Returns:
        Dictionary containing holding period metrics
    """
    if not action_series:
        return {
            "avg_holding_period": 0,
            "max_holding_period": 0,
            "min_holding_period": 0,
            "holding_period_distribution": {}
        }
    
    # Track positions and their holding periods
    positions = []
    holding_periods = []
    current_position = None
    
    for date, action in sorted(action_series.items()):
        direction = action.get('direction', 0)
        
        if direction != 0:  # If there was a trade
            if direction > 0:  # Buy
                if current_position is None:
                    current_position = {"start_date": date, "quantity": action.get('quantity', 0)}
            elif direction < 0:  # Sell
                if current_position is not None:
                    holding_period = (date - current_position["start_date"]).days
                    holding_periods.append(holding_period)
                    positions.append({
                        "start_date": current_position["start_date"],
                        "end_date": date,
                        "holding_period": holding_period,
                        "quantity": current_position["quantity"]
                    })
                    current_position = None
    
    # Calculate metrics
    if holding_periods:
        avg_holding_period = sum(holding_periods) / len(holding_periods)
        max_holding_period = max(holding_periods)
        min_holding_period = min(holding_periods)
        
        # Calculate holding period distribution
        distribution = {}
        for period in holding_periods:
            if period not in distribution:
                distribution[period] = 0
            distribution[period] += 1
    else:
        avg_holding_period = 0
        max_holding_period = 0
        min_holding_period = 0
        distribution = {}
    
    return {
        "avg_holding_period": avg_holding_period,
        "max_holding_period": max_holding_period,
        "min_holding_period": min_holding_period,
        "holding_period_distribution": distribution
    }

def evaluate_performance(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate agent performance by comparing with buy&hold strategy
    
    Args:
        state_dict: Dictionary containing the agent's state information
        
    Returns:
        Dictionary containing performance metrics
    """
    portfolio = state_dict['portfolio']
    
    # Get price series and portfolio value series
    price_series = portfolio.market_price_series
    portfolio_value_series = portfolio.net_worth_series
    date_series = portfolio.date_series
    
    if not price_series or not portfolio_value_series:
        return {"error": "No trading history available"}
    
    # Calculate daily returns
    daily_returns = []
    benchmark_returns = []
    for i in range(1, len(portfolio_value_series)):
        daily_return = (portfolio_value_series[i] - portfolio_value_series[i-1]) / portfolio_value_series[i-1]
        daily_returns.append(daily_return)
        
        # Calculate benchmark (buy&hold) returns
        benchmark_return = (price_series[i] - price_series[i-1]) / price_series[i-1]
        benchmark_returns.append(benchmark_return)
    
    # Calculate buy&hold performance
    initial_price = price_series[0]
    final_price = price_series[-1]
    buy_hold_return = (final_price - initial_price) / initial_price
    
    # Calculate agent performance
    initial_value = portfolio_value_series[0]
    final_value = portfolio_value_series[-1]
    agent_return = (final_value - initial_value) / initial_value
    
    # Calculate trading costs
    total_trading_cost = 0
    for action in portfolio.action_series.values():
        if action.get('direction', 0) != 0:  # If there was a trade
            price = action.get('price', 0)
            quantity = action.get('quantity', 0)
            cost = price * quantity * portfolio.transaction_fee_rate
            total_trading_cost += cost
    
    # Calculate win rate
    winning_trades = 0
    total_trades = 0
    for i in range(1, len(portfolio_value_series)):
        if portfolio_value_series[i] > portfolio_value_series[i-1]:
            winning_trades += 1
        total_trades += 1
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # Calculate additional metrics
    metrics = {
        "trading_period": {
            "start_date": date_series[0],
            "end_date": date_series[-1],
            "total_days": len(date_series)
        },
        "returns": {
            "agent_return": agent_return,
            "buy_hold_return": buy_hold_return,
            "outperformance": agent_return - buy_hold_return,
            "sharpe_ratio": calculate_sharpe_ratio(daily_returns),
            "sortino_ratio": calculate_sortino_ratio(daily_returns),
            "information_ratio": calculate_information_ratio(daily_returns, benchmark_returns)
        },
        "risk_metrics": calculate_max_drawdown(portfolio_value_series),
        "portfolio_stats": {
            "initial_value": initial_value,
            "final_value": final_value,
            "max_value": max(portfolio_value_series),
            "min_value": min(portfolio_value_series)
        },
        "trading_stats": {
            "total_trades": len(portfolio.action_series),
            "buy_trades": sum(1 for action in portfolio.action_series.values() if action.get('direction', 0) > 0),
            "sell_trades": sum(1 for action in portfolio.action_series.values() if action.get('direction', 0) < 0),
            "win_rate": win_rate,
            "total_trading_cost": total_trading_cost,
            "avg_trade_cost": total_trading_cost / len(portfolio.action_series) if portfolio.action_series else 0
        },
        "trading_frequency": analyze_trading_frequency(portfolio.action_series),
        "holding_periods": analyze_holding_periods(portfolio.action_series)
    }
    
    # Print performance summary
    print("\n=== Performance Evaluation ===")
    print(f"Trading Period: {metrics['trading_period']['start_date']} to {metrics['trading_period']['end_date']}")
    print(f"Total Trading Days: {metrics['trading_period']['total_days']}")
    
    print("\nReturns:")
    print(f"Agent Return: {metrics['returns']['agent_return']*100:.2f}%")
    print(f"Buy&Hold Return: {metrics['returns']['buy_hold_return']*100:.2f}%")
    print(f"Outperformance: {metrics['returns']['outperformance']*100:.2f}%")
    print(f"Sharpe Ratio: {metrics['returns']['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio: {metrics['returns']['sortino_ratio']:.2f}")
    print(f"Information Ratio: {metrics['returns']['information_ratio']:.2f}")
    
    print("\nRisk Metrics:")
    print(f"Maximum Drawdown: {metrics['risk_metrics']['max_drawdown']*100:.2f}%")
    print(f"Drawdown Duration: {metrics['risk_metrics']['drawdown_duration']} days")
    
    print("\nPortfolio Statistics:")
    print(f"Initial Value: ${metrics['portfolio_stats']['initial_value']:,.2f}")
    print(f"Final Value: ${metrics['portfolio_stats']['final_value']:,.2f}")
    print(f"Maximum Value: ${metrics['portfolio_stats']['max_value']:,.2f}")
    print(f"Minimum Value: ${metrics['portfolio_stats']['min_value']:,.2f}")
    
    print("\nTrading Statistics:")
    print(f"Total Trades: {metrics['trading_stats']['total_trades']}")
    print(f"Buy Trades: {metrics['trading_stats']['buy_trades']}")
    print(f"Sell Trades: {metrics['trading_stats']['sell_trades']}")
    print(f"Win Rate: {metrics['trading_stats']['win_rate']*100:.2f}%")
    print(f"Total Trading Cost: ${metrics['trading_stats']['total_trading_cost']:,.2f}")
    print(f"Average Trade Cost: ${metrics['trading_stats']['avg_trade_cost']:,.2f}")
    
    print("\nTrading Frequency:")
    print(f"Trades per Day: {metrics['trading_frequency']['trades_per_day']:.2f}")
    print(f"Average Trade Interval: {metrics['trading_frequency']['avg_trade_interval']:.1f} days")
    print(f"Maximum Trades per Day: {metrics['trading_frequency']['max_trades_per_day']}")
    print(f"Trading Days Ratio: {metrics['trading_frequency']['trading_days_ratio']*100:.2f}%")
    
    print("\nHolding Period Analysis:")
    print(f"Average Holding Period: {metrics['holding_periods']['avg_holding_period']:.1f} days")
    print(f"Maximum Holding Period: {metrics['holding_periods']['max_holding_period']} days")
    print(f"Minimum Holding Period: {metrics['holding_periods']['min_holding_period']} days")
    
    return metrics 