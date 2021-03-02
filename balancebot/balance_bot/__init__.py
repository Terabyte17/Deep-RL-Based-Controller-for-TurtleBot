from gym.envs.registration import register

register(id='balancebot-PID-v0', entry_point='balance_bot.envs:BalanceBot')