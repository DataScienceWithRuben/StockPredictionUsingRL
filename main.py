import pandas as pd
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import A2C

class StockTradingEnv(gym.Env):
    def __init__(self, df):
        super(StockTradingEnv, self).__init__()

        self.df = df
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(5,))

    def step(self, action):

        next_step = self.current_step + 1

        if action == 0:  # hold
            reward = 0.5
        elif action == 1:  # buy
            reward = self.df.loc[next_step, 'Close'] - self.df.loc[self.current_step, 'Close']
        else:  # sell
            reward = self.df.loc[self.current_step, 'Close'] - self.df.loc[next_step, 'Close']


        done = next_step == len(self.df.index) - 1
        self.current_step += 1
        obs = self.df.loc[self.current_step, :].values
        return obs, reward, done, {}

    def reset(self):
        self.current_step = 0
        return self.df.loc[self.current_step, :].values

# Let's assume df_stock is your data frame
df_stock = pd.read_csv('stock_data.csv')
df_stock = df_stock.sort_values('Date')

# Drop the 'Date' column
df_stock = df_stock.drop(columns=['Date'])

env = StockTradingEnv(df_stock)
model = A2C('MlpPolicy', env, verbose=1)python
model.learn(total_timesteps=10000)

# Let's assume df_new_stock is the new data
df_new_stock = pd.DataFrame({
    'Open': [163.82],
    'High': [162.22],
    'Low': [164.18],
    'Close': [162.34],
    'Volume': [20726900]
}, index=[0])

# Reset the environment with the new data
env_new_data = StockTradingEnv(df_new_stock)
obs = env_new_data.reset()

# Get the model's prediction
action, _states = model.predict(obs)
action_dict = {0: "Hold", 1: "Buy", 2: "Sell"}

# Print the action
print(action_dict[int(action)])
