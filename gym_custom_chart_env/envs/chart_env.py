from collections import deque
from datetime import datetime
import gymnasium as gym
from gymnasium import spaces
import random
import numpy as np
import pickle 

with open(r'data\train_data\df_final_raw.pkl', 'rb') as f:
    df = pickle.load(f)

class ChartEnv(gym.Env):
    def __init__(self, 
                 action_num=3, 
                 position_interval=1,
                 chart_data=df, 
                 risk_adverse= 1.3,
                 init_balance=100000000, 
                 transaction=0.0004, 
                 max_leverage=10,
                 stop_loss=0.8,
                 max_position_length = 3600,
                 use_random_start = True,
                seed = None):
        
        super(ChartEnv, self).__init__()
        self.chart_data = chart_data 
        self.mid_price_index = self.chart_data.columns.get_loc("mid_price")
        self.chart_len = len(self.chart_data)
        self.use_random_start = use_random_start
        
        if self.use_random_start == False:
            self.idx = 0
        elif self.use_random_start == True:
            random.seed(seed)
            self.idx = random.randint(0, self.chart_len)
        
        self.init_balance = init_balance
        self.risk_adverse = risk_adverse  
        self.transaction = transaction # transaction cost
        self.max_leverage = max_leverage
        self.stop_loss = stop_loss
        self.max_position_length = max_position_length
        
        # below two variable is exist for plot
        self.position_portfolio_value = [self.init_balance]
        self.position_bitcoin_value = [self.chart_data.iloc[self.idx, self.mid_price_index]]
        
        
        self.current_position = {
            "position_size": 0,
            "position_len": 0,
            "position_cash": self.init_balance,
            "position_bitcoin": 0,
            "position_pnl": 0,
            "position_return_array": np.array([]),
            "position_value": self.init_balance
        } 
        
        self.next_position = {
            "position_size": 0,
            "position_len": 0,
            "position_cash": self.init_balance,
            "position_bitcoin": 0,
            "position_pnl": 0,
            "position_return_array": np.array([]),
            "position_value": self.init_balance
        } 

        self.action_space = spaces.Discrete(action_num)
        self.action_position = [position_interval*(multiple_position - action_num//2) for multiple_position in range(action_num)] 
        self.observation_space = spaces.Box(low=-2**60, high=2**60, shape=[self.chart_data.shape[1]], dtype=np.float64)
        
    def step(self, action):
        """
        Env receive input 'action'. 'action' is converted to position corresponding to action.
        But Position is not real 'acting position' because of Env's "Max_leverage".
        So position is converted to 'acting position'.
        'acting position' work in next state.
        
        Env return Reward only when current position is executed(cleared). == Sparse Reward Environment
        Reward mean is position's Sharpe ratio when execute my position at next_state  

        if position is exeuted:
            env return reward
            reset current_position, next_position

        you can see execute condition in self.check_trading_reset_condition()
        """
        self.current_position = self.next_position.copy()
        
        self.current_chart = self.chart_data.iloc[self.idx]
        self.next_chart = self.chart_data.iloc[self.idx+1]                    
        self.current_price = self.chart_data.iloc[self.idx, self.mid_price_index]
        self.next_price = self.chart_data.iloc[self.idx+1, self.mid_price_index]
        
        # check env's stop condition
        reset_by_stop_loss, reset_by_max_position_length, reset_by_index_bound = self.check_trading_reset_condition()


        if (reset_by_stop_loss==False) and (reset_by_max_position_length==False):
            acting_position, is_execute = self.get_position(action)
            self.update_next_position(acting_position, is_execute)
            
            # only get reward when position is executed (sparse reward)  
            if is_execute:
                reward = self.get_reward()
                self.reset_position()
                position_done = True
            elif not is_execute:
                reward = 0 
                position_done = False
                               
            self.position_bitcoin_value.append(self.current_price)
            self.position_portfolio_value.append(self.current_position["position_value"])
            
            self.idx += 1
            done = False
        
        elif (reset_by_max_position_length==True) and (reset_by_stop_loss==False):
            # only get reward when position is executed (sparse reward)  
            reward = self.get_reward()
            self.reset_position()
            self.idx += 1
            
            # just reset position, not reset env
            position_done = True
            done = False
            
        elif (reset_by_stop_loss==True):
            # only get reward when position is executed (sparse reward)  
            reward = self.get_reward()
            self.reset_position()

            # reset position and env
            position_done = True
            done = True    

        # if index > data length: => env.reset()              
        if reset_by_index_bound==True:
            reward = self.get_reward()
            self.reset_position()
            
            # reset position and env
            position_done = True
            done = True    
            
        return np.array(self.next_chart), \
                reward, \
                done, \
                not done, \
                {"state":self.current_chart,
                "next_state":self.next_chart,
                "reward": reward,
                "current_position": self.current_position,
                "next_position": self.next_position,
                "position_done" : position_done,
                "position_value": self.position_portfolio_value,
                "position_value": self.position_bitcoin_value,
                "done": done
        }
            
        
    
    def get_reward(self):
        """
        return reward
        reward is position's sharpe ratio
        """
        pnl = self.next_position["position_pnl"]
        std = np.std(self.next_position["position_return_array"])
        sharpe_ratio = pnl / std
        print(pnl, std)
        if sharpe_ratio < 0:
            sharpe_ratio = -abs(sharpe_ratio) ** self.risk_adverse
        
        return sharpe_ratio
    
    
    
    def check_trading_reset_condition(self):
        """
        check reset codition
        1) reset_by_stop_loss -> execute position & reset env
        2) reset_by_max_position_length -> only execute position
        3) reset_by_index_bound -> execute position & reset env
        """
        reset_by_stop_loss = False
        reset_by_max_position_length = False
        reset_by_index_bound = False
        
        if self.current_position["position_value"] < self.init_balance * self.stop_loss:
            reset_by_stop_loss = True
        
        if self.current_position["position_len"] >= self.max_position_length:
            reset_by_max_position_length = True
            
        if self.idx+1 >= self.chart_data.shape[0]:
            reset_by_index_bound = True
        
        return reset_by_stop_loss, reset_by_max_position_length, reset_by_index_bound 
    
    
    def get_position(self, action): 
        """
        input: Agent's action
        output: realised action
        
        return 
            (1) acting position considering max leverage
            (2) execute <- execute when position sign is changed
             
        ex) max_leverage = 3, current_position = 2, current_action = 3
            -> possible_action = min(max_leverage-current_position, next_action) = 1
            -> return 1        
        """
        
        current_position = self.current_position["position_size"]
        next_action = self.action_position[action]
        execute = False
        if current_position>0:
            if next_action>=0:
                acting_positon = min(self.max_leverage-current_position, next_action)
            elif next_action<0:
                acting_positon = -current_position
                execute = True
                
        elif current_position==0:
            if next_action>0:
                acting_positon = min(self.max_leverage-current_position, next_action)
            elif next_action==0:
                acting_positon = 0
            elif next_action<0:
                acting_positon = max(-self.max_leverage+current_position, next_action)
                
        elif current_position < 0:
            if next_action > 0:
                acting_positon = -current_position
                execute = True
            elif next_action <= 0:
                acting_positon = max(-self.max_leverage-current_position, next_action)

        
        return acting_positon, execute
    
    
    def update_next_position(self, acting_position, is_execute):
        """
        update next_state_position
        
        input: acting_position
        ouput: None
        """
        self.next_position["position_size"] = self.current_position["position_size"] + acting_position 
        self.next_position["position_len"] += 1
        
        cash, bitcoin = self.transaction_result(acting_position)
        
        self.next_position["position_cash"] = cash
        self.next_position["position_bitcoin"] = bitcoin
        self.next_position["position_value"] = self.next_position["position_bitcoin"]*self.next_price + self.next_position["position_cash"]   
        self.next_position["position_pnl"] = np.log(self.next_position["position_value"]) - np.log(self.init_balance)
        
        return_array = self.next_position["position_return_array"]
        log_return = np.log(self.next_position["position_value"]) - np.log(self.current_position["position_value"])
        self.next_position["position_return_array"] = np.append(return_array, log_return)
        return 
    
    
    def transaction_result(self, acting_position):
        """
        input: position_size, acting_position
        output: cash, bitcoin after transaction
        
        transaction budget = acting_position * init_budget
        
        not allow change class variable only allow return 
        """
        current_position = self.current_position["position_size"]
        current_cash = self.current_position["position_cash"]
        current_bitcoin = self.current_position["position_bitcoin"]
        
        return_cash = 0
        return_bitcoin = 0
        
        # neutral
        if acting_position == 0:
            return_cash = current_cash
            return_bitcoin = current_bitcoin
        
        # current_position = long 
        if acting_position > 0:
            # transaction_budget: (+)
            transaction_budget = self.init_balance * acting_position
            
            # long(buy) bitcoin (buy bitcoin)
            if current_position >= 0:
                return_cash = current_cash - transaction_budget * (1+self.transaction)
                return_bitcoin =  current_bitcoin + transaction_budget / self.next_price
            
            # clear current long position (sell bitcoin) 
            # i think calculating 'acting position' as same as '-current position' make error when i clear current position   
            # so i just clear my bitcoin 
            elif current_position < 0:
                return_cash = current_cash + (current_bitcoin) * self.next_price * (1-self.transaction)
                return_bitcoin = 0
                
        # current_position = short                  
        if acting_position < 0:
            # transaction_budget: (-) 
            transaction_budget = self.init_balance * acting_position
            # i think calculating 'acting position' as same as '-current position' make error when i clear current position   
            # so i just clear my bitcoin 
            # clear current short position (buy bitcoin)
            if current_position > 0:
                return_cash = current_cash + (current_bitcoin) * self.next_price * (1+self.transaction)
                return_bitcoin =  0
                
            # short(sell) bitcoin (sell bitcoin)
            elif current_position <= 0:
                return_cash = current_cash - transaction_budget * (1-self.transaction)
                return_bitcoin = current_bitcoin + transaction_budget / self.next_price
        
        return return_cash, return_bitcoin
    
       
    def reset_position(self):
        """
        This function just reset position.
        This mean is that <func> reset_position() != <func> reset()
        
        you can get more info about reset in below <func> check_trading_reset_condition()
        """
        
        self.position_portfolio_value = [self.init_balance]
        self.position_bitcoin_value = [self.chart_data.iloc[self.idx, self.mid_price_index]]
        
        
        self.current_position = {
            "position_size": 0,
            "position_len": 0,
            "position_cash": self.init_balance,
            "position_bitcoin": 0,
            "position_pnl": 0,
            "position_return_array": np.array([]),
            "position_value": self.init_balance
        } 
        
        self.next_position = {
            "position_size": 0,
            "position_len": 0,
            "position_cash": self.init_balance,
            "position_bitcoin": 0,
            "position_pnl": 0,
            "position_return_array": np.array([]),
            "position_value": self.init_balance
        } 
        
        return


        
    def reset(self, seed=None):
        """
        This function work like gym.reset()
        """
        if self.use_random_start == False:
            self.idx = 0
        elif self.use_random_start == True:
            random.seed(seed)
            self.idx = random.randint(0, self.chart_len)
        
        self.position_portfolio_value = [self.init_balance]
        self.position_bitcoin_value = [self.chart_data.iloc[self.idx, self.mid_price_index]]
        
        
        self.current_position = {
            "position_size": 0,
            "position_len": 0,
            "position_cash": self.init_balance,
            "position_bitcoin": 0,
            "position_pnl": 0,
            "position_return_array": np.array([]),
            "position_value": self.init_balance
        } 
        
        self.next_position = {
            "position_size": 0,
            "position_len": 0,
            "position_cash": self.init_balance,
            "position_bitcoin": 0,
            "position_pnl": 0,
            "position_return_array": np.array([]),
            "position_value": self.init_balance
        } 

        self.current_chart = self.chart_data.iloc[self.idx]
        self.next_chart = self.chart_data.iloc[self.idx+1]                    


        init_obserbation = self.chart_data.iloc[self.idx]
        
        return (np.array(init_obserbation), 
                
                {
                "state":self.current_chart,
                "next_state":self.next_chart,
                "current_position": self.current_position,
                "next_position": self.next_position,
                "position_value": self.position_portfolio_value,
                "position_value": self.position_bitcoin_value,
                })