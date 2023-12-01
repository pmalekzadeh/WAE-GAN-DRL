'''A trading environment'''
import numpy as np
import dataclasses
import gym
from gym import spaces
import acme.utils.loggers as log_utils

from agent.demonstrations import DemonstrationRecorder
from domain.asset.barrier_option import BarrierDIPOption
from domain.asset.portfolio import Portfolio
from domain.sde.base import set_seed

from config.config_loader import ConfigLoader



@dataclasses.dataclass
class StepResult:
    """Logging step metrics for analysis
    """
    episode: int = 0                # episode number
    #######################################################
    # 1. time t information before taking action
    #######################################################
    t_1: int = 0                      # time step when taking action
    # action from agent (e.g. total portfolio delta ratio for rl or total stock shares for baselines)
    agent_action_1: float = 0.        # action from agent
    option_shares_action_1: float = 0.        # option shares to hedge gamma
    stock_price_1: float = 0.         # stock price at time t
    atm_vol_1: float = 0.   # implied volatility at time t
    liab_port_value_1: float = 0.   # liability portfolio value at time t without new arrival
    liab_port_value_2: float = 0.   # liability portfolio value at time t including new arrival
    liab_port_delta_1: float = 0.     # liability delta at time t without new arrival
    liab_port_delta_2: float = 0.     # liability delta at time t including new arrival
    liab_port_gamma_1: float = 0.     # liability gamma at time t without new arrival
    liab_port_gamma_2: float = 0.     # liability gamma at time t including new arrival
    liab_port_vega_1: float = 0.      # liability vega at time t without new arrival
    liab_port_vega_2: float = 0.      # liability vega at time t including new arrival
    num_client_arrival: float = 0.    # number of new arrival client options
    hed_port_value_1: float = 0.    # hedging portfolio value at time t
    hed_port_delta_1: float = 0.      # hedging portfolio delta at time t
    hed_port_gamma_1: float = 0.      # hedging portfolio gamma at time t
    hed_port_vega_1: float = 0.       # hedging portfolio vega at time t
    #######################################################
    # 2. time t information after taking action
    #######################################################
    hed_port_value_2: float = 0.    # hedging portfolio value at time t after taking action
    hed_port_delta_2: float = 0.      # hedging portfolio delta at time t after taking action
    hed_port_gamma_2: float = 0.      # hedging portfolio gamma at time t after taking action
    hed_port_vega_2: float = 0.       # hedging portfolio vega at time t after taking action
    hed_cost_2: float = 0.            # hedging cost at time t after taking action
    stock_position_2: float = 0.      # stock shares at time t after taking action
    #######################################################
    # 3. transition period time t to t+1 information
    #######################################################
    stock_pnl_3: float = 0.           # stock pnl from t to t+1
    liab_port_pnl_3: float = 0.       # liability pnl from t to t+1
    hed_port_pnl_3: float = 0.        # hedging pnl from t to t+1
    step_pnl_3: float = 0.            # step pnl from t to t+1
    #######################################################
    # 4. time t+1 information (states) before taking action
    #######################################################
    liab_port_value_4: float = 0.   # liability portfolio value at time t+1 excluding new arrival
    state_stock_price_4: float = 0.         # stock price at time t+1
    state_atm_vol_4: float = 0.           # implied volatility at time t+1
    state_delta_4: float = 0.         # delta at time t+1 after taking action
    state_gamma_4: float = 0.         # gamma at time t+1 after taking action
    state_ttm_4: float = 0.            # time to maturity at time t+1 after taking action
    state_vega_4: float = 0.           # vega at time t+1 after taking action
    state_barrier_crossing_indicator_4: float = 0. # barrier crossing indicator at time t+1 after taking action

def make_logger(work_folder, label, terminal=False):
    loggers = [
        log_utils.CSVLogger(f'./logs/{work_folder}',
                            label=label, add_uid=False)
    ]
    if terminal:
        loggers.append(log_utils.TerminalLogger(label=label))

    logger = log_utils.Dispatcher(loggers, log_utils.to_numpy)
    logger = log_utils.NoneFilter(logger)
    return logger


@ConfigLoader.register_class
class DREnv(gym.Env):
    """
    Trading Environment;
    """

    # trade_freq in unit of day, e.g 2: every 2 day; 0.5 twice a day;
    def __init__(self, 
                 portfolio: Portfolio,
                 episode_length: int,
                 action_low: float,
                 action_high: float,
                 seed: int = 1234,
                 vega_state: bool = False,
                 record_demo: bool = False,
                 scale_action: bool = False,
                 logger_folder: str = None,
                 logger_label: str = 'train_loop',):
        
        super(DREnv, self).__init__()
        self.logger = make_logger(logger_folder, logger_label) if logger_folder else None
        self.recorder = DemonstrationRecorder() if record_demo else None
        self.scale_action = scale_action
        # simulated data: array of asset price, option price and delta paths (num_path x num_period)
        # generate data now
        self.portfolio: Portfolio = portfolio
        self.episode_length = episode_length
        self.n_step = 0
        self.n_episode = -1
        
        # Action space: HIGH value has to be adjusted with respect to the option used for hedging
        self.action_space = spaces.Box(low=np.array([action_low]),
                                       high=np.array([action_high]), dtype=np.float32)

        # Observation space
        obs_lowbound = np.array([
            0.,   # stock price
            0.,   # implied vol 
            -np.inf,  # portfolio delta
            -np.inf,  # portfolio gamma
            0         # episode time to terminal step
        ])
        obs_highbound = np.array([
            np.inf,
            10, 
            np.inf,
            np.inf,
            self.episode_length
        ])
        self.vega_state = vega_state
        if vega_state:
            obs_lowbound = np.concatenate([obs_lowbound, [-999999]])
            obs_highbound = np.concatenate([obs_highbound, [999999]])
        self.observation_space = spaces.Box(low=obs_lowbound, high=obs_highbound, dtype=np.float32)
        # observations for record
        self.prev_obs = None
        self.prev_reward = None
        self.seed(seed)

    def seed(self, seed):
        set_seed(seed)
        np.random.seed(seed)

    def reset(self):
        """
        reset function which is used for each episode (spread is not considered at this moment)
        """
        # repeatedly go through available simulated paths (if needed)
        self.portfolio.sde.reset()  # randomize sde parameters for domain randomization
        self.n_step = 0
        self.n_episode += 1
        result = StepResult(
            episode=self.n_episode,
            t_1=self.n_step,
            stock_price_1=self.portfolio.sde.stock_price(),
            atm_vol_1=self.portfolio.sde.implied_vol(np.array([self.episode_length-self.n_step]), 
                                                     np.array([1.]))[0],
            liab_port_gamma_1=self.portfolio.client_options.get_gamma(),
            liab_port_delta_1=self.portfolio.client_options.get_delta(),
            liab_port_vega_1=self.portfolio.client_options.get_vega(),
            hed_port_gamma_1=self.portfolio.hedging_options.get_gamma(),
            hed_port_delta_1=self.portfolio.hedging_options.get_delta(),
            hed_port_vega_1=self.portfolio.hedging_options.get_vega(),
        )
        self.prev_obs = self.get_state(result)
        return self.prev_obs

    def get_state(self, result):
        states = np.array([self.portfolio.stock.sde.stock_price(), 
                           self.portfolio.sde.implied_vol(np.array(self.portfolio.hedging_options.sim_ttms), 
                                                          np.array([1.]))[0],
                           self.portfolio.get_delta(),
                           self.portfolio.get_gamma(),
                           self.episode_length-self.n_step])
        result.state_stock_price_4 = states[0]
        result.state_atm_vol_4 = states[1]
        result.state_delta_4 = states[2]
        result.state_gamma_4 = states[3]
        result.state_ttm_4 = states[4]
        if self.vega_state:
            states = np.concatenate([states, [self.portfolio.get_vega()]])
            result.state_vega_4 = states[5]
        if isinstance(self.portfolio.client_options, BarrierDIPOption):
            # add barrier crossing indicator to state
            states = np.concatenate([states, [self.portfolio.client_options.get_barrier_crossing_indicator()]])
            result.state_barrier_crossing_indicator_4 = states[6]
        return states


    def terminal_state(self):
        return self.n_step == self.episode_length
    

    def transform_action(self, action):
        # rescale action space to hedging share
        # action 0 means full hedging
        # action -1 means over hedging 100%
        # action 1 means no hedge
        # action 2 means increase delta 100%
        # hedging_ratio = -action[0] + 1
        # delta_action_bound = (-1 * self.portfolio.get_delta(self.t))
        # buy_sell_hed_share = hedging_ratio * delta_action_bound
        # return self.portfolio.underlying.get_delta(
        #     self.t) + buy_sell_hed_share
        hedging_option = self.portfolio.hedging_options.generate_atm_option()
        gamma_action_bound = -self.portfolio.get_gamma()/hedging_option.get_gamma()
        action_low = [0, gamma_action_bound]
        action_high = [0, gamma_action_bound]
        if self.vega_state:
            vega_action_bound = -self.portfolio.get_vega()/hedging_option.get_vega()
            action_low.append(vega_action_bound)
            action_high.append(vega_action_bound)
        low_val = np.min(action_low)
        high_val = np.max(action_high)
        buy_sell_hed_share = (low_val + action[0] * (high_val - low_val))
        return buy_sell_hed_share

    def inverse_transform_action(self, action):
        # scale hedging shares action to hedging ratio
        # delta_action_bound = (-1 * self.portfolio.get_delta(self.t))
        # bull_sell_hed_share = action[0] - \
        #     self.portfolio.underlying.get_delta(self.t)
        # hedging_ratio = bull_sell_hed_share / delta_action_bound
        # action_value = np.array([1 - hedging_ratio])
        # # clip to FLAGS.action_space
        # action_value = np.clip(
        #     action_value, self.action_space.low[0], self.action_space.high[0])
        # return action_value
        hedging_option = self.portfolio.hedging_options.generate_atm_option()
        gamma_action_bound = -self.portfolio.get_gamma()/hedging_option.get_gamma()
        action_low = [0, gamma_action_bound]
        action_high = [0, gamma_action_bound]
        if self.vega_state:
            vega_action_bound = -self.portfolio.get_vega()/hedging_option.get_vega()
            action_low.append(vega_action_bound)
            action_high.append(vega_action_bound)
        low_val = np.min(action_low)
        high_val = np.max(action_high)
        buy_sell_hed_share = action[0]
        hedging_ratio = (buy_sell_hed_share - low_val) / (high_val - low_val)
        return hedging_ratio

    def _step(self, action, scale_action=True):
        """
        profit loss period reward
        """
        result = StepResult(
            episode=self.n_episode,
            t_1=self.n_step,
            stock_price_1=self.portfolio.stock.sde.stock_price(),
            atm_vol_1=self.portfolio.sde.implied_vol(np.array([self.episode_length-self.n_step]), 
                                                     np.array([1.]))[0],
            hed_port_value_1=self.portfolio.hedging_options.get_value(),
            liab_port_delta_1=self.portfolio.client_options.get_delta(),
            hed_port_delta_1=self.portfolio.hedging_options.get_delta(),
            liab_port_gamma_1=self.portfolio.client_options.get_gamma(),
            hed_port_gamma_1=self.portfolio.hedging_options.get_gamma(),
            liab_port_vega_1=self.portfolio.client_options.get_vega(),
            hed_port_vega_1=self.portfolio.hedging_options.get_vega(),
            liab_port_value_1=self.portfolio.client_options.get_value()
        )

        # process action to get hedging share
        if self.scale_action and scale_action:
            result.agent_action_1 = action[0]
            result.option_shares_action_1 = self.transform_action(action)
        else:
            # baseline models use hedging share directly
            result.agent_action_1 = self.inverse_transform_action(action)
            result.option_shares_action_1 = action[0]
        if self.recorder is not None:
            # record transition
            self.recorder.step(
                self.prev_obs, result.agent_action_1, self.prev_reward)
        # Event 1. New arrival client options
        if self.n_step != 0:
            result.num_client_arrival = self.portfolio.simulate_client_trade()
        result.liab_port_delta_2 = self.portfolio.client_options.get_delta()
        result.liab_port_gamma_2 = self.portfolio.client_options.get_gamma()
        result.liab_port_vega_2 = self.portfolio.client_options.get_vega()
        result.liab_port_value_2 = self.portfolio.client_options.get_value()
        # Event 2 & 3. Hedging, exercise expired options 
        result.hed_cost_2 = self.portfolio.trade_hedging(result.option_shares_action_1)
        result.hed_port_value_2 = self.portfolio.hedging_options.get_value()
        result.hed_port_delta_2 = self.portfolio.hedging_options.get_delta()
        result.hed_port_gamma_2 = self.portfolio.hedging_options.get_gamma()
        result.hed_port_vega_2 = self.portfolio.hedging_options.get_vega()
        result.stock_position_2 = self.portfolio.stock.shares
        # Event 4. SDE step
        self.portfolio.sde.step()

        result.step_pnl_3 = reward = self.portfolio.get_pnl()
        result.stock_pnl_3 = self.portfolio.stock_pnl
        result.liab_port_pnl_3 = self.portfolio.client_options_pnl
        result.hed_port_pnl_3 = self.portfolio.hedging_options_pnl
        
        # update time step
        self.n_step = self.n_step + 1
        state = self.get_state(result)
        result.liab_port_value_4 = self.portfolio.client_options.get_value()
        self.prev_obs = state
        self.prev_reward = reward
        self.portfolio.clear()

        # for other info later
        info = {"path_row": self.n_episode}
        if self.logger:
            result_dict = dataclasses.asdict(result)
            if not isinstance(self.portfolio.client_options, BarrierDIPOption):
                result_dict.pop("state_barrier_crossing_indicator_4")
            self.logger.write(result_dict)
        return state, reward, self.terminal_state(), info

    def step(self, action):
        """
        profit loss period reward
        """
        state, reward, done, info = self._step(action, self.scale_action)
        
        if done and self.recorder is not None:
            self.recorder.record_episode()
        return state, reward, done, info
