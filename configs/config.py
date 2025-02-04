from dataclasses import dataclass
import yaml

@dataclass
class DQNConfig:
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_min: float = 0.1
    epsilon_decay: float = 0.001
    batch_size: int = 500
    C:int = 10_000
    grad_clip_val: int = 1
    num_episodes:int = 10_000

@dataclass
class OptimizerConfig:
    alpha: float = 0.00025
    squared_gradient_momentum: float = 0.95
    min_squared_gradient: float = 0.01
    gradient_momentum: float = 0.95

@dataclass
class ENVConfig:
    seed: int = 42
    render_mode = None
    plot_rewards_bool: bool = False
    reward_clip: bool = True
    repeat: int = 4

@dataclass
class MemoryReplayConfig:
    max_memory: int = 10_000
    seed: int = 0
    priority_replay: bool = True
    variant: str = 'rank'
    alpha: float = 0.5
    alpha_decay: float = 0.0
    beta: float = 0.0
    beta_decay: float = 0.0

@dataclass
class Config:
    DQN: DQNConfig
    ENV: ENVConfig
    MemoryReplay: MemoryReplayConfig
    Optimizer: OptimizerConfig

    @staticmethod
    def load_config(config_path: str) -> 'Config':
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        DQN_config = DQNConfig(**config_dict["DQN"])
        ENV_config = ENVConfig(**config_dict["ENV"])
        MemoryReplay_config = MemoryReplayConfig(**config_dict["MemoryReplay"])
        Optimizer_Config = OptimizerConfig(**config_dict["Optimizer"])

        return Config(
            DQN=DQN_config,
            ENV=ENV_config,
            MemoryReplay=MemoryReplay_config,
            Optimizer=Optimizer_Config
        )