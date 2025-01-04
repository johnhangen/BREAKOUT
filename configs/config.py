from dataclasses import dataclass
import yaml

@dataclass
class DQNConfig:
    gamma: float = 0.99
    alpha: float = 0.0001
    epsilon: float = 1.0
    epsilon_min: float = 0.1
    epsilon_decay: float = 0.001
    batch_size: int = 500
    memory_size: int = 100_000
    C:int = 1000

@dataclass
class Config:
    DQN: DQNConfig

    @staticmethod
    def load_config(config_path: str) -> 'Config':
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        DQN_config = DQNConfig(**config_dict["DQN"])

        return Config(
            DQN=DQN_config
        )