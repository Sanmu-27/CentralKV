from dataclasses import dataclass

@dataclass
class CentralKVConfig:
    budget_ratio: float = 0.2
    k: int = 8
    lambda_val: float = 0.5
    local_window: int = 32
    sys_window: int = 8
    
    # Advanced graph params
    graph_update_freq: int = 1
    damping_factor: float = 0.85





