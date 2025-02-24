import hydra  
from omegaconf import DictConfig, OmegaConf

def train(model: str, n_epoches: int, lr: float, batch_size: int):
    print("Start training...")
    ...

@hydra.main(version_base=None, config_path="../../recipes", config_name="hydra_default")  
def my_app(cfg : DictConfig) -> None: 
    print(OmegaConf.to_yaml(cfg)) 

    # train(model, n_epoches, lr, batch_size) # 用给定配置训练模型

if __name__ == "__main__":  
    my_app()