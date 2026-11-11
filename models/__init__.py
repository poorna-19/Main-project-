from .transformer import GraphTransformer
from omegaconf import DictConfig
from .brainnetcnn import BrainNetCNN
from .fbnetgen import FBNETGEN
from .CDCo_MambaNet import CDCo_MambaNet
from .CDCo_MambaNet_V2 import CDCo_MambaNet_V2
from .CDCo_MambaNet_V3 import CDCo_MambaNet_V3
from .CDCo_MambaNet_V4 import CDCo_MambaNet_V4


def model_factory(config: DictConfig):
    if config.model.name in ["LogisticRegression", "SVC"]:
        return None
    return eval(config.model.name)(config).cuda()
