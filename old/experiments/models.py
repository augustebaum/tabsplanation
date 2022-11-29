from typing import List

from torch import nn

from tabsplanation.autoencoder.architectures import AE, VAE, VAENormalized, VAESigmoid
from tabsplanation.classifier import Classifier
from tabsplanation.normalizing_flow import NICEModel

AE_MODELS = {
    "AE": AE,
    "VAE": VAE,
    "VAESigmoid": VAESigmoid,
    "VAENormalized": VAENormalized,
    "NICE": NICEModel,
}
CLF_MODELS = {"Classifier": Classifier}


def get_models(models_dict, model_names, device, **kwargs) -> List[nn.Module]:
    return [models_dict[name].new(**kwargs).to(device) for name in model_names]
