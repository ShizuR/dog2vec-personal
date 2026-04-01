import os
import random

#import fairseq
import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
import torch

class DummyTask:
    class Cfg:
        normalize = True  # matches fairseq default behavior
    cfg = Cfg()

class FeatureExtractor:
    def __init__(self, model_path, device="cuda", max_chunk=1600000, layer=9):
        set_seed(0)
        self.layer = layer
        self.device = torch.device(device)
        self.max_chunk = max_chunk

        checkpoint = torch.load(
            model_path,
            map_location=self.device,
            weights_only=False  # critical fix
        )

        # Try common checkpoint structures
        if "model" in checkpoint:
            model = checkpoint["model"]
        elif "models" in checkpoint:
            model = checkpoint["models"][0]
        else:
            model = checkpoint

        # If model is a state_dict, this will fail → handled below
        if hasattr(model, "extract_features"):
            self.encoder = model.eval().to(self.device)
        else:
            raise RuntimeError(
                "Checkpoint does not contain a loaded model object. "
                "It likely requires fairseq architecture."
            )

        self.task = DummyTask()

if __name__ == '__main__':
    model_path = ''
    extractor = FeatureExtractor(model_path)

    audio = torch.rand((1, 160000))
    feat = extractor.extract(audio)
    print(feat.shape)
    print(feat)
