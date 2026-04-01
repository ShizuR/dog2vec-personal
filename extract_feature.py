import os
import random
#import fairseq
import numpy as np
import torch
import torch.nn.functional as F
import sys
import types

# Create fake fairseq module hierarchy
fairseq = types.ModuleType("fairseq")
sys.modules["fairseq"] = fairseq

# Add submodules that might be referenced
fairseq.models = types.ModuleType("fairseq.models")
sys.modules["fairseq.models"] = fairseq.models

fairseq.modules = types.ModuleType("fairseq.modules")
sys.modules["fairseq.modules"] = fairseq.modules

fairseq.tasks = types.ModuleType("fairseq.tasks")
sys.modules["fairseq.tasks"] = fairseq.tasks

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

        # 🔥 FAKE FAIRSEQ BEFORE LOAD
        import sys, types
        fairseq = types.ModuleType("fairseq")
        sys.modules["fairseq"] = fairseq
        sys.modules["fairseq.models"] = types.ModuleType("fairseq.models")
        sys.modules["fairseq.modules"] = types.ModuleType("fairseq.modules")
        sys.modules["fairseq.tasks"] = types.ModuleType("fairseq.tasks")

        checkpoint = torch.load(
            model_path,
            map_location=self.device,
            weights_only=False
        )

        if "model" in checkpoint:
            model = checkpoint["model"]
        elif "models" in checkpoint:
            model = checkpoint["models"][0]
        else:
            model = checkpoint

        if hasattr(model, "extract_features"):
            self.encoder = model.eval().to(self.device)
        else:
            raise RuntimeError("Model still requires full fairseq implementation")

        self.task = DummyTask()

if __name__ == '__main__':
    model_path = ''
    extractor = FeatureExtractor(model_path)

    audio = torch.rand((1, 160000))
    feat = extractor.extract(audio)
    print(feat.shape)
    print(feat)
