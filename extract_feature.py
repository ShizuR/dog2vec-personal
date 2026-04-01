import os
import random

import fairseq

import numpy as np
import torch
import torch.nn.functional as F

import fairseq.data.dictionary
import argparse
import soundfile
import torchaudio

torch.serialization.add_safe_globals([
    fairseq.data.dictionary.Dictionary
])

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FeatureExtractor:
    def __init__(self, model_path, device="cuda", max_chunk=1600000, layer=9):
        set_seed(0)
        self.layer = layer
        self.device = torch.device(device)
        self.max_chunk = max_chunk

        (
            model,
            cfg,
            task,
        ) = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_path])
        self.encoder = model[0].eval().to(self.device)
        self.task = task

    def extract(self, audio):
        x = audio.to(self.device)
        with torch.no_grad():
            if self.task.cfg.normalize:
                x = F.layer_norm(x, x.shape)
            x = x.view(1, -1)

            feat = []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start: start + self.max_chunk]
                feat_chunk, _ = self.encoder.extract_features(
                    source=x_chunk,
                    padding_mask=None,
                    mask=False,
                    output_layer=self.layer,
                )
                feat.append(feat_chunk)
        feat = torch.cat(feat, 1).squeeze(0)
        return feat


if __name__ == '__main__':
    model_path = '/kaggle/input/models/shizurai/dog2vec/pytorch/default/1/dog2vec_130k_9.pt'
    extractor = FeatureExtractor(model_path)
    parser = argparse.ArgumentParser()
    parser.add_argument('--audioPath', type=str, required=True)
    args = parser.parse_args()

    waveform, sample_rate = soundfile.read(args.audioPath)

    audio = torch.tensor(waveform, dtype=torch.float32)

    # Convert to mono safely
    #if len(audio.shape) == 2:
    #    audio = audio.mean(dim=1)

    # Resample to 16kHz
    #if sample_rate != 16000:
    #    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
    #    audio = resampler(audio.unsqueeze(0)).squeeze(0)

    # Normalize safely
    #if audio.abs().max() > 0:
    #    audio = audio / audio.abs().max()

    feat = extractor.extract(audio)
    print(feat.shape)
    print(feat)
