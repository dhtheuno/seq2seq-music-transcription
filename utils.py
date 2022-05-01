import torch
import logging
import numpy as np 
from typing import Any, Sequence, Tuple
from data import spectrograms

IGNORE_ID = -1

def pad_list(xs, pad_value):
    # From: espnet/src/nets/e2e_asr_th.py: pad_list()
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, * xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]
    return pad

def pad_sequence(batch, audio=True):
    if audio:
        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)
    else:
        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=-1)
    return batch

def audio_to_frames(
    samples: Sequence[float],
    spectrogram_config: spectrograms.SpectrogramConfig,
    ) -> Tuple[Sequence[Sequence[int]], np.ndarray]:
    """Convert audio samples to non-overlapping frames and frame times."""
    frame_size = spectrogram_config.hop_width
    samples = np.pad(samples,
                    [0, frame_size - len(samples) % frame_size],
                    mode='constant')
    frames = spectrograms.split_audio(samples, spectrogram_config)
    num_frames = len(samples) // frame_size
    times = np.arange(num_frames) / spectrogram_config.frames_per_second
    return frames, times

class AttrDict(dict):
    """
    Dictionary whose keys can be accessed as attributes.
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)

    def __getattr__(self, item):
        if item not in self:
            return None
        if type(self[item]) is dict:
            self[item] = AttrDict(self[item])
        return self[item]

    def __setattr__(self, item, value):
        self.__dict__[item] = value

def get_saved_folder_name(config):
    return '_'.join([config.data.name, config.training.save_model])


def count_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'predictor' in name:
            dec += param.nelement()
    return n_params, enc, dec


def init_parameters(model, type='xnormal'):
    for p in model.parameters():
        if p.dim() > 1:
            if type == 'xnoraml':
                torch.nn.init.xavier_normal_(p)
            elif type == 'uniform':
                torch.nn.init.uniform_(p, -0.1, 0.1)
        else:
            pass

def init_logger(log_file=None):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger
