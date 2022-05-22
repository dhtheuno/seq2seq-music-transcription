import os
import json
import torch
import librosa
import note_seq
import random
import numpy as np
from tqdm import tqdm
from glob import glob
from abc import abstractmethod
from torch.utils.data import Dataset

from data import note_sequences, spectrograms, run_length_encoding
from utils import audio_to_frames, pad_sequence

class MidiAudioDataset(Dataset):
    def __init__(
            self, 
            config,
            type,
            codec,
            run_length_encode_shifts,
            spectrogram_config,
            vocab
    ):
        
        self.sample_rate = config.sample_rate
        self.path = config.path
        
        if type == 'train':
            self.groups = ['train']
        else:
            self.groups = ['validation']
        
        self.codec = codec
        self.run_length_encode_shifts = run_length_encode_shifts
        self.spectrogram_config = spectrogram_config
        self.vocab = vocab
        self.vocab_size = vocab.vocab_size
        self.data = []
        
        print(f"Loding {len(self.groups)} group{'s' if len(self.groups) > 1 else ''} "
              f"of {self.__class__.__name__} at {self.path}")
        for group in self.groups:
            for input_files in tqdm(self.files(group), desc='Loading groups %s' %group):
                self.data.append(self.load(*input_files))

    def __getitem__(self, index):
        data = self.data[index]

        #sequence_length = np.random.randint(1,250)
        target_length = 0 
        while target_length == 0:
            sequence_length = random.randint(1,200)
            frames_length = data['frames'].shape[0]
            #start_seuqnece = np.random.randint(0,int(frames_length-sequence_length))
            start_seuqnece = random.randint(0,int(frames_length-sequence_length)-1)
            end_sequence = start_seuqnece+sequence_length
            
            start = data['event_start_indices'][start_seuqnece]
            end = data['event_end_indices'][end_sequence]
            
            target_events = {}
            target_events['targets']= data['events'][start:end]
            
            output = self.run_length_encode_shifts(target_events)
            target_length = len(output['targets'])

        input_samples = spectrograms.flatten_frames(data['frames'][start_seuqnece:end_sequence])
        audio = spectrograms.compute_spectrogram(input_samples, self.spectrogram_config).numpy()
        
        audio = torch.FloatTensor(audio)
        label = torch.LongTensor(self.vocab._encode(output['targets']))
        label_length = int(len(label))+1
        #label = torch.LongTensor((output['targets']))


        audio_length = int(audio.size(0))
        if audio_length <= 0:
            print("yesesesesesese")
            print(label)
            print(label_length)
            print(sequence_length)
            print(start_seuqnece)
            print(end_sequence)
            print(data['frames'][start_seuqnece:end_sequence])
        #label_length = int(target_length)

        return {
            'audio': audio,
            'label': label,
            'audio_length': audio_length,
            'label_length': label_length
        }

    def __len__(self):
        return len(self.data)

    @classmethod
    @abstractmethod
    def available_groups(cls):
        raise NotImplementedError
    
    @abstractmethod
    def files(self, group):
        raise NotImplementedError
        
    def load(self, audio_path, midi_path):
        saved_data_path = audio_path.replace('.flac', '.pt').replace('.wav', '.pt')
        if os.path.exists(saved_data_path):
            return torch.load(saved_data_path)
        

        samples, sr = librosa.load(audio_path, sr= self.sample_rate)
        assert sr == self.sample_rate

        ns = note_seq.midi_file_to_note_sequence(midi_path)
        note_sequences.validate_note_sequence(ns)

        frames, frame_times = audio_to_frames(samples, self.spectrogram_config)
        ns = note_seq.apply_sustain_control_changes(ns)
        times, values = (note_sequences.note_sequence_to_onsets_and_offsets(ns))
        include_ties = False
        
        del ns.control_changes[:]

        (events, event_start_indices, event_end_indices,
            _, _) = (
                run_length_encoding.encode_and_index_events(
                state=note_sequences.NoteEncodingState() if include_ties else None,
                event_times=times,
                event_values=values,
                encode_event_fn=note_sequences.note_event_data_to_events,
                codec=self.codec,
                frame_times=frame_times,
                encoding_state_to_events_fn=(
                    note_sequences.note_encoding_state_to_events
                    if include_ties else None)))
        
        data = dict(path=audio_path,
                    frames = frames,
                    events = events,
                    event_start_indices = event_start_indices,
                    event_end_indices = event_end_indices)
        torch.save(data, saved_data_path)
        return data

class MAESTRO(MidiAudioDataset):
    def __init__(self, 
                config,
                type,
                codec,
                run_length_encode_shifts,
                spectrogram_config,
                vocab):
        super().__init__(config,
                         type,
                         codec,
                         run_length_encode_shifts,
                         spectrogram_config,
                         vocab)

    @classmethod
    def available_groups(cls):
        return ['train', 'validation', 'test']

    def files(self, group):
        if group not in self.available_groups():
            flacs = sorted(glob(os.path.join(self.path, group, '*.flac')))
            if len(flacs) == 0:
                flacs = sorted(glob(os.path.join(self.path, group, '*.wav')))

            midis = sorted(glob(os.path.join(self.path, group, '*.')))
            files = list(zip(flacs, midis))
            if len(files) ==0:
                raise RuntimeError(f'Group {group} is empty')

        else:
            metadata = json.load(open(os.path.join(self.path, 'maestro-v1.0.0.json')))
            files = sorted([(os.path.join(self.path, row['audio_filename'].replace('.wav', '.flac')),
                             os.path.join(self.path, row['midi_filename'])) for row in metadata if row['split'] == group])

            files = [(audio if os.path.exists(audio) else audio.replace('.flac', '.wav'), midi) for audio, midi in files]

        result = []
        for audio_path, midi_path in files:
            result.append((audio_path, midi_path))
        
        if group == "train":
            return result
        else:
            return result
def worker_init_fn(worker_id):
    torch_seed = torch.initial_seed()
    random.seed(torch_seed + worker_id)
    if torch_seed >= 2**30:  # make sure torch_seed + workder_id < 2**32
        torch_seed = torch_seed % 2**30
    np.random.seed(torch_seed + worker_id)

def collate_fn(batch):
    audios = []
    labels = []
    audio_lengths  = []
    labels_lengths = []
    
    for item in batch:
        audios += [item['audio']]
        labels += [item['label']]
        audio_lengths.append(item['audio_length'])
        labels_lengths.append(item['label_length'])
  
    audio_batch = pad_sequence(audios, audio=True)
    label_batch = pad_sequence(labels, audio=False)
    audio_lengths = torch.Tensor(audio_lengths)
    labels_lengths = torch.LongTensor(labels_lengths)
    
    sorted_input_lengths, indices = torch.sort(audio_lengths, descending=True)
    
    audio_batch = audio_batch[indices]
    label_batch = label_batch[indices]
    labels_lengths = labels_lengths[indices]

    return audio_batch, label_batch, sorted_input_lengths, labels_lengths