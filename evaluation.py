
import torch
from data import run_length_encoding
from data import event_codec
from data import vocabularies
from data import spectrograms
from data import note_sequences
import note_seq
import numpy as np
import pretty_midi
import sklearn
from typing import Callable, Mapping, Optional, Sequence, Tuple
import editdistance
import mir_eval
import librosa
from utils import _audio_to_frames, AttrDict
#from model.ListenAttendSpell import ListenAttendSpell 
from tqdm import tqdm
import yaml
import functools
import multiprocessing
import os
import pandas as pd

def pad_sequence(batch, audio):
    if audio:
        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)
    else:
        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=-1)
    return batch

codec = event_codec.Codec(
        max_shift_steps=300,
        steps_per_second=100,
        event_ranges=[
                event_codec.EventRange('pitch', note_seq.MIN_MIDI_PITCH,
                            note_seq.MAX_MIDI_PITCH),
                event_codec.EventRange('velocity', 0, 127)
        ])

def events_to_ns(events):
    decoding_state = note_sequences.NoteDecodingState()
    invalid_ids, dropped_events = run_length_encoding.decode_events(
        state=decoding_state, tokens=events, start_time=0, max_time=None,
        codec=codec, decode_event_fn=note_sequences.decode_note_event)
    ns = note_sequences.flush_note_decoding_state(decoding_state)
    return ns 

def get_piano_roll(ns, fps=62.5):
    pm = note_seq.note_sequence_to_pretty_midi(ns)
    end_time = pm.get_end_time()
    cc = [
        # all sound off
        pretty_midi.ControlChange(number=120, value=0, time=end_time),
        # all notes off
        pretty_midi.ControlChange(number=123, value=0, time=end_time)
    ]
    pm.instruments[0].control_changes = cc
    for inst in pm.instruments:
        inst.is_drum = False
    piano_roll = pm.get_piano_roll(fs=fps)
    return piano_roll

def frame_metrics(ref_pianoroll: np.ndarray,
                  est_pianoroll: np.ndarray,
                  velocity_threshold: int) -> Tuple[float, float, float]:
    """Frame Precision, Recall, and F1."""
    # Pad to same length
    if ref_pianoroll.shape[1] > est_pianoroll.shape[1]:
        diff = ref_pianoroll.shape[1] - est_pianoroll.shape[1]
        est_pianoroll = np.pad(est_pianoroll, [(0, 0), (0, diff)], mode='constant')
    elif est_pianoroll.shape[1] > ref_pianoroll.shape[1]:
        diff = est_pianoroll.shape[1] - ref_pianoroll.shape[1]
        ref_pianoroll = np.pad(ref_pianoroll, [(0, 0), (0, diff)], mode='constant')

  # For ref, remove any notes that are too quiet (consistent with Cerberus.)
    ref_frames_bool = ref_pianoroll > velocity_threshold
    # For est, keep all predicted notes.
    est_frames_bool = est_pianoroll > 0
    precision, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(
        ref_frames_bool.flatten(),
        est_frames_bool.flatten(),
        labels=[True, False])
    return precision[0], recall[0], f1[0]

def computer_cer(pred, label):
    dist = editdistance.eval(label, pred)
    total = len(label)
    return dist, total

def segmentize_audio(audio_data, spectrogram_config):
    def_sr = 16000
    def_seg_frame_len = 200
    audio, sr = librosa.load(audio_data, sr = def_sr)
    frames, frame_times = _audio_to_frames(audio, spectrogram_config)
    segmented_frames = [frames[i:i+def_seg_frame_len, :] 
        for i in range(0, len(frame_times),def_seg_frame_len)]
    segmented_times = [frame_times[i:i+def_seg_frame_len+1]
        for i in range(0,len(frame_times),def_seg_frame_len)]
    return segmented_frames, segmented_times

def get_test_files(csv_path):
    df = pd.read_csv(csv_path)
    df_test = df.loc[df['split'] == 'test']
    test_files = df_test['audio_filename'].to_list()
    final_files = []
    midi_files = []
    base_path = os.path.split(csv_path)[0]

    for i in test_files:
        file_name = i.replace('.wav', '_test.pt')
        midi_name = i.replace('.wav', '.midi')
        final_file = os.path.join(base_path, file_name)
        final_files.append(final_file)
        final_file = os.path.join(base_path, midi_name)
        midi_files.append(final_file)
    return final_files, midi_files

def make_spectrogram(segmented_frame, spectrogram_config):
    input_samples = spectrograms.flatten_frames(segmented_frame)
    input_samples = spectrograms.compute_spectrogram(input_samples, spectrogram_config).numpy()
    return input_samples

def concate_to_batch():
    return NotImplemented

#make_pred_ns(data['segmented_spectrograms'], data['segmented_times'], codec, vocab, model, config, is_gpu=True)
def make_pred_ns(audio_data, times, codec, vocab, model, config, is_gpu=False):
    
    encoding_spec = note_sequences.NoteEncodingSpec
    init_state_fn=encoding_spec.init_decoding_state_fn

    state = init_state_fn()
    begin_segment_fn = encoding_spec.begin_decoding_segment_fn
    decode_tokens_fn=functools.partial(
            run_length_encoding.decode_events,
            codec=codec,
            decode_event_fn=encoding_spec.decode_event_fn)
        

    for input_samples, time in tqdm(zip(audio_data, times), total=len(audio_data)):
        start_time = time[0]
        end_time = time[-1]
        
        input = torch.from_numpy(input_samples).float()
        input_length = torch.tensor([input.size(0)], dtype=torch.int)
        if is_gpu:
            input = input.cuda()
            #input_length = input_length.cuda()
        output = model.recognize(input, input_length, config.training)
        #print(output)
        events = vocab._decode(output)
        ignore_tokens = [-2,-1]
        events = [i for i in events if i not in ignore_tokens]
        
        begin_segment_fn(state)
        invalid_events, dropped_events = decode_tokens_fn(
            state, events, start_time, end_time)
    ns = encoding_spec.flush_decoding_state_fn(state)
    return ns

def make_batch_pred_ns(data, codec, vocab, model, is_gpu=False):
    batch_size = 32
    audio_batch = [torch.from_numpy(i).float() for i in data['segmented_spectrograms']]
    audio_batch = pad_sequence(audio_batch, -1)
    total_length = audio_batch.size(0)
    audio_length = [len(i) for i in data['segmented_spectrograms']]
    audio_length = torch.tensor(audio_length, dtype=torch.int)
    audio_batchs = [audio_batch[i:i+batch_size, : ,: ] for i in range(0, total_length, batch_size)]
    audio_lengths = [audio_length[i:i+batch_size] for i in range(0, total_length, batch_size)]
    outputs = []
    for audio_batch, audio_length in zip(audio_batchs, audio_lengths):
        if is_gpu:
            audio_batch = audio_batch.cuda()
        output = model.batch_greedy_recognize(audio_batch,audio_length)
        output = output.cpu().detach().tolist()
        outputs += output
    encoding_spec = note_sequences.NoteEncodingSpec
    init_state_fn=encoding_spec.init_decoding_state_fn

    state = init_state_fn()
    begin_segment_fn = encoding_spec.begin_decoding_segment_fn
    decode_tokens_fn=functools.partial(
            run_length_encoding.decode_events,
            codec=codec,
            decode_event_fn=encoding_spec.decode_event_fn)

    for i, time in zip(outputs, data['segmented_times']):
        start_time = time[0]
        end_time = time[-1]
        
        events = vocab._decode(i)
        ignore_tokens = [-2,-1]
        events = [i for i in events if i not in ignore_tokens]
        begin_segment_fn(state)
        invalid_events, dropped_events = decode_tokens_fn(
            state, events, start_time, end_time)

    ns = encoding_spec.flush_decoding_state_fn(state)
    return ns
    
def evaluation_mir_eval(ref_ns, est_ns):
    track_scores = {}
    est_intervals, est_pitches, est_velocities = (
            note_seq.sequences_lib.sequence_to_valued_intervals(est_ns))

    ref_intervals, ref_pitches, ref_velocities = (
            note_seq.sequences_lib.sequence_to_valued_intervals(ref_ns))
    
    precision, recall, f_measure, avg_overlap_ratio = (
        mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals=ref_intervals,
            ref_pitches=ref_pitches,
            est_intervals=est_intervals,
            est_pitches=est_pitches,
            offset_ratio=None))
    del avg_overlap_ratio
    track_scores['Onset precision'] = precision
    track_scores['Onset recall'] = recall
    track_scores['Onset F1'] = f_measure
    
    precision, recall, f_measure, avg_overlap_ratio = (
        mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals=ref_intervals,
            ref_pitches=ref_pitches,
            est_intervals=est_intervals,
            est_pitches=est_pitches))
    del avg_overlap_ratio
    track_scores['Onset + offset precision'] = precision
    track_scores['Onset + offset recall'] = recall
    track_scores['Onset + offset F1'] = f_measure

    precision, recall, f_measure, avg_overlap_ratio = (
        mir_eval.transcription_velocity.precision_recall_f1_overlap(
            ref_intervals=ref_intervals,
            ref_pitches=ref_pitches,
            ref_velocities=ref_velocities,
            est_intervals=est_intervals,
            est_pitches=est_pitches,
            est_velocities=est_velocities,
            offset_ratio=None))
    track_scores['Onset + velocity precision'] = precision
    track_scores['Onset + velocity recall'] = recall
    track_scores['Onset + velocity F1'] = f_measure

     
    # Precision / recall / F1 using onsets, offsets, and velocities.
    precision, recall, f_measure, avg_overlap_ratio = (
        mir_eval.transcription_velocity.precision_recall_f1_overlap(
            ref_intervals=ref_intervals,
            ref_pitches=ref_pitches,
            ref_velocities=ref_velocities,
            est_intervals=est_intervals,
            est_pitches=est_pitches,
            est_velocities=est_velocities))
    track_scores['Onset + offset + velocity precision'] = precision
    track_scores['Onset + offset + velocity recall'] = recall
    track_scores['Onset + offset + velocity F1'] = f_measure
    return track_scores

def evaluation_test_set(file_lists):
    return NotImplemented

if __name__ == "__main__":
    '''
    load all the necessary module and configuration
    load model 
    '''
    with open('config/model_las.yaml', 'r') as f:
        file_config = yaml.safe_load(f)
    config = AttrDict(file_config)
    config.training.beam_size = 2
    codec = event_codec.Codec(
            max_shift_steps=300,
            steps_per_second=100,
            event_ranges=[
                    event_codec.EventRange('pitch', note_seq.MIN_MIDI_PITCH,
                                note_seq.MAX_MIDI_PITCH),
                    event_codec.EventRange('velocity', 0, 127)
            ])
    vocab = vocabularies.vocabulary_from_codec(codec)
    spectrogram_config = spectrograms.SpectrogramConfig()
    
    model = Seq2Seq.load_model("/Users/donghyunlee/thesis/rnn_lstm/test_data/las_model.epoch849.chkpt")
    model.eval()
    audio_path = "/Users/donghyunlee/thesis/rnn_lstm/test_data/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.flac"
    pred_ns = make_pred_ns(audio_path,spectrogram_config, codec, vocab, model, config)
    
    midi_path = '/Users/donghyunlee/thesis/rnn_lstm/test_data/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.midi'
    ref_ns = note_seq.midi_file_to_note_sequence(midi_path)
    note_sequences.validate_note_sequence(ref_ns)
    ref_ns = note_seq.apply_sustain_control_changes(ref_ns)
    output = evaluation_mir_eval(ref_ns, pred_ns)
    print(output)