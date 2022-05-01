import os
import shutil
import yaml
import argparse
import note_seq
import torch
import torch.nn as nn
import time
from tensorboardX import SummaryWriter
from optim import Optimizer
from model.encoder import build_encoder
from model.decoder import build_decoder
from model.ListenAttendSpell import ListenAttendSpell
from utils import  AttrDict, init_logger
from dataset import MAESTRO, collate_fn, worker_init_fn
from torch.utils.data import DataLoader
from data import run_length_encoding, event_codec, vocabularies, spectrograms

def train(epoch, config, model, training_dataloader, optimizer, logger, visualizer=None):
    model.train() 
    start_epoch = time.process_time()
    optimizer.epoch()   
    total_loss = 0
    batch_steps = len(training_dataloader)

    for step, (audio_batch, label_batch, audio_lengths, labels_lengths) in enumerate(training_dataloader):
        
        if config.training.num_gpu > 0:
            audio_batch = audio_batch.cuda()
            audio_lengths = audio_lengths.cuda()
    
        start= time.process_time()
        optimizer.zero_grad()
        
        loss = model(audio_batch, audio_lengths, label_batch)
        if config.training.num_gpu > 1:
            loss = torch.mean(loss)
        
        loss.backward()
        total_loss += loss.item()

        grad_norm = nn.utils.clip_grad_norm_(
            model.parameters(),
            config.training.max_grad_norm
        )

        optimizer.step()
        if visualizer is not None:
            visualizer.add_scalar(
                'train_loss', loss.item(), optimizer.global_step)
            visualizer.add_scalar(
                'learn_rate', optimizer.lr, optimizer.global_step)

        avg_loss = total_loss / (step + 1)
        if optimizer.global_step % config.training.show_interval == 0:
            end = time.process_time()
            process = step / batch_steps * 100
            logger.info('-Training-Epoch:%d(%.5f%%), Global Step:%d, Learning Rate:%.6f, Grad Norm:%.5f, Loss:%.5f, '
                        'AverageLoss: %.5f, Run Time:%.3f' % (epoch, process, optimizer.global_step, optimizer.lr,
                                                              grad_norm, loss.item(), avg_loss, end-start))
    if visualizer is not None:
        visualizer.add_scalar('avg_train_loss', avg_loss, epoch)
    end_epoch = time.process_time()
    logger.info('-Training-Epoch:%d, Average Loss: %.5f, Epoch Time: %.3f' %
                (epoch, total_loss / (step+1), end_epoch-start_epoch))


def eval(epoch, config, model, validation_dataloader, logger, vocab, visualizer=None):
    model.eval()
    total_loss = 0
    batch_steps = len(validation_dataloader)
    
    for step, (audio_batch, label_batch, audio_lengths, labels_lengths) in enumerate(validation_dataloader):
        if config.training.num_gpu > 0:
            audio_batch = audio_batch.cuda()
            audio_lengths = audio_lengths.cuda()
            label_batch = label_batch.cuda()
        loss = model(audio_batch, audio_lengths, label_batch)
        total_loss += loss.item()
        avg_loss = total_loss / (step + 1)
        if step % config.training.show_interval == 0:
            process = step / batch_steps * 100
            logger.info('-Validation-Epoch:%d(%.5f%%), avg_val_loss: %.5f %%' % (epoch, process, avg_loss))
    
    check = model.recognize(audio_batch[0], audio_lengths[0].unsqueeze(0), config)
    pred_output = vocab._decode(check.cpu().numpy())
    check = label_batch[0][:labels_lengths[0]].cpu().numpy()
    ref_output = vocab._decode(check)
       
    logger.info('-Validation-Epoch:%4d, AverageLoss: %.5f %%' %
                    (epoch, avg_loss))
    logger.info("predicted output")
    logger.info(pred_output)
    logger.info("reference output")
    logger.info(ref_output)         
    
    if visualizer is not None:
        visualizer.add_scalar('avg_val_loss', avg_loss, epoch)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='config/model.yaml')
    parser.add_argument('-log', type=str, default='train.log')
    parser.add_argument('-mode', type=str, default='retrain')
    opt = parser.parse_args()

    configfile = open(opt.config)
    config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))

    exp_name = os.path.join('MAESTRO', config.data.name, 'exp', config.training.save_model)
    if not os.path.isdir(exp_name):
        os.makedirs(exp_name)
    logger = init_logger(os.path.join(exp_name, opt.log))

    shutil.copyfile(opt.config, os.path.join(exp_name, 'config.yaml'))
    logger.info('Save config info.')

    codec = event_codec.Codec(
                max_shift_steps=config.data.max_shift_steps,
                steps_per_second=config.data.steps_per_second,
                event_ranges=[
                    event_codec.EventRange('pitch', note_seq.MIN_MIDI_PITCH,
                                note_seq.MAX_MIDI_PITCH),
                    event_codec.EventRange('velocity', 0, 127)
                ])
    run_length_encode_shifts = run_length_encoding.run_length_encode_shifts_fn(
            codec=codec)
    vocab = vocabularies.vocabulary_from_codec(codec)
    spectrogram_config = spectrograms.SpectrogramConfig()

    train_dataset =MAESTRO(
        config = config.data,
        type = 'train',
        codec = codec,
        run_length_encode_shifts = run_length_encode_shifts,
        spectrogram_config = spectrogram_config,
        vocab = vocab)
    training_dataloader = DataLoader(
        train_dataset, 
        batch_size = config.data.batch_size * config.training.num_gpu,
        shuffle = True,
        collate_fn = collate_fn,
        worker_init_fn = worker_init_fn,
        num_workers=2 )
    logger.info('Load Train Set!')
    
    validation_dataset =MAESTRO(
        config = config.data,
        type = 'validation',
        codec = codec,
        run_length_encode_shifts = run_length_encode_shifts,
        spectrogram_config = spectrogram_config,
        vocab = vocab)
    validation_dataloader = DataLoader(
        validation_dataset, 
        batch_size = config.data.batch_size * config.training.num_gpu,
        shuffle=False,
        collate_fn = collate_fn,
        worker_init_fn = worker_init_fn,
        num_workers=2)
    logger.info('Load Dev Set!')

    if config.training.num_gpu > 0:
        torch.cuda.manual_seed(config.training.seed)
        torch.backends.cudnn.deterministic = True
    else:
        torch.manual_seed(config.training.seed)
    logger.info('Set random seed: %d' % config.training.seed)
    
    config.model.decoder.vocab_size = validation_dataset.vocab_size
    logger.info('Total vocab size: %d' % validation_dataset.vocab_size)

    encoder = build_encoder(config.model.encoder)
    decoder = build_decoder(config.model.decoder)
    model = ListenAttendSpell(encoder, decoder)
    start_epoch = 0
    
    if config.training.num_gpu > 0:
        model = model.cuda()
        if config.training.num_gpu > 1:
            device_ids = list(range(config.training.num_gpu))
            model = torch.nn.DataParallel(model, device_ids=device_ids)
        logger.info('Loaded the model to %d GPUs' % config.training.num_gpu)
    
    optimizer = Optimizer(model.parameters(), config.optim)
    logger.info('Created a %s optimizer.' % config.optim.type)

    if config.training.visualization:
        visualizer = SummaryWriter(os.path.join(exp_name, 'log'))
        logger.info('Created a visualizer.')
    else:
        visualizer = None
   
    for epoch in range(start_epoch, config.training.epochs):
        train(epoch, config, model, training_dataloader,
              optimizer, logger, visualizer)
    
    if (epoch+1)% config.training.eval_interval == 0:
        eval(epoch, config, model, validation_dataloader, logger, vocab, visualizer)
        
    if (epoch+1) % config.training.save_interval == 0:
        save_name = os.path.join(exp_name, '%s.epoch%d.chkpt' % (config.training.save_model, epoch))
        model_states = model.save_las_model(model, epoch)
        torch.save(model_states, save_name)
        logger.info('Epoch %d model has been saved.' % epoch)
    else:
        logger.info('skipp saving')