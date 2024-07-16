import os
import gc #limpar memória
import sys
import yaml
import time
import glob
import h5py
#import faiss # environment tfpy, porque é onde está o faiss. Faiss não é compatível com tensorflow e python versions, tem de estar organizado assim
import click
import curses
import librosa
import wave

import numpy as np # tem de ser 1.23.5, por causa do deepdish
#import pandas as pd
import deepdish as dd
import tensorflow as tf


from model_RA.fp_RA.melspec.melspectrogram_RA import get_melspec_layer
from model_RA.fp_RA.nnfp import get_fingerprinter
from model_RA.utils.dataloader_keras import genUnbalSequence



###########---2. Leitura do modelo Neural---###########
def load_config(config_fname):
    config_filepath = './config/' + config_fname + '.yaml'
    if os.path.exists(config_filepath):
        print(f'cli: Configuration from {config_filepath}')
    else:
        sys.exit(f'cli: ERROR! Configuration file {config_filepath} is missing!!')

    with open(config_filepath, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def build_fp(cfg):
    """ Build fingerprinter """
    # m_pre: log-power-Mel-spectrogram layer, S.
    m_pre = get_melspec_layer(cfg, trainable=False)

    # m_fp: fingerprinter g(f(.)).
    m_fp = get_fingerprinter(cfg, trainable=False)
    return m_pre, m_fp


@tf.function
def predict(X, m_pre, m_fp):
    """ 
    Test step used for mini-search-validation 
    X -> (B,1,8000)
    """
    #tf.print(X)
    feat = m_pre(X)  # (nA+nP, F, T, 1)
    m_fp.trainable = False
    emb_f = m_fp.front_conv(feat)  # (BSZ, Dim)
    emb_gf = m_fp.div_enc(emb_f)
    emb_gf = tf.math.l2_normalize(emb_gf, axis=1)
    
    return emb_gf # L2(g(f(.))


### Carrega áudio
def get_audio(audiofile, sr_target=8000):
    audio, fs = librosa.load(audiofile, mono=True, sr=sr_target)
    return audio, fs


def nframe(audio, win_size, hop_size):
    frames =librosa.util.frame(x=audio, frame_length=win_size, hop_length=hop_size)
    return frames


def h5filesgenerator(cfg, files, root_out, m_pre, m_fp, win_size_sec, hop_size_sec):
    enq = tf.keras.utils.OrderedEnqueuer(files, use_multiprocessing=True, shuffle=False)
    enq.start(workers=cfg['DEVICE']['CPU_N_WORKERS'], max_queue_size=cfg['DEVICE']['CPU_MAX_QUEUE'])
    try:
        iteration_count = 0  # Contador para limpeza de memória periódica
        memory_cleaning_interval = 10  # Intervalo para limpeza de memória
        i = 0
        while i < len(enq.sequence):
            if iteration_count % memory_cleaning_interval == 0:
                tf.keras.backend.clear_session()  # Limpar memória da GPU
                gc.collect()  # Limpar memória da CPU

            X = next(enq.get()) # X: Tuple(Xa, Xp)
            print(f"{i}-{X}")
            audio, fs = get_audio(audiofile=X)
            audio_frames = nframe(audio, int(win_size_sec * fs), int(hop_size_sec*fs))
            audio_frames = np.transpose(audio_frames[np.newaxis, ...], (2, 0,1))

            #print(audio_frames.shape)


            # b) gerar o embedded
            #with tf.device("/GPU:0"):
            emb = predict(audio_frames, m_pre, m_fp) # TensorShape([473, 128])
            
            emb = emb.numpy() # (473, 128)
            


            #Cria .h5 files
            parts = X.split("/")
            dir_name = parts[-2]
            file_name = parts[-1].split(".")[0]
            output_dir = os.path.join(root_out, dir_name)
            
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            file_out =  os.path.join(output_dir, file_name + '.h5')
            dd.io.save(file_out, emb)

            i += 1
            iteration_count+= 1
        #enq.stop()

    except Exception as e:
        print(f"Erro encontrado: {e}")
    finally:
        enq.stop()

    
@click.group()
def cli():
    pass


#Run it as:  "python generateH5.py main"
@cli.command()
def main():
    input_dir = '/mnt/dataset/public/Fingerprinting/neural-audio-fp-dataset/music/test-dummy-db-100k-full/fma_full/'
    root_out = '/mnt/dataset/public/Fingerprinting/Embeddings_BFTRI/dummyEmb/'
    win_size_sec = 1
    hop_size_sec = 0.5

    config = "default_RA"
    cfg = load_config(config)

    m_pre, m_fp = build_fp(cfg)

    checkpoint_root_dir:str = "./logs/CHECK_BFTRI_100/101/"
    checkpoint = tf.train.Checkpoint(m_fp)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_root_dir))

    files = glob.glob(os.path.join(input_dir, '**/*.wav') ,recursive = True)
    files = sorted(files)

    h5filesgenerator(cfg, files, root_out, m_pre, m_fp, win_size_sec, hop_size_sec)


if __name__ == '__main__':
    cli()