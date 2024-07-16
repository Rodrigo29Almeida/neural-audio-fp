import os
import sys
import yaml
import glob
import h5py
import click
import librosa

import numpy as np
import soundfile as sf
import deepdish as dd
import tensorflow as tf

from model_RA.fp_RA.melspec.melspectrogram_RA import get_melspec_layer
from model_RA.fp_RA.nnfp import get_fingerprinter
from model_RA.utils.dataloader_keras import genUnbalSequence


@click.group()
def cli():
    pass


def load_config(config_fname):
    config_filepath = './config/' + config_fname + '.yaml'
    if os.path.exists(config_filepath):
        print(f'cli: Configuration from {config_filepath}')
    else:
        sys.exit(f'cli: ERROR! Configuration file {config_filepath} is missing!!')

    with open(config_filepath, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def load_audio(file_path):
    audio, _ = librosa.load(file_path, sr=8000)
    return audio


def split_audio_into_segments(audio, dur=1.0, hop=.5, fs=8000):
    segment_samples = int(dur * fs) # 8000
    hop_samples = int(hop * fs) # 4000

    segments = []
    for start in range(0, len(audio) - segment_samples + 1, hop_samples):
        #len(audio) = 1898580; segment_samples = 8000; hop_samples = 4000
        #0, 1898580 - 8000 + 1, 4000

        #fica a faltar as residual frames, que são as ultimas que são menores que 1 segundo

        segment = audio[start:start + segment_samples] #faz os vetores de 8000 em 8000, isto é, de 1s em 1s, com 

        if len(segment) == segment_samples:
            segments.append(segment)

    return segments


def build_fp(cfg):
    """ Build fingerprinter """
    # m_pre: log-power-Mel-spectrogram layer, S.
    m_pre = get_melspec_layer(cfg, trainable=False)

    # m_fp: fingerprinter g(f(.)).
    m_fp = get_fingerprinter(cfg, trainable=False)
    return m_pre, m_fp


@tf.function
def embeddingGenerator(X, m_pre, m_fp):
    """ 
    X -> (B,1,8000)
    """
    #tf.print(f"X:{X}")
    #feat = m_pre(X)  # (nA+nP, F, T, 1)
    m_fp.trainable = False
    #emb_f = m_fp.front_conv(feat)  # (BSZ, Dim)
    #emb_gf = m_fp.div_enc(emb_f)
    #emb_gf = tf.math.l2_normalize(emb_gf, axis=1)
    
    #return emb_gf # f(.), L2(f(.)), L2(g(f(.))
    return m_fp(m_pre(X))


def generate_h5_files(audio_files, output_root_dir, m_pre, m_fp, dur, hop, fs):
    for file_path in audio_files[20001:30000]:
        base_name = os.path.splitext(os.path.basename(file_path))[0]

        dir_name = f'{base_name[:3]}'

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        audio = load_audio(file_path)
        segments = split_audio_into_segments(audio, dur, hop, fs)
        X_segments = tf.convert_to_tensor(segments, dtype=tf.float32)

        #Tendo cada segmento do audio na forma de tensor de tamanho 1s a um hop de 0.5s, pode-se converter cada segmento num embedding
        emb = [] 
        for i in range(len(X_segments)):
            X = tf.reshape(X_segments[i],(1, 1,-1)) # podia meter um append, mas pode ser uma variável auxiliar, porque só preciso do tensor para gerar o embedding

            embedding = embeddingGenerator(X, m_pre, m_fp)
            emb.append(embedding.numpy())
        
        #Tendo todos os segementos do audio em vetores embedding, pode-se criar um ficheiro .h5 para o audio,pode-se fazer assim:
        #000003.h5
        #vetores:[vetor1, vetor2, vetor3], sendo total de vetores igual ao número de segmentos 
        emb_array = np.array(emb)

        output_file_path = os.path.join(output_root_dir +  dir_name + '/', base_name + '.h5')
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

        with h5py.File(output_file_path, 'w') as hf:
                hf.create_dataset('embeddings', data=emb_array)


def generate_h5_files_new(cfg, audio_files, output_root_dir, m_pre, m_fp, dur, hop, fs):
    for file_path in audio_files[30001:40000]:
        base_name = os.path.splitext(os.path.basename(file_path))[0]

        dir_name = f'{base_name[:3]}' #nome das pastas

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            
        ts_dummy_db_source_fps = file_path

        bsz = ts_batch_sz = 125

        _ts_n_anchor = ts_batch_sz
        ds = genUnbalSequence(
            ts_dummy_db_source_fps,
            ts_batch_sz,
            _ts_n_anchor,
            dur,
            hop,
            fs,
            shuffle=False,
            random_offset_anchor=False,
            drop_the_last_non_full_batch=False)

        enq = tf.keras.utils.OrderedEnqueuer(ds,use_multiprocessing=True,shuffle=False)
        enq.start(workers=cfg['DEVICE']['CPU_N_WORKERS'], max_queue_size=cfg['DEVICE']['CPU_MAX_QUEUE'])

        i = 0
        emb_list = []

        while i < len(enq.sequence):
            X, _ = next(enq.get())
            emb = embeddingGenerator(X, m_pre, m_fp)
            emb_list.append(emb.numpy())
            i += 1
        enq.stop()

        # vetor[4][125] 98
        # vetor (473, 128)
        
        
        #Tendo todos os segementos do audio em vetores embedding, pode-se criar um ficheiro .h5 para o audio,pode-se fazer assim:
        #000003.h5
        #vetores:[vetor1, vetor2, vetor3], sendo total de vetores igual ao número de segmentos
        #emb_array = np.vstack(emb_list)
        #print(emb_array.shape, emb_array) 
        emb_array = np.concatenate(emb_list,axis=0)
        #emb_array_list = np.array(emb_array)

        output_file_path = os.path.join(output_root_dir,  dir_name, base_name + '.h5')

        #os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        #deepdish - encapsula
        
        # save -> dd.io.save(emb_array, output_file_path)
        # load -> emb_array = dd.io.load(output_file_path)

        with h5py.File(output_file_path, 'w') as hf:
                hf.create_dataset('embeddings', data=emb_array)


@cli.command()
def main():
    config = "default_RA"
    cfg = load_config(config)

    dur=cfg['MODEL']['DUR'] 
    hop=cfg['MODEL']['HOP'] 
    fs=cfg['MODEL']['FS'] 

    source_root_dir = cfg['DIR']['SOURCE_ROOT_DIR']
    audio_files = sorted(glob.glob(source_root_dir + 'test-dummy-db-100k-full/' + '**/*.wav', recursive=True))
    output_root_dir = '/mnt/dataset/public/Fingerprinting/Embeddings_BFTRI/dummy_db/'

    m_pre, m_fp = build_fp(cfg)

    checkpoint_root_dir:str = "./logs/CHECK_BFTRI_100/101/"
    checkpoint = tf.train.Checkpoint(m_fp)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_root_dir))

    generate_h5_files_new(cfg, audio_files, output_root_dir, m_pre, m_fp, dur, hop, fs)


if __name__ == '__main__':
    cli()