import glob
from model_RA.utils.dataloader_keras import genUnbalSequence
import os
import sys
import yaml
import numpy as np
from eval_RA.utils.get_index_faiss import get_index
from eval_RA.utils.print_table import PrintTable
import time
from model_RA.fp_RA.melspec.melspectrogram_RA import get_melspec_layer
import tensorflow as tf
from model_RA.fp_RA.nnfp import get_fingerprinter
import curses
import faiss
import click
from pydub import AudioSegment

def ensure_8khz(file_path):
    audio = AudioSegment.from_wav(file_path)
    if audio.frame_rate != 8000:
        audio = audio.set_frame_rate(8000)
        new_file_path = file_path.replace('.wav', '_8kHz.wav')
        audio.export(new_file_path, format='wav')
        print(f'Converted {file_path} to 8kHz and saved as {new_file_path}')
        return new_file_path
    return file_path

def load_config(config_fname):
    config_filepath = './config/' + config_fname + '.yaml'
    if os.path.exists(config_filepath):
        print(f'cli: Configuration from {config_filepath}')
    else:
        sys.exit(f'cli: ERROR! Configuration file {config_filepath} is missing!!')

    with open(config_filepath, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def emb_step(X, m_pre, m_fp):
    m_fp.trainable = False
    return m_fp(m_pre(X))#emb_gf


def load_memmap_data(source_dir,
                     fname,
                     append_extra_length=None,
                     shape_only=False,
                     display=True):

    path_shape = source_dir + fname + '_shape.npy'
    path_data = source_dir + fname + '.mm'
    data_shape = np.load(path_shape)
    if shape_only:
        return data_shape

    if append_extra_length:
        data_shape[0] += append_extra_length
        data = np.memmap(path_data, dtype='float32', mode='r+',
                         shape=(data_shape[0], data_shape[1]))
    else:
        data = np.memmap(path_data, dtype='float32', mode='r',
                         shape=(data_shape[0], data_shape[1]))
    if display:
        print(f'Load {data_shape[0]:,} items from \033[32m{path_data}\033[0m.')
    return data, data_shape


def pesquisa_um_query(query_embeddings, db_embeddings, index_type='ivfpq', nogpu=True, n_centroids=256, code_sz=64, nbits=8, k=1):
    
    d = db_embeddings.shape[1]  # Dim emb



    indice = faiss.IndexFlatL2(d)

    code_sz = 64 # power of 2
    n_centroids = 256#
    nbits = 8  # nbits must be 8, 12 or 16, The dimension d should be a multiple of M.
    index = faiss.IndexIVFPQ(indice, d, n_centroids, code_sz, nbits)

    # Se não usar GPU
    if not nogpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    if not index.is_trained:
        index.train(db_embeddings)

    # Adicionando os embeddings ao índice
    index.add(db_embeddings)

    # Pesquisando no índice
    D, I = index.search(query_embeddings, k)  # D: Distâncias, I: Índices dos resultados

    return D, I



"""
def pesquisa(query_dir, dummy_db, dummy_db_shape, query_1, index_type, nogpu, max_train, test_ids, test_seq_len, k_probe, display_interval):
    index = get_index(index_type, dummy_db, dummy_db.shape, (not nogpu),
                max_train)

    # Add items to index
    start_time = time.time()

    index.add(dummy_db); print(f'{len(dummy_db)} items from dummy DB')
    #index.add(db); print(f'{len(db)} items from reference DB')
    index.add(query_1); print(f'{len(query_1)} items from dummy DB')

    t = time.time() - start_time
    print(f'Added total {index.ntotal} items to DB. {t:>4.2f} sec.')

    # Get test_ids
    print(f'test_id: \033[93m{test_ids}\033[0m,  ', end='')
    test_ids = np.load(
        glob.glob('./**/test_ids_icassp2021.npy', recursive=True)[0])

    n_test = len(test_ids)
    gt_ids  = test_ids + dummy_db_shape[0]
    print(f'n_test: \033[93m{n_test:n}\033[0m')

    # Segement/sequence-level search & evaluation
    # Define metric
    top1_exact = np.zeros((n_test, len(test_seq_len))).astype(int) # (n_test, test_seg_len)
    top1_near = np.zeros((n_test, len(test_seq_len))).astype(int)

    scr = curses.initscr()
    pt = PrintTable(scr=scr, test_seq_len=test_seq_len,
                    row_names=['Top1 exact', 'Top1 near'])
    start_time = time.time()

    for ti, test_id in enumerate(test_ids):
        gt_id = gt_ids[ti]
        for si, sl in enumerate(test_seq_len):
            assert test_id <= len(query_1)
            q = query_1[test_id:(test_id + sl), :] # shape(q) = (length, dim)

            # segment-level top k search for each segment
            _, I = index.search(
                q, k_probe) # _: distance, I: result IDs matrix

            # offset compensation to get the start IDs of candidate sequences
            for offset in range(len(I)):
                I[offset, :] -= offset

            # unique candidates
            candidates = np.unique(I[np.where(I >= 0)])   # ignore id < 0

            # Sequence match score 
            _scores = np.zeros(len(candidates))
            for ci, cid in enumerate(candidates):
                _scores[ci] = np.mean(
                    np.diag(
                        # np.dot(q, index.reconstruct_n(cid, (cid + l)).T)
                        np.dot(q, fake_recon_index[cid:cid + sl, :].T)
                        )
                    )

            # Evaluate 
            pred_ids = candidates[np.argsort(-_scores)[:10]]
            # pred_id = candidates[np.argmax(_scores)] <-- only top1-hit

            # top1 hit
            top1_exact[ti, si] = int(gt_id == pred_ids[0])
            top1_near[ti, si] = int(
                pred_ids[0] in [gt_id - 1, gt_id, gt_id + 1])
            # top1_song = need song info here...



        if (ti != 0) & ((ti % display_interval) == 0):
            avg_search_time = (time.time() - start_time) / display_interval \
                / len(test_seq_len)
            top1_exact_rate = 100. * np.mean(top1_exact[:ti + 1, :], axis=0)
            top1_near_rate = 100. * np.mean(top1_near[:ti + 1, :], axis=0)

            # top1_song = 100 * np.mean(tp_song[:ti + 1, :], axis=0)

            pt.update_counter(ti, n_test, avg_search_time * 1000.)
            pt.update_table((top1_exact_rate, top1_near_rate))
            start_time = time.time() # reset stopwatch
    # Summary
    top1_exact_rate = 100. * np.mean(top1_exact, axis=0)
    top1_near_rate = 100. * np.mean(top1_near, axis=0)

    # top1_song = 100 * np.mean(top1_song[:ti + 1, :], axis=0)

    pt.update_counter(ti, n_test, avg_search_time * 1000.)
    pt.update_table((top1_exact_rate, top1_near_rate))
    pt.close_table() # close table and print summary
    del fake_recon_index, query, db
    np.save(f'{query_dir}/raw_score.npy',
            np.concatenate(
                (top1_exact, top1_near), axis=1))
    np.save(f'{query_dir}/test_ids.npy', test_ids)
    print(f'Saved test_ids and raw score to {query_dir}.')
"""


@click.command()
def main():
    config = "default_RA"
    cfg = load_config(config)

    # Data location
    source_root_dir = '/mnt/dataset/public/Fingerprinting/query_procura'
    #source_root_dir = '/mnt/dataset/public/Fingerprinting/neural-audio-fp-dataset/music/train-10k-30s/fma_small_8k_plus_medium_2k/000'
    #source_root_dir = '/mnt/dev/rodrigoalmeida/Audio'

    # Source (music) file paths
    ts_query_icassp_fps =sorted(glob.glob(source_root_dir + '/000134.wav', recursive=True)) #'/*.wav'
    #ts_query_icassp_fps = sorted(glob.glob(source_root_dir + '/Diatonic_scale_on_C_30s_8kHz.wav', recursive=True))
    #sorted(glob.glob(source_root_dir + '/000002.wav', recursive=True)) #sorted(glob.glob(source_root_dir + '/000134.wav', recursive=True)) #'/*.wav'

    #ts_query_icassp_fps = [ensure_8khz(f) for f in ts_query_icassp_fps] #não eliminar

    # BSZ
    ts_batch_sz = cfg['BSZ']['TS_BATCH_SZ']

    # Model parameters
    dur = cfg['MODEL']['DUR']
    hop = cfg['MODEL']['HOP']
    fs = cfg['MODEL']['FS']

    m_pre = get_melspec_layer(cfg, trainable=False)
    m_fp = get_fingerprinter(cfg, trainable=False)

    _ts_n_anchor = ts_batch_sz
    ds_query = genUnbalSequence(
        ts_query_icassp_fps,
        ts_batch_sz,
        _ts_n_anchor,
        dur,
        hop,
        fs,
        shuffle=False,
        random_offset_anchor=False,
        drop_the_last_non_full_batch=False) # No augmentations...
    
    ds = dict()
    ds['query_3'] = ds_query

    output_root_dir = '/mnt/dev/rodrigoalmeida/neural-audio-fp/logs/emb/query-134/'

    sz_check = dict() # for warning message
    for key in ds.keys():
        bsz = int(cfg['BSZ']['TS_BATCH_SZ'])  # Do not use ds.bsz here.
        # n_items = len(ds[key]) * bsz
        n_items = ds[key].n_samples
        dim = cfg['MODEL']['EMB_SZ']
        #print(key, bsz, n_items, dim)

        # Create memmap, and save shapes
        assert n_items > 0
        arr_shape = (n_items, dim)
        arr = np.memmap(f'{output_root_dir}/{key}.mm',
                        dtype='float32',
                        mode='w+',
                        shape=arr_shape)
        np.save(f'{output_root_dir}/{key}_shape.npy', arr_shape)


        # Fingerprinting loop
        enq = tf.keras.utils.OrderedEnqueuer(ds[key],
                                                use_multiprocessing=True,
                                                shuffle=False)
        enq.start(workers=cfg['DEVICE']['CPU_N_WORKERS'],
                    max_queue_size=cfg['DEVICE']['CPU_MAX_QUEUE'])
        i = 0
        while i < len(enq.sequence):
            X, _ = next(enq.get())
            X = tf.concat(X, axis=0)
            emb = emb_step(X, m_pre, m_fp)
            arr[i * bsz:(i + 1) * bsz, :] = emb.numpy() # Writing on disk.
            i += 1
        enq.stop()

        sz_check[key] = len(arr)
        #print(len(arr))
        arr.flush(); del(arr) # Close memmap


    logsDir = '/mnt/dev/rodrigoalmeida/neural-audio-fp/logs/emb/CHECK_BFTRI_100/101'
    emb_dir = logsDir + '/'
    emb_dummy_dir = None
    index_type='ivfpq'
    nogpu=True
    max_train=1e7
    test_ids='icassp'
    test_seq_len='1'
    k_probe=20
    display_interval=5

    test_seq_len='1'

    test_seq_len = np.asarray(
            list(map(int, test_seq_len.split())))  # '1 3 5' --> [1, 3, 5]

    #query, query_shape = load_memmap_data(emb_dir, 'query')
    #db, db_shape = load_memmap_data(emb_dir, 'db')

    if emb_dummy_dir is None:
        emb_dummy_dir = emb_dir

    dummy_db, dummy_db_shape = load_memmap_data(emb_dummy_dir, 'dummy_db') #embeddings

    query_dir = '/mnt/dev/rodrigoalmeida/neural-audio-fp/logs/emb/query-134/'
    query_1, query_1_shape = load_memmap_data(query_dir, 'query_3')

    # Carregar os nomes das músicas da base de dados
    """
    _prefix = 'train-10k-30s/'
    source_root_dir = '/mnt/dataset/public/Fingerprinting/neural-audio-fp-dataset/music/'
    music_files = sorted(glob.glob(source_root_dir + _prefix + '**/*.wav',
                        recursive=True))
    music_names = [os.path.basename(music_file) for music_file in music_files]
    """

    #pesquisa(query_dir, dummy_db, dummy_db_shape, query_1, index_type, nogpu, max_train, test_ids, test_seq_len, k_probe, display_interval)
    distances, indices = pesquisa_um_query(query_1, dummy_db, index_type='ivfpq', nogpu=True, n_centroids=256, code_sz=64, nbits=8, k=1)
    #pesquisa(query_dir, dummy_db, dummy_db_shape, query_1, index_type, nogpu, max_train, test_ids, test_seq_len, k_probe, display_interval)

    print("Distâncias:", distances)
    print("Índices dos resultados:", indices)


if __name__ == "__main__":
    curses.wrapper(main())