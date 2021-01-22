from tests.speech_recognition.asr_test_base import TestBaseFairseqModelBase
import torch
import faiss
import math
import numpy as np
from fairseq import utils
import time
from fairseq.data import Dictionary
from fairseq.dima import DimaServer
import os

#from durbango import *
def read_index(indexfile, gpu=False):
    index = faiss.read_index(indexfile, faiss.IO_FLAG_ONDISK_SAME_DIR)
    if not gpu: return index
    res = faiss.StandardGpuResources()  # use a single GPU
    co = faiss.GpuClonerOptions()
    co.useFloat16 = True
    co.useFloat16LookupTables = True
    index = faiss.index_cpu_to_gpu(res, 0, index, co)
    return index

import itertools


from pathlib import Path
def time_index_stuff():
    pass

DEFAULT_INDEX_DIR = '~/knnlm/myle_355/sub_idx/'

class MultiFaiss:

    def __init__(self, index_dir=DEFAULT_INDEX_DIR, n=4):
        files = list(Path(index_dir).glob('*.faiss'))
        if n is not None:
            files = files[:n]
        assert len(files) > 0, f'No patterns matching {index_dir}/*.faiss were found'
        self.index_paths = files
        self.sub_indices = [faiss.read_index(str(f), faiss.IO_FLAG_ONDISK_SAME_DIR) for f in files]
        self.num_indexes = len(self.sub_indices)

        #self.pool = ThreadPool(self.num_indexes)

    def search(self, query, topk: int):

        q_size = query.shape[0]
        res_heap = faiss.ResultHeap(q_size, topk)

        def to_matrix(l, n):
            return [l[i : i + n] for i in range(0, len(l), n)]

        meta = []
        cur_idx = 0
        #all_results = self.pool.imap( lambda idx: idx.search(query, topk), self.sub_indexes, )
        all_results = [idx.search(query, topk) for idx in  self.sub_indices]
        #import ipdb; ipdb.set_trace()
        assert len(all_results[0]) == 2

        for DI, indices in all_results:
            # Two hacks:
            # 2) create new indexed to later map metadata to the best indexes
            #merged = list(itertools.chain(*indices))
            #meta.extend(merged)
            #Ii = np.reshape(np.arange(cur_idx, cur_idx + q_size * topk), (q_size, topk))
            res_heap.add_result(DI, indices)
            #cur_idx += q_size * topk
        res_heap.finalize()
        #ids = np.reshape(res_heap.I, (-1,)).tolist()
        # selected = list(itertools.compress(meta, ids))
        return res_heap.D, res_heap.I

def is_serverlist_path(path:str):
    return True

class KNN_Dstore:
    def __init__(self, args):
        self.half = args.fp16
        self.dimension = args.decoder_embed_dim
        self.k = args.k
        self.dstore_size = args.dstore_size
        self.metric_type = args.faiss_metric_type
        self.sim_func = args.knn_sim_func
        self.dstore_fp16 = args.dstore_fp16     
        if os.path.isdir(args.indexfile):
            index = MultiFaiss(args.indexfile, args.n_idx)
            for x in index.sub_indices:
                x.nprobe = args.probe
        elif args.indexfile.endswith('.index'):
            index = read_index(args.indexfile)
            index.nprobe = args.probe
        else:
            index = DimaServer(args.indexfile)
            index.client.set_nprobe(args.nprobe)
            
        self.index = index

        if args.dstore_fp16:
            print('Keys are fp16 and vals are int16')
            if not args.no_load_keys:
                self.keys = np.memmap(args.dstore_filename+'_keys.npy', dtype=np.float16, mode='r', shape=(self.dstore_size, self.dimension))
            self.vals = np.memmap(args.dstore_filename+'_vals.npy', dtype=np.int16, mode='r', shape=(self.dstore_size, 1))
        else:
            print('Keys are fp32 and vals are int64')
            if not args.no_load_keys:
                self.keys = np.memmap(args.dstore_filename+'_keys.npy', dtype=np.float32, mode='r', shape=(self.dstore_size, self.dimension))
            self.vals = np.memmap(args.dstore_filename+'_vals.npy', dtype=np.int, mode='r', shape=(self.dstore_size, 1))

        # If you wish to load all the keys into memory
        # CAUTION: Only do this if your RAM can handle it!
        if args.move_dstore_to_mem:
            self.move_numpy_to_memory(args)
    
    @property
    def backend(self):
        return type(self.index)
    
    def move_numpy_to_memory(self, args):
        if not args.no_load_keys:
            del self.keys
            self.keys_from_memmap = np.memmap(args.dstore_filename+'_keys.npy', dtype=np.float32, mode='r', shape=(self.dstore_size, self.dimension))
            self.keys = np.zeros((self.dstore_size, self.dimension), dtype=np.float16 if args.dstore_fp16 else np.float32)
            self.keys = self.keys_from_memmap[:]
            self.keys = self.keys.astype(np.float16 if args.dstore_fp16 else np.float32)

        del self.vals
        self.vals_from_memmap = np.memmap(args.dstore_filename+'_vals.npy', dtype=np.int, mode='r', shape=(self.dstore_size, 1))
        self.vals = np.zeros((self.dstore_size, 1), dtype=np.int16 if args.dstore_fp16 else np.int)
        self.vals = self.vals_from_memmap[:]
        self.vals = self.vals.astype(np.int16 if args.dstore_fp16 else np.int)
        #print('Loading to memory took {} s'.format(time.time() - start))

    def get_knns(self, queries):
        qcast = queries.detach().cpu().float().numpy()
        # Note(SS): search only takes numpy array, normalization to prevent faiss returning None
        qn = qcast#/ 
        dists, knns = self.index.search(qn, self.k)
        fac = np.linalg.norm(qcast, ord=2)
        dists1, knns1 = self.index.search(qn/fac, self.k)
        # assert dists.max() < 1e6, 'Huge distance returned.'
        return dists, knns

    def get_knn_log_prob(self, queries, tgt, pad_idx):
        """Equation 2, page 2 in paper. prob = softmax over vocab, sum(1/D[k] for k in neighbors)"""
        def recompute_l2(d, k, q, qsize):
            knns_vecs = torch.from_numpy(self.keys[k]).cuda().view(qsize[0], self.k, -1)
            if self.half:
                knns_vecs = knns_vecs.half()
            query_vecs = q.view(qsize[0], 1, qsize[1]).repeat(1, self.k, 1)
            # TODO(SS): do this with broadcasting, this takes huge memory at the moment.
            l2 = torch.sum((query_vecs - knns_vecs.detach())**2, dim=2)
            return l2

        def dist_func(d, k, q, function=None):
            if not function:
                # Default behavior for L2 metric is to recompute distances.
                # Default behavior for IP metric is to return faiss distances.
                qsize = q.shape
                if self.metric_type == 'l2':
                    l2 = recompute_l2(d, k, q, qsize)
                    #print(f'computed l2 in {time.time()-start}')
                    return -1 * l2
                return d

            if function == 'dot':
                qsize = q.shape
                return (torch.from_numpy(self.keys[k]).cuda() * q.view(qsize[0], 1, qsize[1])).sum(dim=-1)

            if function == 'do_not_recomp_l2':
                return -1 * d

            raise ValueError("Invalid knn similarity function!")

        # queries  are TxBxC
        # reshape: (TxB)xC
        qshape = queries.shape
        queries = queries.view(-1, qshape[-1])
        tgt = tgt.contiguous().view(-1)
        dists, knns = self.get_knns(queries[tgt != pad_idx])
        # (T_reducedxB)xK
        dists = torch.from_numpy(dists).cuda()
        dists = dist_func(dists, knns, queries[tgt != pad_idx, :], function=self.sim_func)
        probs = utils.log_softmax(dists, dim=-1)
        # Probs

        vals = torch.from_numpy(self.vals[knns]).long().cuda().squeeze(-1)
        val_eq_tgt_mask = torch.eq(vals, tgt[tgt != pad_idx].unsqueeze(-1)).float()
        val_eq_tgt_mask[val_eq_tgt_mask == 0] = -10000 # for stability
        val_eq_tgt_mask[val_eq_tgt_mask == 1] = 0

        # (T_reducedxB)
        yhat_knn_prob = torch.logsumexp(probs + val_eq_tgt_mask, dim=-1).clone()
        full_yhat_knn_prob = yhat_knn_prob.new_full([qshape[0]*qshape[1]], -10000)
        full_yhat_knn_prob[tgt != pad_idx] = yhat_knn_prob

        # TxBx1
        return full_yhat_knn_prob.view(qshape[0], qshape[1], 1)

