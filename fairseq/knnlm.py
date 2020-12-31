import torch
import faiss
import math
import numpy as np
from fairseq import utils
import time
from fairseq.data import Dictionary

# from durbango import *
def read_index(indexfile, gpu=True):
    index = faiss.read_index(indexfile, faiss.IO_FLAG_ONDISK_SAME_DIR)
    if not gpu:
        return index
    res = faiss.StandardGpuResources()  # use a single GPU
    co = faiss.GpuClonerOptions()
    co.useFloat16 = True
    co.useFloat16LookupTables = True
    index = faiss.index_cpu_to_gpu(res, 0, index, co)
    return index


class KNN_Dstore:
    def __init__(self, args):
        self.half = args.fp16
        self.dimension = args.decoder_embed_dim
        self.k = args.k
        self.dstore_size = args.dstore_size
        self.metric_type = args.faiss_metric_type
        self.sim_func = args.knn_sim_func
        self.dstore_fp16 = args.dstore_fp16
        index = read_index(args.indexfile, gpu=True)
        print("Reading datastore took {} s".format(time.time() - start))
        index.nprobe = args.probe
        self.index = index

        if args.dstore_fp16:
            print("Keys are fp16 and vals are int16")
            if not args.no_load_keys:
                self.keys = np.memmap(
                    args.dstore_filename + "_keys.npy",
                    dtype=np.float16,
                    mode="r",
                    shape=(self.dstore_size, self.dimension),
                )
            self.vals = np.memmap(
                args.dstore_filename + "_vals.npy",
                dtype=np.int16,
                mode="r",
                shape=(self.dstore_size, 1),
            )
        else:
            print("Keys are fp32 and vals are int64")
            if not args.no_load_keys:
                self.keys = np.memmap(
                    args.dstore_filename + "_keys.npy",
                    dtype=np.float32,
                    mode="r",
                    shape=(self.dstore_size, self.dimension),
                )
            self.vals = np.memmap(
                args.dstore_filename + "_vals.npy",
                dtype=np.int,
                mode="r",
                shape=(self.dstore_size, 1),
            )

        # If you wish to load all the keys into memory
        # CAUTION: Only do this if your RAM can handle it!
        if args.move_dstore_to_mem:
            self.move_numpy_to_memory(args)

    def move_numpy_to_memory(self, args):
        if not args.no_load_keys:
            del self.keys
            self.keys_from_memmap = np.memmap(
                args.dstore_filename + "_keys.npy",
                dtype=np.float32,
                mode="r",
                shape=(self.dstore_size, self.dimension),
            )
            self.keys = np.zeros(
                (self.dstore_size, self.dimension),
                dtype=np.float16 if args.dstore_fp16 else np.float32,
            )
            self.keys = self.keys_from_memmap[:]
            self.keys = self.keys.astype(np.float16 if args.dstore_fp16 else np.float32)

        del self.vals
        self.vals_from_memmap = np.memmap(
            args.dstore_filename + "_vals.npy",
            dtype=np.int,
            mode="r",
            shape=(self.dstore_size, 1),
        )
        self.vals = np.zeros(
            (self.dstore_size, 1), dtype=np.int16 if args.dstore_fp16 else np.int
        )
        self.vals = self.vals_from_memmap[:]
        self.vals = self.vals.astype(np.int16 if args.dstore_fp16 else np.int)
        # print('Loading to memory took {} s'.format(time.time() - start))

    def get_knns(self, queries):
        qcast = queries.detach().cpu().float().numpy()
        # Note(SS): search only takes numpy array
        dists, knns = self.index.search(qcast, self.k)
        return dists, knns

    def get_knn_log_prob(self, queries, tgt, pad_idx):
        """Equation 2, page 2 in paper. prob = softmax over vocab, sum(1/D[k] for k in neighbors)"""
        # import ipdb; ipdb.set_trace()
        def recompute_l2(d, k, q, qsize):
            knns_vecs = torch.from_numpy(self.keys[k]).cuda().view(qsize[0], self.k, -1)
            if self.half:
                knns_vecs = knns_vecs.half()
            query_vecs = q.view(qsize[0], 1, qsize[1]).repeat(1, self.k, 1)
            # TODO(SS): do this with broadcasting
            l2 = torch.sum((query_vecs - knns_vecs.detach()) ** 2, dim=2)
            return l2

        def dist_func(d, k, q, function=None):
            if not function:
                # Default behavior for L2 metric is to recompute distances.
                # Default behavior for IP metric is to return faiss distances.
                qsize = q.shape
                if self.metric_type == "l2":
                    l2 = recompute_l2(d, k, q, qsize)
                    # print(f'computed l2 in {time.time()-start}')
                    return -1 * l2
                return d

            if function == "dot":
                qsize = q.shape
                return (
                    torch.from_numpy(self.keys[k]).cuda()
                    * q.view(qsize[0], 1, qsize[1])
                ).sum(dim=-1)

            if function == "do_not_recomp_l2":
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
        start = time.time()
        dists = dist_func(
            dists, knns, queries[tgt != pad_idx, :], function=self.sim_func
        )
        probs = utils.log_softmax(dists, dim=-1)
        # Probs

        vals = torch.from_numpy(self.vals[knns]).long().cuda().squeeze(-1)
        val_eq_tgt_mask = torch.eq(vals, tgt[tgt != pad_idx].unsqueeze(-1)).float()
        val_eq_tgt_mask[val_eq_tgt_mask == 0] = -10000  # for stability
        val_eq_tgt_mask[val_eq_tgt_mask == 1] = 0

        # (T_reducedxB)
        yhat_knn_prob = torch.logsumexp(probs + val_eq_tgt_mask, dim=-1).clone()
        full_yhat_knn_prob = yhat_knn_prob.new_full([qshape[0] * qshape[1]], -10000)
        full_yhat_knn_prob[tgt != pad_idx] = yhat_knn_prob

        # TxBx1
        return full_yhat_knn_prob.view(qshape[0], qshape[1], 1)
