import argparse
import numpy as np
import time
from tqdm import tqdm
from pathlib import Path

import torch

from distributed_faiss.client import IndexClient
from distributed_faiss.index_state import IndexState
from distributed_faiss.index_cfg import IndexCfg
import time
import os, time
import time
import logging
from fire import Fire

logging.basicConfig(level=4)

class DimaServer:
    def __init__(
        self,
        server_list_path,
        index_id="lang_en",
        load_index=False,
        idx_cfg_path=None,
    ):
        self.index_id = index_id
        self.client = IndexClient(server_list_path)
        if idx_cfg_path is not None:
            self.cfg = IndexCfg.from_json(idx_cfg_path)
        else:
            self.cfg = IndexCfg(metric='l2', faiss_factory="IVF{centroids},PQ32")
        if not load_index:
            self.client.create_index(self.index_id, self.cfg)
        else:
            self.client.load_index(self.index_id, self.cfg)

    def add_vectors(self, embeddings, ids, bs=1000) -> None:
        num_vec = embeddings.shape[0]
        since_save = 0
        for i in tqdm(list(range(0, num_vec, bs))):
            end = min(i + bs, num_vec)
            emb, id = embeddings[i:end].copy().astype(np.float32), ids[i:end]
            self.client.add_index_data(self.index_id, emb, id.tolist())
            while self.client.get_state(self.index_id) == IndexState.TRAINING:
                time.sleep(1)

            since_save += 1
            if (since_save / self.client.num_indexes) >= (1e7 / bs):
                since_save = 0
                self.client.save_index()

        if self.client.get_state(self.index_id) == IndexState.NOT_TRAINED:
            self.client.sync_train(self.index_id)
        while self.client.get_state(self.index_id) != IndexState.TRAINED:
            time.sleep(10)
        self.search(torch.rand((1, 768)).numpy()) # Sanity Check
        print(f'ntotal: {self.client.get_ntotal(self.index_id)}')

    def search(self, query, k=4):
        return self.client.search(query, k, self.index_id)

def index_data(
    serverlist_path, mmap_file, dstore_size, d, 
    dstore_fp16=True, load_index=False, start=0, bs=1000, cfg=None):
    c= DimaServer(serverlist_path, load_index=load_index, idx_cfg_path=cfg)
    k = np.memmap(mmap_file, shape=(dstore_size, d), mode='r', dtype=np.float16 if dstore_fp16 else np.float32)
    c.add_vectors(k, np.arange(start, dstore_size), bs=bs)
    # Check search results
    d, i = c.search(torch.rand((1, d)).numpy(), k=4)
    assert d.shape == (1,4)
    assert (len(i), len(i[0])) == d.shape
    assert all(x is not None for x in i[0]), 'Found neighbors with None'

if __name__ == '__main__':
    Fire(index_data)