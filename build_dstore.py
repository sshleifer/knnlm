import argparse
import os
import numpy as np
import faiss
import time
from tqdm import tqdm
from pathlib import Path

import unittest
import torch
import random
import string

from distributed_faiss.rpc import Client
from distributed_faiss.server import IndexServer, DEFAULT_PORT
from distributed_faiss.client import IndexClient
from distributed_faiss.index_state import IndexState

import os, pdb, pickle, time, errno, sys, _thread, traceback, socket, threading, gc
import time
import tempfile
import json
import logging

logging.basicConfig(level=4)


# assert os.path.exists(args.faiss_index+".trained")
from distributed_faiss.rpc import Client as RPC

class DimaServer:
    def __init__(
        self,
        path="dima.cfg",
        num_servers=4,
        num_clients=1,
        save_dir="/checkpoint/sshleifer/faiss_idx",
        index_id="lang_en",
        ncentroids=4096,
        server_list_path='/private/home/sshleifer/distributed-faiss/discover.txt',
    ):
        self.index_id = index_id
        self.servers = []
        #ports = []
        self.clients = []
        self.ncentroids = ncentroids
        # TODO(SS): take discover file
        self.client = IndexClient(server_list_path)
        #self.client = RPC('lang_en', 'learnfair5222', port=12033)
        self.port = 12033
        # self.config = {}
        # self.config["index_storage_dir"] = save_dir
        # Path(save_dir).mkdir(exist_ok=True)

        # random.seed(0)
        # for i in range(num_servers):
        #     port = random.randint(1234, 12345)
        #     server = IndexServer(i, self.config)
        #     _thread.start_new_thread(server.start_blocking, (port,))
        #     self.servers.append(server)
        #     ports.append(port)

        # if True:  # not os.path.exists(path):
        #     with open(path, "wb") as fp:
        #         fp.write(f"{num_servers}\n".encode())
        #         for i in range(num_servers):
        #             fp.write(f"localhost,{ports[i]}\n".encode())
        #         fp.seek(0)

        # for i in range(num_clients):
        #     client = IndexClient(path)
        #     self.clients.append(client)

    def tearDown(self):
        [c.close() for c in self.clients]
        [s.stop() for s in self.servers]
        print("Done tearing down")

    def add_vectors(
        self, embeddings, ids, embed_dim=1024, client_id=0, batch_size=1000
    ):
        # start single flat index as souce of truth
        #port = 12345
        # single_server = IndexServer(0, self.config)
        # _thread.start_new_thread(single_server.start_blocking, (port,))
        #cfg = IndexCfg()
        #cfg.dim = embed_dim
        #cfg.ncentroids = 32768
        #cfg.faiss_factory = "flat"
        # single_client = IndexClient(fp.name)
        # single_client.create_index(index_id, cfg)
        # self.assertEqual(single_client.get_state(index_id), IndexState.NOT_TRAINED)

        #client = self.clients[client_id]
        #self.client.create_index(self.index_id, cfg)
        # self.assertEqual(client.get_state(index_id), IndexState.NOT_TRAINED)
        num_vec = embeddings.shape[0]
        since_save = 0
        for i in tqdm(list(range(0, num_vec, batch_size))):
            end = min(i + batch_size, num_vec)
            emb, id = embeddings[i:end].copy(), ids[i:end]
            self.client.add_train_data(self.index_id, emb, id.tolist())
            # single_client.add_index_data(index_id, embeddings, meta)
            # we added training data but did not start training yet
            # self.assertEqual(client.get_state(index_id), IndexState.NOT_TRAINED)
            #
            # if (
            #     self.servers[0].get_ntotal(self.index_id) > 2e6
            #     and client.get_state(self.index_id) == IndexState.NOT_TRAINED
            # ):
            #     client.sync_train(self.index_id)

            since_save += 1
            if since_save >= (1e7 / batch_size):
                since_save = 0
                self.client.save_index()

    def test_index_client_server(self):
        index_id = "lang_en"
        embded_dim = 512
        num_docs_per_batch = 12800
        num_docs_per_query = 16
        num_batches = 4
        topk_per_search = 5
        meta_lenght = 5

        cfg = IndexCfg()
        cfg.dim = embded_dim
        cfg.faiss_factory = "flat"

        for client in self.clients:
            client.create_index(index_id, cfg)
            self.assertEqual(client.get_state(index_id), IndexState.NOT_TRAINED)
            for i in range(num_batches):
                embeddings = torch.rand(num_docs_per_batch, embded_dim).numpy()
                meta = [
                    "".join(
                        random.choices(
                            string.ascii_uppercase + string.digits, k=meta_lenght
                        )
                    )
                    for doc in range(num_docs_per_batch)
                ]
                client.add_index_data(index_id, embeddings, meta)
            # we added training data but did not start training yet
            self.assertEqual(client.get_state(index_id), IndexState.NOT_TRAINED)

        self.clients[0].sync_train(index_id)
        for client in self.clients:
            for i in range(num_batches):
                embeddings = torch.rand(num_docs_per_batch, embded_dim).numpy()
                meta = [
                    "".join(
                        random.choices(
                            string.ascii_uppercase + string.digits, k=meta_lenght
                        )
                    )
                    for doc in range(num_docs_per_batch)
                ]
                client.add_index_data(index_id, embeddings, meta)
        for server in self.servers:
            # Make sure that data is ballanced among servers
            self.assertEqual(
                server.get_ntotal(index_id),
                2
                * num_batches
                * num_docs_per_batch
                * len(self.clients)
                / len(self.servers),
            )
        for client in self.clients:
            self.assertEqual(client.get_state(index_id), IndexState.TRAINED)
            self.assertEqual(
                client.get_ntotal(index_id),
                2 * num_batches * num_docs_per_batch * len(self.clients),
            )
            self.assertEqual(client.get_ntotal("wrong_id"), 0)
            query = torch.rand(num_docs_per_query, embded_dim).numpy()
            scores, meta = client.search(query, topk_per_search, index_id)
            self.assertEqual((num_docs_per_query, topk_per_search), scores.shape)
            self.assertEqual(num_docs_per_query, len(meta))
            self.assertEqual(topk_per_search, len(meta[0]))

        self.clients[0].save_index(index_id)
        self.clients[0].drop_index(index_id)
        for client in self.clients:
            self.assertEqual(client.get_ntotal(index_id), 0)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dstore-mmap", type=str, help="memmap where keys and vals are stored"
)
parser.add_argument(
    "--dstore-size", type=int, help="number of items saved in the datastore memmap"
)
parser.add_argument("--dimension", type=int, default=1024, help="Size of each key")
parser.add_argument("--dstore-fp16", default=False, action="store_true")
# parser.add_argument('--seed', type=int, default=1, help='random seed for sampling the subset of vectors to train the cache')
parser.add_argument(
    "--ncentroids",
    type=int,
    default=4096,
    help="number of centroids faiss should learn",
)
# parser.add_argument('--code-size', type=int, default=64, help='size of quantized vectors')
# parser.add_argument('--probe', type=int, default=8, help='number of clusters to query')
# parser.add_argument('--trained-index', type=str, help='file to write the faiss index')
# parser.add_argument('--save-path', type=str, help='file to write the faiss index')
parser.add_argument(
    "--bs",
    default=1000,
    type=int,
    help="can only load a certain amount of data to memory at a time.",
)
parser.add_argument(
    "--start", default=0, type=int, help="index to start adding keys at"
)
parser.add_argument(
    "--discover", type=str, help="serverlist_path",
)
args = parser.parse_args()

print(args)

if args.dstore_fp16:
    keys = np.memmap(
        args.dstore_mmap + "_keys.npy",
        dtype=np.float16,
        mode="r",
        shape=(args.dstore_size, args.dimension),
    )
    vals = np.memmap(
        args.dstore_mmap + "_vals.npy",
        dtype=np.int16,
        mode="r",
        shape=(args.dstore_size, 1),
    )
else:
    keys = np.memmap(
        args.dstore_mmap + "_keys.npy",
        dtype=np.float32,
        mode="r",
        shape=(args.dstore_size, args.dimension),
    )
    vals = np.memmap(
        args.dstore_mmap + "_vals.npy",
        dtype=np.int,
        mode="r",
        shape=(args.dstore_size, 1),
    )

print("Adding Keys")
#assert os.path.exists(args.trained_index)
server = DimaServer(ncentroids=args.ncentroids, server_list_path=args.discover)
start_time = time.time()
server.add_vectors(
    keys,
    np.arange(
        args.start,
        args.dstore_size,
    ),
    batch_size=args.bs,
)
# for i in tqdm(range(args.start, args.dstore_size, args.nk)):
#     end = min(args.dstore_size, i+args.nk)
#     to_add = keys[i:end].copy()
#     ids = np.arange(i, end)
#     server.add_vectors(to_add, ids)
# break
# index.add_with_ids(to_add.astype(np.float32), )
# faiss.write_index(index, args.save_path)

print("Adding total %d keys" % i)
print("Adding took {} s".format(time.time() - start_time))
print("Writing Index")
i = time.time()
# faiss.write_index(index, args.save_path)
print("Writing index took {} s".format(time.time() - i))
