import argparse
import os
import numpy as np
import faiss
import time

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dstore-mmap', type=str, help='memmap where keys and vals are stored')
    parser.add_argument('--dstore-size', type=int, help='number of items saved in the datastore memmap')
    parser.add_argument('--dimension', type=int, default=1024, help='Size of each key')
    parser.add_argument('--dstore-fp16', default=False, action='store_true')
    parser.add_argument('--seed', type=int, default=1, help='random seed for sampling the subset of vectors to train the cache')

    parser.add_argument('--ntrain', type=int, default=int(1e6), help='number of data points faiss should use to learn centroids.')
    parser.add_argument('--ncentroids', type=int, default=4096, help='number of centroids faiss should learn')
    parser.add_argument('--code-size', type=int, default=64, help='size of quantized vectors')
    parser.add_argument('--probe', type=int, default=32, help='number of clusters to query')
    parser.add_argument('--save-path', type=str, help='file to write the faiss index')
    parser.add_argument('--gpu', action='store_true')
    # parser.add_argument('--num_keys_to_add_at_a_time', default=1000000, type=int,
    #                     help='can only load a certain amount of data to memory at a time.')
    #parser.add_argument('--starting_point', type=int, help='index to start adding keys at')
    args = parser.parse_args()
    return args


def move_index_gpu(index, fp16=True):
        res = faiss.StandardGpuResources()  # use a single GPU
        co = faiss.GpuClonerOptions()
        if fp16:
            co.useFloat16 = True
            co.useFloat16LookupTables = True

        #index = faiss.read_index(args.indexfile, faiss.IO_FLAG_ONDISK_SAME_DIR)
        # faiss.index_cpu_to_all_gpus
        index = faiss.index_cpu_to_gpu(res, 0, index, co)
        return index

from tqdm import tqdm
def add_keys(index, keys, save_path, stop=None, start=0, nk=500000):
    if stop is None:
        stop = len(keys)
    for i in tqdm(range(start, stop, nk)):
        end = min(stop, i+nk)
        to_add = keys[i:end].copy()
        index.add_with_ids(to_add.astype(np.float32), np.arange(i, end))
        faiss.write_index(index, save_path)


def train_faiss(args):
    # Initialize faiss index
    if args.dstore_fp16:
        keys = np.memmap(args.dstore_mmap+'_keys.npy', dtype=np.float16, mode='r', shape=(args.dstore_size, args.dimension))
        # vals = np.memmap(args.dstore_mmap+'_vals.npy', dtype=np.int16, mode='r', shape=(args.dstore_size, 1))
    else:
        keys = np.memmap(args.dstore_mmap+'_keys.npy', dtype=np.float32, mode='r', shape=(args.dstore_size, args.dimension))
        # vals = np.memmap(args.dstore_mmap+'_vals.npy', dtype=np.int, mode='r', shape=(args.dstore_size, 1))
    save_path = args.save_path
    #assert not os.path.exists(save_path)
    quantizer = faiss.IndexFlatL2(args.dimension)
    index = faiss.IndexIVFPQ(quantizer, args.dimension, args.ncentroids, args.code_size, 8)
    index.nprobe = args.probe
    if args.gpu:
        index = move_index_gpu(index, fp16=True)

    print('Training Index')
    np.random.seed(args.seed)
    random_sample = np.random.randint(low=0, high=args.dstore_size, size=args.ntrain, )  #random choice, replace=False?
    print('got random sample')
    train_keys = keys[random_sample].copy().astype(np.float32)
     # Faiss does not handle adding keys in fp16 as of writing this.
    print('Done Sampling')
    start = time.time()
   
    index.train(train_keys)
    print('Training took {:.1f} s'.format(time.time() - start))

    print('Writing index after training')
    start = time.time()
    if args.gpu:
        index_cpu = faiss.index_gpu_to_cpu(index)
        faiss.write_index(index_cpu, save_path)
    else:
        faiss.write_index(index, save_path)
    print('Writing index took {:.1f} s'.format(time.time()-start))
    add_keys(index, keys, args.save_path)
    

if __name__ == '__main__':
    args = get_args()
    train_faiss(args)