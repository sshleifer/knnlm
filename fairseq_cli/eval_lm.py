#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluate the perplexity of a trained language model.
"""

import logging
import math
import os

import torch
import numpy as np

from fairseq import checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.data import LMContextWindowDataset
from fairseq.meters import StopwatchMeter, TimeMeter
from fairseq.sequence_scorer import SequenceScorer
from fairseq.knnlm import KNN_Dstore, read_index
import faiss


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("fairseq_cli.eval_lm")


class WordStat(object):
    def __init__(self, word, is_bpe):
        self.word = word
        self.is_bpe = is_bpe
        self.log_prob = 0
        self.next_word_prob = 0
        self.count = 0
        self.missing_next_words = 0

    def add(self, log_prob, next_word_prob):
        """increments counters for the sum of log probs of current word and next
        word (given context ending at current word). Since the next word might be at the end of the example,
        or it might be not counted because it is not an ending subword unit,
        also keeps track of how many of those we have seen"""
        if next_word_prob is not None:
            self.next_word_prob += next_word_prob
        else:
            self.missing_next_words += 1
        self.log_prob += log_prob
        self.count += 1

    def __str__(self):
        return "{}\t{}\t{}\t{}\t{}\t{}".format(
            self.word,
            self.count,
            self.log_prob,
            self.is_bpe,
            self.next_word_prob,
            self.count - self.missing_next_words,
        )


def main(parsed_args):
    assert parsed_args.path is not None, "--path required for evaluation!"

    utils.import_user_module(parsed_args)

    logger.info(parsed_args)

    use_cuda = torch.cuda.is_available() and not parsed_args.cpu

    task = tasks.setup_task(parsed_args)

    # Load ensemble
    logger.info("loading model(s) from {}".format(parsed_args.path))
    models, args = checkpoint_utils.load_model_ensemble(
        parsed_args.path.split(os.pathsep),
        arg_overrides=eval(parsed_args.model_overrides),
        task=task,
    )

    for arg in vars(parsed_args).keys():
        if arg not in {
            "self_target",
            "future_target",
            "past_target",
            "tokens_per_sample",
            "output_size_dictionary",
            "add_bos_token",
        }:
            setattr(args, arg, getattr(parsed_args, arg))

    # reduce tokens per sample by the required context window size
    args.tokens_per_sample -= args.context_window
    task = tasks.setup_task(args)

    # Load dataset splits
    task.load_dataset(args.gen_subset)
    dataset = task.dataset(args.gen_subset)
    if args.context_window > 0:
        dataset = LMContextWindowDataset(
            dataset=dataset,
            tokens_per_sample=args.tokens_per_sample,
            context_window=args.context_window,
            pad_idx=task.source_dictionary.pad(),
        )
    logger.info("{} {} {} examples".format(args.data, args.gen_subset, len(dataset)))

    # Optimize ensemble for generation and set the source and dest dicts on the model (required by scorer)
    for model in models:
        model.make_generation_fast_()
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    assert len(models) > 0

    logger.info(
        "num. model params: {}".format(sum(p.numel() for p in models[0].parameters()))
    )

    itr = task.get_batch_iterator(
        dataset=dataset,
        max_tokens=args.max_tokens or 36000,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=True,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=args.shuffle)

    gen_timer = StopwatchMeter()
    scorer = SequenceScorer(task.target_dictionary, args.softmax_batch, args=args)

    score_sum = 0.0
    count = 0

    if args.remove_bpe is not None:
        if args.remove_bpe == "sentencepiece":
            raise NotImplementedError
        else:
            bpe_cont = args.remove_bpe.rstrip()
            bpe_toks = {
                i
                for i in range(len(task.source_dictionary))
                if task.source_dictionary[i].endswith(bpe_cont)
            }
        bpe_len = len(bpe_cont)
    else:
        bpe_toks = None
        bpe_len = 0

    word_stats = dict()

    if args.knnlm and args.save_knnlm_dstore:
        raise ValueError("Cannot use knnlm while trying to build the datastore!")

    if args.knnlm:
        knn_dstore = KNN_Dstore(args)
    if args.save_knnlm_dstore and args.save_index:
        assert args.indexfile is not None
        index = read_index(args.indexfile, gpu=False)
    FLUSH_THRESH = 5000000
    with progress_bar.build_progress_bar(args, itr) as t:
        wps_meter = TimeMeter()

        if args.save_knnlm_dstore:
            print("keytype being saved:", args.knn_keytype)
            dstore_vals = np.memmap(
                args.dstore_mmap + "_vals.npy",
                dtype=np.int32,
                mode="w+",
                shape=(args.dstore_size, 1),
            )
            if args.save_keys:
                dstore_keys = np.memmap(
                    args.dstore_mmap + "_keys.npy",
                    dtype=np.float16,
                    mode="w+",
                    shape=(args.dstore_size, args.decoder_embed_dim),
                )
        # np_dtypes = (np.float16, np.int16) if args.dstore_fp16 else (np.float32, np.int)
        dstore_idx = 0
        buffer_k = []
        last_buffer_idx = 0
        for ex_i, sample in enumerate(t):
            if "net_input" not in sample:
                continue
            elif args.save_knnlm_dstore and dstore_idx >= args.dstore_size:
                break

            sample = utils.move_to_cuda(sample) if use_cuda else sample

            gen_timer.start()
            if args.knnlm:
                hypos = scorer.generate(models, sample, knn_dstore=knn_dstore)
            else:
                hypos = scorer.generate(models, sample)
            gen_timer.stop(sample["ntokens"])

            for i, hypos_i in enumerate(hypos):
                hypo = hypos_i[0]
                dk, dv = hypo["dstore_keys"], hypo["tokens"]
                if args.save_knnlm_dstore:
                    num_to_add = dk.shape[0]
                    end_idx = num_to_add + dstore_idx
                    if num_to_add == args.tokens_per_sample:
                        if end_idx > args.dstore_size:
                            num_to_add = (
                                args.dstore_size - dstore_idx
                            )  # however much left.
                            print(f"Last obs: only have space to add {num_to_add}")
                            dk = dk[:num_to_add]
                            dv = dv[:num_to_add]
                            end_idx = num_to_add + dstore_idx  # should be dstore_size
                        keys_to_add = dk.view(-1, args.decoder_embed_dim).cpu().numpy()
                        # index.add_with_ids(keys_to_add, np.arange(dstore_idx, end_idx))
                        if args.save_keys:
                            dstore_keys[dstore_idx:end_idx] = keys_to_add.astype(
                                dstore_keys.dtype
                            )
                        if args.save_index:
                            buffer_k.append(keys_to_add.astype(np.float32))
                        dstore_vals[dstore_idx:end_idx] = (
                            dv.view(-1, 1).cpu().numpy().astype(np.int16)
                        )
                        dstore_idx += num_to_add
                    else:
                        print(f"Skipping this one with shape: {dk.shape}")

                sample_id = sample["id"][i]

                tokens = hypo["tokens"]
                tgt_len = tokens.numel()
                pos_scores = hypo["positional_scores"].float()

                if args.add_bos_token:
                    assert hypo["tokens"][0].item() == task.target_dictionary.bos()
                    tokens = tokens[1:]
                    pos_scores = pos_scores[1:]

                skipped_toks = 0
                if bpe_toks is not None:
                    for i in range(tgt_len - 1):
                        if tokens[i].item() in bpe_toks:
                            skipped_toks += 1
                            pos_scores[i + 1] += pos_scores[i]
                            pos_scores[i] = 0
                score_sum += pos_scores.sum().cpu()
                count += pos_scores.numel() - skipped_toks

            if args.output_word_probs or args.output_word_stats:
                w = ""
                word_prob = []
                is_bpe = False
                for i in range(len(tokens)):
                    w_ind = tokens[i].item()
                    w += task.source_dictionary[w_ind]
                    if bpe_toks is not None and w_ind in bpe_toks:
                        w = w[:-bpe_len]
                        is_bpe = True
                    else:
                        word_prob.append((w, pos_scores[i].item()))

                        next_prob = None
                        ind = i + 1
                        while ind < len(tokens):
                            if pos_scores[ind].item() != 0:
                                next_prob = pos_scores[ind]
                                break
                            ind += 1

                        word_stats.setdefault(w, WordStat(w, is_bpe)).add(
                            pos_scores[i].item(), next_prob
                        )
                        is_bpe = False
                        w = ""
                if args.output_word_probs:
                    logger.info(
                        str(int(sample_id))
                        + " "
                        + (
                            "\t".join(
                                "{} [{:2f}]".format(x[0], x[1]) for x in word_prob
                            )
                        )
                    )
            if args.save_index and (dstore_idx - last_buffer_idx) > FLUSH_THRESH:
                # import ipdb; ipdb.set_trace()
                keys_to_add = np.vstack(buffer_k)
                ids = np.arange(last_buffer_idx, dstore_idx)
                index.add_with_ids(keys_to_add, ids)
                faiss.write_index(index, args.indexfile + "_populated")
                last_buffer_idx = dstore_idx
                buffer_k = []

            wps_meter.update(sample["ntokens"])
            t.log({"wps": round(wps_meter.avg), "dstore_idx": dstore_idx})

        print(f"final dstore idx: {dstore_idx}")

    if args.save_knnlm_dstore and args.save_index:
        print("Keys", dstore_keys.shape, dstore_keys.dtype)
        print("Vals", dstore_vals.shape, dstore_vals.dtype)
        keys_to_add = np.vstack(buffer_k)
        ids = np.arange(last_buffer_idx, args.dstore_size)
        index.add_with_ids(keys_to_add, ids)
        faiss.write_index(index, args.indexfile + "_populated")
    avg_nll_loss = -score_sum / count / math.log(2)  # convert to base 2
    logger.info(
        "Evaluated {} tokens in {:.1f}s ({:.2f} tokens/s)".format(
            gen_timer.n, gen_timer.sum, 1.0 / gen_timer.avg
        )
    )
    logger.info(
        "Loss (base 2): {:.4f}, Perplexity: {:.2f}".format(
            avg_nll_loss, 2 ** avg_nll_loss
        )
    )

    if args.output_word_stats:
        for ws in sorted(word_stats.values(), key=lambda x: x.count, reverse=True):
            logger.info(ws)
    # if args.train_after:
    # from train_faiss import train_faiss
    # train_faiss()


def cli_main():
    parser = options.get_eval_lm_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == "__main__":
    cli_main()
