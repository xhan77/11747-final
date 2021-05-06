import argparse
import glob
import logging
import os
import pickle
import random
import re
import csv
from typing import Dict, List, Tuple
import time

import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.nn import CrossEntropyLoss

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    GPT2Config,
    GPT2Tokenizer,
    GPT2Model,
    GPT2PreTrainedModel,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


from model.gpt2_lm import MyGPT2LMHeadModel, MnliDataset, ContraDataset, load_and_cache_examples, set_seed


logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    "gpt2": (GPT2Config, MyGPT2LMHeadModel, GPT2Tokenizer),
}


def gather_flat_grad(grads):
    views = []
    for p in grads:
        if p.data.is_sparse:
            view = p.data.to_dense().view(-1)
        else:
            view = p.data.view(-1)
        views.append(view)
    return torch.cat(views, 0)


def hv(loss, model_params, v):
    grad = autograd.grad(loss, model_params, create_graph=True, retain_graph=True)
    Hv = autograd.grad(grad, model_params, grad_outputs=v)
    return Hv


def get_inverse_hvp_lissa(v, model, param_influence, train_lissa_loader, args):
#     tb_writer = SummaryWriter(args.tensorboard_output_dir)

    ihvp = None
    for i in range(args.lissa_repeat):
        cur_estimate = v
        lissa_data_iterator = iter(train_lissa_loader)
        for j in range(args.lissa_depth):
            try:
                tmp_elem = next(lissa_data_iterator)
                inputs, labels = tmp_elem, tmp_elem
            except StopIteration:
                lissa_data_iterator = iter(train_lissa_loader)
                tmp_elem = next(lissa_data_iterator)
                inputs, labels = tmp_elem, tmp_elem
            inputs = inputs[0].to(args.device) # Han: using the test contrastive correct sentence ([0]) as training for now, need to modify later
            labels = labels[0].to(args.device) # Han: using the test contrastive correct sentence ([0]) as training for now, need to modify later
            
            model.zero_grad()
            outputs = model(inputs, labels=labels)
            train_loss = outputs[0]
            hvp = hv(train_loss, param_influence, cur_estimate)
            cur_estimate = [_a + (1 - args.damping) * _b - _c / args.scale
                            for _a, _b, _c in zip(v, cur_estimate, hvp)]
            
            if (j % args.logging_steps == 0) or (j == args.lissa_depth - 1):
                logger.info(" Recursion at depth %d: norm is %f", j,
                            np.linalg.norm(gather_flat_grad(cur_estimate).cpu().numpy()))
        if ihvp == None:
#             ihvp = [_a / args.scale for _a in cur_estimate]
            ihvp = cur_estimate # Han: no need to scale here again?
        else:
#             ihvp = [_a + _b / args.scale for _a, _b in zip(ihvp, cur_estimate)]
            ihvp = [_a + _b for _a, _b in zip(ihvp, cur_estimate)] # Han: no need to scale here again?
    
    return_ihvp = gather_flat_grad(ihvp)
    return_ihvp /= args.lissa_repeat
    return return_ihvp


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--train_data_file", default=None, type=str, required=True, help="The input training data file (a text file)."
    )
    parser.add_argument(
        "--eval_data_file",
        default=None,
        type=str,
        required=True,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--model_type", type=str, required=True, help="The model architecture to be trained or fine-tuned.",
    )
    parser.add_argument(
        "--task", type=str, required=True, help="Task we are doing, currently MNLI or SBF.",
    )

    # Other parameters
    parser.add_argument(
        "--train_data_field",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--eval_data_field",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )
    parser.add_argument(
        "--tensorboard_output_dir",
        default="tensorboard",
        type=str,
    )

    parser.add_argument(
        "--block_size",
        default=-1,
        type=int,
        help="Optional input sequence length after tokenization."
        "The training dataset will be truncated in block of this size for training."
        "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )

    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    
    parser.add_argument('--damping',
                        type=float,
                        default=0.0,
                        help="probably need damping for deep models")
    parser.add_argument('--scale',
                        type=float,
                        default=1e4,
                        help="probably need scaling for deep models")
    parser.add_argument('--test_idx',
                        type=int,
                        default=1,
                        help="test index we want to examine")
    parser.add_argument("--lissa_repeat",
                        default=1,
                        type=int)
    parser.add_argument("--lissa_depth_pct",
                        default=1.0,
                        type=float)
    parser.add_argument('--start_test_idx',
                        type=int,
                        default=-1,
                        help="when not -1, --test_idx will be disabled")
    parser.add_argument('--end_test_idx',
                        type=int,
                        default=-1,
                        help="when not -1, --test_idx will be disabled")
    parser.add_argument('--influence_metric',
                        type=str,
                        default="IF",
                        help="IF - influence functions, GC - gradient cosine similarity")

    args = parser.parse_args()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device
    
    args.train_batch_size = args.per_gpu_train_batch_size
    args.eval_batch_size = 1 # Han: We need to pass in test example one by one because we will do HVP

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Set seed
    set_seed(args)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    config = config_class.from_pretrained(args.model_name_or_path)

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

    if args.block_size <= 0:
        args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        args.block_size = min(args.block_size, tokenizer.max_len)

    # Load model
    model = model_class.from_pretrained(args.model_name_or_path, config=config)

    model.to(args.device)
    
    # Load dataset
    train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)
    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            if args.task == "contra":
                return pad_sequence(list(map(lambda x: x[:][0], examples)), batch_first=True, padding_value=2),\
                       pad_sequence(list(map(lambda x: x[:][1], examples)), batch_first=True, padding_value=2)
            return pad_sequence(examples, batch_first=True, padding_value=2) # Han: GPT2 has no paddings. We use # (index=2) as padding so that the loss function knows to ignore.
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    train_lissa_sampler = RandomSampler(train_dataset)
    train_lissa_dataloader = DataLoader(
        train_dataset, sampler=train_lissa_sampler, batch_size=args.train_batch_size, collate_fn=collate
    )
    
    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.eval_batch_size, collate_fn=collate
    )

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate
    )
    
    # Prepare optimizer and schedule (linear warmup and decay)
    param_influence = list(model.parameters())
    model_size = 0
    for p in param_influence:
        tmp_p = p.clone().detach()
        model_size += torch.numel(tmp_p)
    logger.info(" model size = %d parameters", model_size)
        
    # Lissa depth calculation
    args.lissa_depth = int(args.lissa_depth_pct * len(train_dataset))
    
    # Influence functions!
    influence_dict = dict()
    ihvp_dict = dict()
    
    for tmp_idx, batch in enumerate(eval_dataloader):
        if args.start_test_idx != -1 and args.end_test_idx != -1:
            if tmp_idx < args.start_test_idx:
                continue
            if tmp_idx > args.end_test_idx:
                break
        else:
            if tmp_idx < args.test_idx:
                continue
            if tmp_idx > args.test_idx:
                break
                
        if args.task != "contra":
            raise ValueError("only supporting contrastive setup now")

        inputs, labels = batch, batch
        correct_inputs, incorrect_inputs = inputs[0].to(args.device), inputs[1].to(args.device)
        correct_labels, incorrect_labels = labels[0].to(args.device), labels[1].to(args.device)
        
        if len(correct_inputs[0]) <= 1 or len(incorrect_inputs[0]) <= 1: # prevent empty label after shift since we have no <SOS>
            continue
        influence_dict[tmp_idx] = np.zeros(len(train_dataset))
        
        ######## L_TEST GRADIENT ########
        model.eval()
        model.zero_grad()
        correct_outputs = model(correct_inputs, labels=correct_labels)
        incorrect_outputs = model(incorrect_inputs, labels=incorrect_labels)
        lm_loss = correct_outputs[0]
        test_loss = correct_outputs[0] - incorrect_outputs[0] # this is the diff_loss
        
        test_grads = autograd.grad(test_loss, param_influence)
        ################
        
        set_seed(args)
        
        if args.influence_metric == "IF":
            ######## IHVP ########
            model.train()
            logger.info("######## START COMPUTING IHVP (IDX %d) ########", tmp_idx)
            inverse_hvp = get_inverse_hvp_lissa(test_grads, model, param_influence, train_lissa_dataloader, args)
            logger.info("######## FINISHED COMPUTING IHVP (IDX %d) ########", tmp_idx)
            ################
            ihvp_dict[tmp_idx] = inverse_hvp.detach().cpu() # put to CPU to save GPU memory
        elif args.influence_metric == "GC":
            ihvp_dict[tmp_idx] = gather_flat_grad(test_grads).detach().cpu() # put to CPU to save GPU memory
        else:
            raise ValueError("N/A")
        
    # Han: put ihvps back to GPU, but may run out of GPU memory
    for tmp_idx in ihvp_dict.keys():
        ihvp_dict[tmp_idx] = ihvp_dict[tmp_idx].to(args.device)
    
    set_seed(args)
    for train_idx, train_batch in enumerate(tqdm(train_dataloader, desc="Train data gradient")):
        _inputs, _labels = train_batch, train_batch
        
        _inputs = _inputs[0].to(args.device) # Han: using the contrastive correct sentence ([0]) as training for now, need to modify later
        _labels = _labels[0].to(args.device) # Han: using the contrastive correct sentence ([0]) as training for now, need to modify later
        
        if len(_inputs[0]) <= 1: # prevent empty label after shift since we have no <SOS>
            continue

        ######## L_TRAIN GRADIENT ########
        model.eval()
        model.zero_grad()
        _outputs = model(_inputs, labels=_labels)
        train_loss = _outputs[0]
        train_grads = autograd.grad(train_loss, param_influence)
        ################
        
        with torch.no_grad():
            for tmp_idx in ihvp_dict.keys():
                if args.influence_metric == "IF":
                    influence_dict[tmp_idx][train_idx] = torch.dot(ihvp_dict[tmp_idx], gather_flat_grad(train_grads)).item()
                elif args.influence_metric == "GC":
                    influence_dict[tmp_idx][train_idx] = nn.functional.cosine_similarity(ihvp_dict[tmp_idx], gather_flat_grad(train_grads),
                                                                                         dim=0, eps=1e-12).item()
                else:
                    raise ValueError("N/A")
    
#     print(influence_dict) # Han: debug
    for k, v in influence_dict.items():
        print("test idx:", k, " -- most influential train idx:", np.argsort(v)[:10])
    
    for k, v in influence_dict.items():
        influence_filename = f"influence_test_idx_{k}.pkl"
        pickle.dump(v, open(os.path.join(args.output_dir, influence_filename), "wb"))


if __name__ == "__main__":
    main()
