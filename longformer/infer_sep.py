import argparse
import os
import random
import warnings

import numpy as np
import pandas as pd
import tez
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from utils import prepare_training_data, evaluate, target_id_map

from ast import literal_eval

warnings.filterwarnings("ignore")


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, default="model", required=False)
    parser.add_argument("--input", type=str, default="../data", required=False)
    parser.add_argument("--max_len", type=int, default=1024, required=False)
    parser.add_argument("--valid_batch_size", type=int, default=8, required=False)
    return parser.parse_args()

class FeedbackModel(tez.Model):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        config = AutoConfig.from_pretrained(model_name)

        hidden_dropout_prob: float = 0.1
        layer_norm_eps: float = 1e-7
        config.update(
            {
                "output_hidden_states": True,
                "hidden_dropout_prob": hidden_dropout_prob,
                "layer_norm_eps": layer_norm_eps,
                "add_pooling_layer": False,
            }
        )
        self.transformer = AutoModel.from_config(config)
        self.output = nn.Linear(config.hidden_size, self.num_labels)

    def forward(self, ids, mask):
        transformer_out = self.transformer(ids, mask)
        sequence_output = transformer_out.last_hidden_state
        logits = self.output(sequence_output)
        logits = torch.softmax(logits, dim=-1)
        return logits, 0, {}


if __name__ == "__main__":
    NUM_JOBS = 12
    args = parse_args()
    seed_everything(42)
    os.makedirs(args.output, exist_ok=True)
    df = pd.read_csv(os.path.join(args.input, "train_folds.csv"))

    valid_df = df[df["kfold"] == args.fold].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    valid_samples = prepare_training_data('test', valid_df, tokenizer, args, num_jobs=NUM_JOBS)
    # print(tokenizer.decode(training_samples[0]['input_ids']))

    model = FeedbackModel(model_name=args.model, num_labels=len(target_id_map) - 1)
    model.load(os.path.join(args.output, f"model_{args.fold}.bin"), weights_only=True)

    score = evaluate(model, valid_samples, tokenizer, args.valid_batch_size, valid_df, "max")


