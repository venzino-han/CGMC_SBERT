from prettytable import PrettyTable


import math, copy

import torch as th
import numpy as np
import torch.nn as nn
from torch import optim

import time
from easydict import EasyDict

from utils import get_logger, get_args_from_yaml, evaluate

from dataset import get_dataloader

# from models.egmc import EGMC
from baselines.igmc import IGMC
from models.cgmc_v2 import CGMC



def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
    

import yaml

def main():
    with open('./train_configs/train_list.yaml') as f:
        files = yaml.load(f, Loader=yaml.FullLoader)
        file_list = files['files']
        for f in file_list:
            print(f)
            args = get_args_from_yaml(f)

            in_feats = (args.hop+1)*2
            if args.model_type == 'IGMC':
                model = IGMC(in_feats=in_feats,
                            latent_dim=args.latent_dims,
                            num_relations=5,
                            num_bases=4,
                            regression=True,
                            edge_dropout=args.edge_dropout,
                            )
            elif args.model_type == 'CGMC':
                model = CGMC(in_nfeats=args.in_nfeats,
                                out_nfeats=args.out_nfeats,
                                in_efeats=args.in_efeats,
                                num_heads=args.num_heads,
                                review=args.review,
                                rating=args.rating,
                                timestamp=args.timestamp,
                                node_features=args.node_features,
                                )

            count_parameters(model)


if __name__ == '__main__':
    main()