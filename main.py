"""Evaluate model."""
import argparse

from train import evaluate, load_model
from data import PolynomialLanguage
from utils import set_seed 
from prettytable import PrettyTable

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dirpath", type=str)
    parser.add_argument("--model_ckpt", type=str, default="model.ckpt")
    parser.add_argument("--data_path", type=str, default="/data/test_set.txt")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    set_seed(args.seed)
    pairs = PolynomialLanguage.load_pairs(args.data_path)
    model = load_model(args.dirpath, args.model_ckpt)
    #print(model)
    #count_parameters(model)
    evaluate(model, pairs)
