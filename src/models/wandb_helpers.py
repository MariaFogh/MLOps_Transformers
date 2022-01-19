import wandb
import argparse
from os import environ

def wandb_arg_parser():
    environ["WANDB_API_KEY"] = "719c09fb68fba7368ba93fd0b304d7e1a2fb1a4a"
    environ["WANDB_MODE"] = "online"

    parser = argparse.ArgumentParser()
    args, leftovers = parser.parse_known_args()
    config_defaults = {"learning_rate": 5e-5, "epochs": 10, "batch_size" : 8}

    if hasattr(args, "epochs"):
        config_defaults["epochs"] = args.epochs
    if hasattr(args, "learning_rate"):
        config_defaults["learning_rate"] = args.learning_rate
    if hasattr(args, "batch_size"):
        config_defaults["batch_size"] = args.batch_size

    wandb.init(config=config_defaults, project="MLOps Transformers Google Cloud")

    config = wandb.config
    return config