import torch
import random
from sympy import false
from transformers import AutoTokenizer, AutoModel
from load_data import load_dataset, limit_dataset, load_test_dataset, augment_with_sentence_sampling, load_synthetic_dataset, merge_real_and_synthetic
from block_roberta import block_based_embedding
from simple_model import Classifier
from evaluation import evaluate_predictions, print_evaluation_report, log_evaluation_report, log_line
from utils import set_global_seed
from load_data import balance_and_limit_dataset



def run_experiment_lang(experiment_data):
    # Get the experiment setup
    log_path = experiment_data["log_path"]
    seed = experiment_data["seed"]
    train_languages = experiment_data["train_languages"]
    test_seen_languages = experiment_data["test_seen_languages"]
    test_unseen_languages = experiment_data["test_unseen_languages"]
    use_sythetic = experiment_data["use_sythetic"]
    use_sent_augmentation = experiment_data["use_sent_augmentation"]

    # (2) Set global seed and use gpu ==================================================================================
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)


    # (3) Setup data and model==========================================================================================
    # Load tokenizer and embedder generator.
    model_name = "xlm-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    roberta = AutoModel.from_pretrained(model_name).to(device)
    roberta.eval()

    # Load dataset
    print("loading data...")
    data = load_dataset("./data", languages=train_languages, split="train")
    max_per_lang = experiment_data["max_per_lang"]
    data = balance_and_limit_dataset(data, train_languages, max_per_lang, seed=seed)


#max mora bit djeljiv s 3 i 2 !
def incriment_languages(languages, max, seed, unseen):
    experiments = []
    for i in range(1, len(languages)+1):
        experimentlang = languages[:i]

        max_per_lang = max // len(experimentlang)

        experiment = {
            "name": f"exp_{i}",
            "train_languages": experimentlang,
            "max_per_lang": max_per_lang,
            "log_path": f"./logs/exp_{i}.txt",
            "seed": seed,
            "test_seen_languages": experimentlang,
            "test_unseen_languages": [],
            "use_sythetic": False,
            "use_sent_augmentation": True
        }
        experiments.append(experiment)

    for e in experiments:
        run_experiment_lang(e)


