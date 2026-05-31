import torch
import random

from sympy import false
from transformers import AutoTokenizer, AutoModel
from load_data import load_dataset, limit_dataset, load_test_dataset, augment_with_sentence_sampling, load_synthetic_dataset, merge_real_and_synthetic
from block_roberta import block_based_embedding
from simple_model import Classifier
from evaluation import evaluate_predictions, print_evaluation_report, log_evaluation_report, log_line
from utils import set_global_seed



def run_experiment(experiment_data):
    # (1) Get the experiment setup =====================================================================================
    log_path = experiment_data["log_path"]
    seed = experiment_data["seed"]
    train_languages = experiment_data["train_languages"]
    test_seen_languages = experiment_data["test_seen_languages"]
    test_unseen_languages = experiment_data["test_unseen_languages"]
    use_synthetic = experiment_data["use_sythetic"]
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


    # Add synthetic data
    if use_synthetic:
        synthetic = load_synthetic_dataset("./synthetic data")
        data = merge_real_and_synthetic(data, synthetic)

    # Add sentence augmented data.
    if use_sent_augmentation:
        data = augment_with_sentence_sampling(data)

    # Shuffle the final data mix.
    random.shuffle(data)
    print("data loaded! :)", len(data))


def compare_augmentation_methods():
    experiment1 = {

    }

    experiment2 = {

    }

    experiment3 = {

    }

    experiment4 = {

    }

    experiments = [
        experiment1,
        experiment2,
        experiment3,
        experiment4
    ]

    for e in experiments:
        run_experiment(e)