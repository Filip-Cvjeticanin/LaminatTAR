import torch
from sympy import false
from transformers import AutoTokenizer, AutoModel


import random

from load_data import load_dataset, limit_dataset, load_test_dataset, augment_with_sentence_sampling, load_synthetic_dataset, merge_real_and_synthetic

from block_roberta import block_based_embedding
from simple_model import Classifier
from evaluation import evaluate_predictions, print_evaluation_report, log_evaluation_report, log_line
from utils import set_global_seed
from experiments.first_test import first_experiment
from experiments.data_agumentation_coparisson import compare_augmentation_methods
from experiments.increment import incriment_languages


#torch.set_num_threads(10)
#torch.set_num_interop_threads(8)

compare_augmentation_methods()
incriment_languages(["en","fr", "ge"], 432)