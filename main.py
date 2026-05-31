import torch
from sympy import false
from transformers import AutoTokenizer, AutoModel


import random

from load_data import load_dataset, limit_dataset, load_test_dataset, augment_with_sentence_sampling, load_synthetic_dataset, merge_real_and_synthetic

from block_roberta import block_based_embedding
from simple_model import Classifier
from evaluation import evaluate_predictions, print_evaluation_report, log_evaluation_report, log_line
from utils import set_global_seed


def experiment(synthetic_data):

    seed = 67
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    #load transformer model
    model_name = "xlm-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    roberta = AutoModel.from_pretrained(model_name).to(device)
    roberta.eval()

    #load dataset
    print("loading data...")
    data = load_dataset("./data", languages=["en","fr","ge","ru"],split="train")
    #data = limit_dataset(data, ["en","fr","ge","ru"], max_per_lang=10)
    #data = augment_with_sentence_sampling(data)
    #random.shuffle(data)
    synthetic = load_synthetic_dataset("./synthetic data")
    if synthetic_data:
        data = merge_real_and_synthetic(data, synthetic)
    #data = limit_dataset(data, ["en","fr","ge","ru"], max_per_lang=50)
    print("data loaded! :)", len(data))



    #mapiranje labela
    label_map = { "satire": 0, "opinion": 1, "reporting": 2}

    X = []
    y = []

    print("creating embeddings...")
    for i, item in enumerate(data):
        emb = block_based_embedding(item["text"], roberta, tokenizer)

        X.append(emb.detach().cpu().numpy())
        y.append(label_map[item["label"]])

        print("data emb: ", i / len(data) * 100, "%", sep="")


    #X.append(emb.detach().cpu().numpy())
    #y.append(label_map[item["label"]])
    print("done embeddings! ^^")


    #save embeddings
    torch.save({
        "X": X,
        "y": y,
    }, "models/train_embeddingrealsintetic.pt")

    classifier = Classifier()
    classifier.train_model(X, y, epochs=5000, lr=1e-3)
    classifier.save_model("classifier.pt")


    test_data = load_dataset(
        "./data",
        languages=["po","it"],   # unseen language
        split="dev"
    )
    #test_data = limit_dataset(test_data, ["po","it"], max_per_lang=10)

    test_data_seen = load_dataset(
        "./data",
        languages=["en","fr","ge","ru"],
        split="dev"
    )

    #test_data_seen = limit_dataset(test_data_seen, ["en","fr","ge","ru"], max_per_lang=10)

    X_test = []
    y_test = []
    languages_test = []


    X_test_seen = []
    y_test_seen = []
    languages_seen = []

    for i, item in enumerate(test_data):
        emb = block_based_embedding(item["text"], roberta, tokenizer)
        X_test.append(emb.detach().cpu().numpy())
        y_test.append(label_map[item["label"]])
        languages_test.append(item["language"])

        print("test_data emb: ", i / len(test_data) * 100, "%", sep="")


    for i, item in enumerate(test_data_seen):
        emb = block_based_embedding(item["text"], roberta, tokenizer)
        X_test_seen.append(emb.detach().cpu().numpy())
        y_test_seen.append(label_map[item["label"]])
        languages_seen.append(item["language"])

        print("test_data_seen emb: ", i / len(test_data_seen) * 100, "%", sep="")










    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! TEST !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    print("\nTesting seen....")

    y_pred_seen = []


    for x in X_test_seen:
        pred = classifier.predict(x)["class"]
        y_pred_seen.append(int(pred))

    metrics_seen = evaluate_predictions(y_true=y_test_seen, y_pred=y_pred_seen, languages=languages_seen)
    print_evaluation_report(metrics_seen)
    log_line("SEEN:", "./logs/log.txt")
    log_evaluation_report(metrics_seen, "./logs/log.txt")


    print("\nTesting unseen...")


    y_pred = []

    for x in X_test:
        pred = classifier.predict(x)["class"]
        y_pred.append(pred)

    metrics = evaluate_predictions(y_true=y_test, y_pred=y_pred, languages=languages_test)
    print_evaluation_report(metrics)
    log_line("UNSEEN:", "./logs/log.txt")
    log_evaluation_report(metrics, "./logs/log.txt")




torch.manual_seed(42)
random.seed(42)
log_line("-------non-synt:----------:", "./logs/log.txt")
experiment(False)
log_line("-------synt:----------:", "./logs/log.txt")
experiment(True)