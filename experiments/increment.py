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
    epochs = experiment_data["epochs"]
    model_path = experiment_data["model_path"]
    lr = experiment_data["lr"]

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

    random.shuffle(data)
    print("data loaded! :)", len(data))
    label_map = { "satire": 0, "opinion": 1, "reporting": 2}
    X = []
    y = []

    # Generate embeddings.
    print("creating embeddings...")
    for i, item in enumerate(data):
        emb = block_based_embedding(item["text"], roberta, tokenizer)
        X.append(emb.detach().cpu().numpy())
        y.append(label_map[item["label"]])
        print("data emb: ", i / len(data) * 100, "%", sep="")
    print("done embeddings! ^^")

    # Save embeddings.
    torch.save({
        "X": X,
        "y": y,
    }, "models/train_incriment.pt")

    classifier = Classifier()
    classifier.train_model(X, y, epochs=epochs, lr=lr)
    classifier.save_model(model_path)

    # (6) Load test data. ==============================================================================================
    # Load data from memory.
    test_data_seen = load_dataset(
        "./data",
        languages=test_seen_languages,
        split="dev"
    )

    test_data_unseen = load_dataset(
        "./data",
        languages=test_unseen_languages,
        split="dev"
    )

    #  TODO remove limiters
    #test_data_seen = limit_dataset(test_data_seen, test_seen_languages, 3, seed)
    #test_data_unseen = limit_dataset(test_data_unseen, test_unseen_languages, 3, seed)

    # Create emb lists.
    X_test_seen = []
    y_test_seen = []
    languages_seen = []

    X_test_unseen = []
    y_test_unseen = []
    languages_test = []

    # Apply embeddings.
    for i, item in enumerate(test_data_seen):
        emb = block_based_embedding(item["text"], roberta, tokenizer)
        X_test_seen.append(emb.detach().cpu().numpy())
        y_test_seen.append(label_map[item["label"]])
        languages_seen.append(item["language"])
        print("test_data emb: ", i / len(test_data_seen) * 100, "%", sep="")

    for i, item in enumerate(test_data_unseen):
        emb = block_based_embedding(item["text"], roberta, tokenizer)
        X_test_unseen.append(emb.detach().cpu().numpy())
        y_test_unseen.append(label_map[item["label"]])
        languages_test.append(item["language"])
        print("test_data_seen emb: ", i / len(test_data_unseen) * 100, "%", sep="")


    # (7) Test model on test data ======================================================================================
    log_line(log_header_line, log_path)

    print("\nTesting seen....")
    y_pred_seen = []
    for x in X_test_seen:
        pred = classifier.predict(x)["class"]
        y_pred_seen.append(int(pred))
    print(y_test_seen)
    print(y_pred_seen)
    metrics_seen = evaluate_predictions(y_true=y_test_seen, y_pred=y_pred_seen, languages=languages_seen)
    print_evaluation_report(metrics_seen)
    log_line("SEEN:", log_path)
    log_evaluation_report(metrics_seen, log_path)

    print("\nTesting unseen...")
    y_pred_unseen = []
    for x in X_test_unseen:
        pred = classifier.predict(x)["class"]
        y_pred_unseen.append(pred)
    metrics = evaluate_predictions(y_true=y_test_unseen, y_pred=y_pred_unseen, languages=languages_test)
    print_evaluation_report(metrics)
    log_line("UNSEEN:", log_path)
    log_evaluation_report(metrics, log_path)



#max mora bit djeljiv s 3 i 2 !
def incriment_languages(languages, max, seed, unseen, lr, epochs):
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
            "use_sent_augmentation": True,
            "lr": lr,
            "epochs": epochs,
            "model_path": "./models/01_no_augmentation.pt",
        }
        experiments.append(experiment)

    for e in experiments:
        run_experiment_lang(e)


