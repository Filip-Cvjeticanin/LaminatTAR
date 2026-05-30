import torch
from transformers import AutoTokenizer, AutoModel
import random

from loadData import load_dataset, limit_dataset, load_test_dataset, augment_with_sentence_sampling, load_synthetic_dataset, merge_real_and_synthetic
from block_roberta import block_based_embedding
from simple_model import Classifier
from evaluation import evaluate_predictions, print_evaluation_report

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
data = merge_real_and_synthetic(data, synthetic)
print("data loaded! :)", len(data))

#mapiranje labela
label_map = { "satire": 0, "opinion": 1, "reporting": 2}

X = []
y = []

print("creating embeddings...")
for item in data:
    emb = block_based_embedding(item["text"], roberta, tokenizer)

    X.append(emb.detach().cpu().numpy())
    y.append(label_map[item["label"]])
print("done embeddings! ^^")


#save embeddings
torch.save({
    "X": X,
    "y": y,
}, "train_embeddingrealsintetic.pt")


clf = Classifier()
clf.train_model(X, y, epochs=5, lr=1e-4)

clf.save_model("classifier.pt")

test_data = load_dataset(
    "./data",
    languages=["po","it"],   # unseen language
    split="dev"
)

test_data_seen = load_dataset(
    "./data",
    languages=["en","fr","ge","ru"],
    split="dev"
)

X_test = []
y_test = []
languages_test = []

X_test_seen = []
y_test_seen = []
languages_seen = []

for item in test_data:
    emb = block_based_embedding(item["text"], roberta, tokenizer)
    X_test.append(emb.detach().cpu().numpy())
    y_test.append(label_map[item["label"]])
    languages_test.append(item["language"])


for item in test_data_seen:
    emb = block_based_embedding(item["text"], roberta, tokenizer)
    X_test_seen.append(emb.detach().cpu().numpy())
    y_test_seen.append(label_map[item["label"]])
    languages_seen.append(item["language"])


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! TEST !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
print("\nTesting unseen...")

y_pred = []

for x in X_test:
    pred = clf.predict(x)["class"]
    y_pred.append(pred)

metrics = evaluate_predictions(y_true=y_test, y_pred=y_pred, languages=languages_test)
print_evaluation_report(metrics)


print("\nTesting seen....")

y_pred_seen = []


for x in X_test_seen:
    pred = clf.predict(x)["class"]
    y_pred_seen.append(int(pred))

metrics_seen = evaluate_predictions(y_true=y_test_seen, y_pred=y_pred_seen, languages=languages_seen)
print_evaluation_report(metrics_seen)
