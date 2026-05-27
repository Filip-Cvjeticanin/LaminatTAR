import torch
from transformers import AutoTokenizer, AutoModel

from loadData import load_dataset, limit_dataset
from block_roberta import block_based_embedding
from simple_model import Classifier

#load transformer model
model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
roberta = AutoModel.from_pretrained(model_name)

roberta.eval()

#load dataset
print("loading data...")
data = load_dataset("./data", languages=["en"], split="train")
data = limit_dataset(data, ["en"], max_per_lang=10)
print("data loaded! :)")

#mapiranje labela
label_map = { "satire": 0, "opinion": 1, "reporting": 2}

X = []
y = []

print("creating embeddings...")
for item in data:
    emb = block_based_embedding(item["text"], roberta, tokenizer)

    X.append(emb.numpy())
    y.append(label_map[item["label"]])
print("done embeddings! ^^")


clf = Classifier()
clf.train_model(X, y, epochs=5, lr=1e-4)

clf.save_model("classifier.pt")

test_data = load_dataset(
    "./data",
    languages=["fr"],   # unseen language
    split="dev"
)

X_test = []
y_test = []

for item in test_data:
    emb = block_based_embedding(item["text"], roberta, tokenizer)
    X_test.append(emb.numpy())
    y_test.append(label_map[item["label"]])


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!SIMPLE TEST !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
print("\nTesting...")

correct = 0

for x, y_true in zip(X_test, y_test):
    pred = clf.predict(x)["class"]

    if pred == y_true:
        correct += 1

accuracy = correct / len(X_test)

print(f"Test Accuracy: {accuracy:.4f}")


