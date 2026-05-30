import torch
from transformers import AutoTokenizer, AutoModel
from load_data import load_dataset, limit_dataset
from block_roberta import block_based_embedding
from simple_model import Classifier
from evaluation import evaluate_predictions, print_evaluation_report


def main():
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


    classifier = Classifier()
    classifier.train_model(X, y, epochs=5, lr=1e-4)

    classifier.save_model("classifier.pt")

    test_data = load_dataset(
        "./data",
        languages=["fr"],   # unseen language
        split="dev"
    )

    X_test = []
    y_test = []
    languages_test = []

    for item in test_data:
        emb = block_based_embedding(item["text"], roberta, tokenizer)
        X_test.append(emb.numpy())
        y_test.append(label_map[item["label"]])
        languages_test.append(item["language"])


    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!SIMPLE TEST !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    print("\nTesting...")

    y_pred = []

    for x in X_test:
        pred = classifier.predict(x)["class"]
        y_pred.append(pred)

    metrics = evaluate_predictions(y_true=y_test, y_pred=y_pred, languages=languages_test)
    print_evaluation_report(metrics)


if __name__ == "__main__":
    main()



