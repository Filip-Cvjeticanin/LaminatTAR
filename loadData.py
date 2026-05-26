import os

def load_dataset(base_path, languages=None, split="train"):
    dataset = []

    for lang in os.listdir(base_path):
        lang_path = os.path.join(base_path, lang)

        if not os.path.isdir(lang_path):
            continue

        if languages is not None and lang not in languages:
            continue

        # 🔹 use split here
        articles_path = os.path.join(lang_path, f"{split}-articles-subtask-1")
        labels_path = os.path.join(lang_path, f"{split}-labels-subtask-1.txt")

        # skip if missing
        if not os.path.exists(labels_path) or not os.path.exists(articles_path):
            print(f"Skipping {lang} ({split} data missing)")
            continue

        # load labels
        labels = {}
        with open(labels_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    article_id, label = parts
                    labels[article_id] = label

        # load articles
        for filename in os.listdir(articles_path):
            if filename.endswith(".txt"):
                article_id = filename.replace("article", "").replace(".txt", "")
                file_path = os.path.join(articles_path, filename)

                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()

                label = labels.get(article_id, None)

                dataset.append({
                    "id": article_id,
                    "text": text,
                    "label": label,
                    "language": lang,
                    "split": split   # optional but useful
                })

    return dataset

def count_articles_in_dataset(dataset, language):
    return sum(1 for item in dataset if item["language"] == language)

import os

def load_test_dataset(base_path, languages=None):
    dataset = []

    for lang in os.listdir(base_path):
        lang_path = os.path.join(base_path, lang)

        if not os.path.isdir(lang_path):
            continue

        if languages is not None and lang not in languages:
            continue

        articles_path = os.path.join(lang_path, "test-articles-subtask-1")

        # skip missing folders
        if not os.path.exists(articles_path):
            print(f"Skipping {lang} (no test folder)")
            continue

        for filename in os.listdir(articles_path):
            if filename.endswith(".txt"):
                article_id = filename.replace("article", "").replace(".txt", "")
                file_path = os.path.join(articles_path, filename)

                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()

                dataset.append({
                    "id": article_id,
                    "text": text,
                    "language": lang
                })

    return dataset