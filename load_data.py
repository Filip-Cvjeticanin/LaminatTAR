import os
import random
from collections import defaultdict
from nltk.tokenize import sent_tokenize

def load_dataset(base_path, languages=None, split="train"):
    """
    Reads the TRAIN dataset form folder structure.
    :param base_path:
    :param languages:
    :param split:
    :return:
    """
    dataset = []

    for lang in os.listdir(base_path):
        lang_path = os.path.join(base_path, lang)

        if not os.path.isdir(lang_path):
            continue

        if languages is not None and lang not in languages:
            continue

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



def load_test_dataset(base_path, languages=None):
    """
    Reads the TEST dataset form folder structure.
    :param base_path:
    :param languages:
    :return:
    """
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




def limit_dataset(dataset, languages, max_per_lang, seed=67676767):
    """
    Returns a smaller dataset with filtered languages.
    :param dataset:
    :param languages:
    :param max_per_lang:
    :param seed:
    :return:
    """
    random.seed(seed)

    result = []

    for lang in languages:
        # take only items from this language
        items = [x for x in dataset if x["language"] == lang]
        random.shuffle(items)
        # take up to max_per_lang
        selected = items[:max_per_lang]
        #print(f"{lang}: {len(selected)}")
        result.extend(selected)
    #print(f"Total: {len(result)}")

    return result

def augment_with_sentence_sampling(dataset, sample_ratio=0.3, seed=42):
    random.seed(seed)

    augmented = []

    for item in dataset:
        text = item["text"]

        sentences = sent_tokenize(text)

        if len(sentences) == 0:
            continue

        # take 30% of sentences (at least 1)
        k = max(1, int(len(sentences) * sample_ratio))

        sampled_sentences = random.sample(sentences, k)

        new_text = " ".join(sampled_sentences)

        augmented.append({
            "id": item["id"],
            "text": new_text,
            "label": item.get("label", None),
            "language": item["language"],
            "augmented": True
        })

    return dataset + augmented


from pathlib import Path

def load_synthetic_dataset(base_path):
    dataset = []

    base_path = Path(base_path)

    labels_path = base_path / "synthetic_train_labels.txt"

    # load global labels
    labels = {}
    if labels_path.exists():
        with open(labels_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    article_id, label = parts
                    labels[article_id] = label

    # each language folder
    for lang_folder in base_path.iterdir():
        if not lang_folder.is_dir():
            continue

        lang = lang_folder.name.replace("_synt", "")

        print("Loading:", lang_folder)

        for file in lang_folder.iterdir():
            if file.suffix == ".txt" and file.name.startswith("synthetic_article"):
                article_id = file.stem.replace("synthetic_article", "")

                text = file.read_text(encoding="utf-8")

                dataset.append({
                    "id": article_id,
                    "text": text,
                    "label": labels.get(article_id),
                    "language": lang,
                    "synthetic": True
                })

    print("Synthetic loaded:", len(dataset))
    return dataset

def merge_real_and_synthetic(real_data, synthetic_data):
    merged = []

    # group synthetic by language
    synth_by_lang = defaultdict(list)
    for item in synthetic_data:
        synth_by_lang[item["language"]].append(item)

    # group real by language
    real_by_lang = defaultdict(list)
    for item in real_data:
        real_by_lang[item["language"]].append(item)

    languages = set(real_by_lang.keys()) | set(synth_by_lang.keys())

    for lang in languages:
        merged.extend(real_by_lang[lang])
        merged.extend(synth_by_lang[lang])

    return merged