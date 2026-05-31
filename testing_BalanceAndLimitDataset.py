from load_data import load_dataset, limit_dataset, balance_and_limit_dataset

print("Test 1:")
seed = 42
languages = ['en', 'fr', "ge"]
data_en = load_dataset("./data", ['en'], "train")
data_fr = load_dataset("./data", ['fr'], "train")
data_ge = load_dataset("./data", ['ge'], "train")
data_en = limit_dataset(data_en, ['en'], 12)
data_fr = limit_dataset(data_fr, ['fr'], 3)
data_ge = limit_dataset(data_ge, ['ge'], 2)

lang_num = len(languages)
items_per_lang = int(len(data_en) / lang_num)

print(lang_num, items_per_lang)
data_combined = data_en + data_fr + data_ge
print(len(data_combined))
balanced_dataset = balance_and_limit_dataset(data_combined, languages, items_per_lang, seed=seed)
for lang in languages:
    lang_articles = [i for i in balanced_dataset if i["language"] == lang]
    print(lang,len(lang_articles))