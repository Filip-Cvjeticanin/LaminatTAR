from load_data import load_dataset
from load_data import count_articles_in_dataset
from load_data import load_test_dataset
from load_data import limit_dataset
from load_data import load_synthetic_dataset
from load_data import merge_real_and_synthetic

languages = []
languages.append("en")
languages.append("ge")
languages.append("fr")
languages.append("it")
languages.append("po")
languages.append("ru")
data = load_dataset("./data", languages)
data_dev = load_dataset("./data", languages, "dev")
print(data[0])
print(data_dev[0])
print("RADIIII !!!!! :3")
print("en: ", count_articles_in_dataset(data, "en"))
print("ge: ", count_articles_in_dataset(data, "ge"))
print("fr: ", count_articles_in_dataset(data, "fr"))
print("po: ", count_articles_in_dataset(data, "po"))
print("ru: ", count_articles_in_dataset(data, "ru"))

data_test = load_test_dataset("./data", languages)
print(data_test[1])

data_limited = limit_dataset(dataset=data, languages=["en"], max_per_lang=100)
data_limited2 = limit_dataset(dataset=data, languages=["en", "fr"], max_per_lang=50)



synthetic = load_synthetic_dataset("./synthetic data")
print(len(synthetic))

data_combine = merge_real_and_synthetic(data, synthetic)


