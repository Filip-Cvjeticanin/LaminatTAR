from loadData import load_dataset
from loadData import count_articles_in_dataset
from loadData import load_test_dataset
from loadData import limit_dataset

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
