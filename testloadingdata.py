from loadData import load_dataset
from loadData import count_articles_in_dataset

languages = []
languages.append("en")
languages.append("ge")
languages.append("fr")
languages.append("it")
languages.append("po")
languages.append("ru")
data = load_dataset("./data")
#print(data[0])
print("RADIIII !!!!! :3")
print("en: ", count_articles_in_dataset(data, "en"))
print("ge: ", count_articles_in_dataset(data, "ge"))
print("fr: ", count_articles_in_dataset(data, "fr"))
print("po: ", count_articles_in_dataset(data, "po"))
print("ru: ", count_articles_in_dataset(data, "ru"))
