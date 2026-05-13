from GithubDatasetExtraction import get_article, get_number_of_articles
import pandas as pd

print(get_article(0))


filereader = pd.read_csv('test.csv')

num_articles = get_number_of_articles(filereader)
print("Number of articles:", num_articles)