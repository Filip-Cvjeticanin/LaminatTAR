import pandas as pd
from GithubDatasetExtraction import get_article
import torch
from transformers import AutoTokenizer, AutoModel


model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


article_id = 2
clanak_tekst = get_article(article_id)

inputs = tokenizer(clanak_tekst, return_tensors="pt", truncation=True, max_length=512)

with torch.no_grad():
    outputs = model(**inputs)


embedding = outputs.last_hidden_state[0, 0, :]

print(f"Vektor za članak {article_id} je spreman!")
print(f"Dimenzija: {embedding.shape}")
print(f"{embedding}")
