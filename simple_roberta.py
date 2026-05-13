import pandas as pd
from GithubDatasetExtraction import get_article
import torch
from transformers import AutoTokenizer, AutoModel
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')

def block_based_embedding(text, model, tokenizer, max_len=512):
    sentences = sent_tokenize(text) #list of full sentences using nltk

    for s in sentences:
        print(s)
        print()

    blocks = []
    current_block = []
    current_count = 0
    max_limit = 512 - 2 # XLM-RoBERTa limit is 512, reserve 2 slots for [CLS] and [SEP] tokens

    for sentence in sentences:
        sentence_tokens = tokenizer.encode(sentence, add_special_tokens = False) #without special tokens
        num_tokens = len(sentence_tokens)

        if num_tokens > max_limit:
            sentence_tokens = sentence_tokens[:max_limit]
            num_tokens  = max_limit

        if current_count + num_tokens <= max_limit:
            current_block.append(sentence)
            current_count += num_tokens
        else:
            blocks.append(" ".join(current_block))
            current_block = [sentence]
            current_count = num_tokens
            
    if current_block:
        blocks.append(" ".join(current_block))

    block_embeddings = []
    
    for block in blocks:
        inputs = tokenizer(block, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            
            block_emb = outputs.last_hidden_state.mean(dim=1)
            block_embeddings.append(block_emb)

    
    final_embedding = torch.cat(block_embeddings).mean(dim=0)
    return final_embedding


model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


article_id = 2

article_text = get_article(article_id)
final_embedding = block_based_embedding(article_text, model, tokenizer)


print(f"Vektor za članak {article_id} je spreman!")
print(f"Dimenzija: {final_embedding.shape}")
print(f"{final_embedding}")
