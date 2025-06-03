import torch
from transformers import BertTokenizer, BertModel, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
from torch.nn.functional import cosine_similarity


# Checking the similarity of 2 sentences
'''
def get_cls_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :]  # CLS

text1 = "Cats are great pets."
text2 = "I have a huge brick"


vec1 = get_cls_embedding(text1)
vec2 = get_cls_embedding(text2)

similarity = cosine_similarity(vec1, vec2)
print("Similarity:", similarity.item())  # Closer to 1 = more similar
'''

# Embedding vectors and shi

'''def get_word_embedding(word):
    tokens = tokenizer(word, return_tensors="pt", add_special_tokens=True)
    with torch.no_grad():
        outputs = model(**tokens)
    input_ids = tokens['input_ids'][0]
    token_strs = tokenizer.convert_ids_to_tokens(input_ids)

    # Ignore CLS and SEP tokens
    word_tokens = token_strs[1:-1]
    word_indices = list(range(1, 1 + len(word_tokens)))

    word_embedding = outputs.last_hidden_state[0, word_indices, :].mean(dim=0, keepdim=True)
    return word_embedding

vec1 = get_word_embedding("uncle")
vec2 = get_word_embedding("aunt")
vec3 = get_word_embedding("horse")
vec4 = get_word_embedding("mare")

csimilarity = cosine_similarity(vec3+vec2-vec1, vec4)
print("Cosine Similarity:", csimilarity.item()) # This give not a very good cosine similairty ~ 0.68, this is because BERT'e embedding are contextual and not absoulte like vec2vec'''

# Contextual Embeddings
'''
def get_word_embedding(word, sentence):
    tokens = tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
    with torch.no_grad():
        outputs = model(**tokens)

    # print(tokens)
    input_ids = tokens['input_ids'][0]
    token_strs = tokenizer.convert_ids_to_tokens(input_ids)

    # print(token_strs)

    # Ignore CLS and SEP tokens
    word_tokens = token_strs[1:-1]
    word_index = word_tokens.index(word)
    word_indices = list(range(1, 1 + len(word_tokens)))

    # Just need the embedding of bank

    word_embedding = outputs.last_hidden_state[0, [word_index], :].mean(dim=0, keepdim=True)
    return word_embedding

a= get_word_embedding("bank", sentence="I deposited cash at the bank")
c=get_word_embedding("bank", sentence="I deposited cash at the bank")
b= get_word_embedding("bank", sentence="The boat was near the river bank")

csimilarity = cosine_similarity(a,b)
print(csimilarity) # These two embedding will not be similar to each other, score ~0.42

csimilarity = cosine_similarity(a,c)
print(csimilarity)
'''

# Clustering analysis
# --------------------------- Varinace explained by 2 principle comonents -> ~70% --------------------------------------- 

import numpy as np
from sklearn.decomposition import PCA

def get_word_embedding(word):
    tokens = tokenizer(word, return_tensors="pt", add_special_tokens=True)
    with torch.no_grad():
        outputs = model(**tokens)
    input_ids = tokens['input_ids'][0]
    token_strs = tokenizer.convert_ids_to_tokens(input_ids)

    # Ignore CLS and SEP tokens
    word_tokens = token_strs[1:-1]
    word_indices = list(range(1, 1 + len(word_tokens)))

    word_embedding = outputs.last_hidden_state[0, word_indices, :].mean(dim=0, keepdim=True)
    return word_embedding

# Get embeddings for multiple words
words = ["uncle", "aunt", "king", "queen", "man", "woman"]
embeddings = []

for word in words:
    vec = get_word_embedding(word).squeeze(0).numpy()  # shape: (hidden_dim,)
    embeddings.append(vec)

vecs = np.stack(embeddings)

pca = PCA(n_components=2)
pc = pca.fit_transform(vecs)
print(pca.explained_variance_ratio_)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.DataFrame(pc, columns=["PC1", "PC2"])
df["word"] = words

plt.figure(figsize=(8, 6))
ax = sns.scatterplot(data=df, x="PC1", y="PC2", hue="word", s=100, palette="tab10", legend=False)

for i in range(len(df)):
    ax.text(df["PC1"][i] + 0.01, df["PC2"][i] + 0.01, df["word"][i], fontsize=12)

plt.tight_layout()
plt.show()