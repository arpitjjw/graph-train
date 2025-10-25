from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import numpy as np
import torch

from numpy import dot
from numpy.linalg import norm

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v1',device='cuda:0')

def embed(text):
    sentences = sent_tokenize(text)
    embeddings = model.encode(sentences)

    del sentences

    mean=np.mean(embeddings, axis=0)
    del embeddings
    torch.cuda.empty_cache()
    return mean

def cos_sim(a,b):
   return dot(a, b)/(norm(a)*norm(b))

if __name__ == "__main__":
    text1='That is a happy person. That is a very happy person'
    text2='That is a happy person.'
    embed1= embed(text1)
    embed2= embed(text2)
    print(cos_sim(embed1,embed2))
