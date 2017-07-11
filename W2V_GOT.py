from __future__ import absolute_import, division, print_function

# for word encodings
import codecs
#regex for effective searching
import glob
import logging
# to perform concurrency
import multiprocessing
# reading the file
import os
# print it human readible pretty print
import pprint
#regular expression
import re

import nltk
import gensim.models.word2vec as w2v
import sklearn.manifold
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#to visualize the dataset
import seaborn as sns

# process our data
#cleaning our data
nltk.download("punkt")   #pre trained tockenizer and it tokenize the text
nltk.download("stopwords")   #and , the , an ,a  etc they dont matter much and we shall remove these words

#get the books

book_filenames = sorted(glob.glob("/Users/ravs/*.txt"))
print(book_filenames)

#combine the books into one string

corpus_raw = u""  #unicode string and we want to convert it to utf-8 and corpus_raw has all the data now
for book_filename in book_filenames:
    print("Reading '{0}'...".format(book_filename))
    with codecs.open(book_filename, "r", "utf-8") as book_file:
        corpus_raw += book_file.read()
    print("Corpus is now {0} characters long".format(len(corpus_raw)))
    print()



#splitting the corpus into sentences
#punkt is loaded into tokenizer

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
raw_sentences = tokenizer.tokenize(corpus_raw)


#convert into a list of words
#rtemove unnnecessary,, split into words, no hyphens
#list of words

def sentence_to_wordlist(raw):
    clean = re.sub("[^a-zA-Z]"," ", raw)
    words = clean.split()
    return words

#sentence where each word is tokenized
sentences = []
for raw_sentence in raw_sentences:
    if len(raw_sentence) > 0:
        sentences.append(sentence_to_wordlist(raw_sentence))


print(raw_sentences[5])
print(sentence_to_wordlist(raw_sentences[5]))

token_count = sum([len(sentence) for sentence in sentences])
print("The book corpus contains {0:,} tokens".format(token_count))

#trainng w2v
#ONCE we have vectors
#step 3 - build model
#3 main tasks that vectors help with
#DISTANCE, SIMILARITY, RANKING

# Dimensionality of the resulting word vectors.
#more dimensions, more computationally expensive to train
#but also more accurate
#more dimensions = more generalized
num_features = 300
# Minimum word count threshold.
min_word_count = 3

# Number of threads to run in parallel.
#more workers, faster we train
num_workers = multiprocessing.cpu_count()

# Context window length.
context_size = 7

# Downsample setting for frequent words.
#0 - 1e-5 is good for this
downsampling = 1e-3

# Seed for the RNG, to make the results reproducible.
#random number generator
#deterministic, good for debugging
seed = 1


thrones2vec = w2v.Word2Vec(
    sg=1,
    seed=seed,
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)


thrones2vec.build_vocab(sentences)

print("Word2Vec vocabulary length:", len(thrones2vec.wv.vocab))

#start training

thrones2vec.train(sentences, total_examples=thrones2vec.corpus_count, epochs=thrones2vec.iter)

#save the model

if not os.path.exists("trained"):
    os.makedirs("trained")


thrones2vec.save(os.path.join("trained", "thrones2vec.w2v"))


# exploring the training model
thrones2vec = w2v.Word2Vec.load(os.path.join("trained", "thrones2vec.w2v"))


print(thrones2vec.most_similar("weak"))
#print(thrones2vec.most_similar(""))
#print(thrones2vec.most_similar("direwolf"))


#Linear similarities : analogies
def nearest_similarity_cosmul(start1, end1, end2):
    similarities = thrones2vec.most_similar_cosmul(
        positive=[end2, start1],
        negative=[end1]
    )
    start2 = similarities[0][0]
    print("{start1} is related to {end1}, as {start2} is related to {end2}".format(**locals()))
    return start2


nearest_similarity_cosmul("strong", "men", "women")
#nearest_similarity_cosmul("Jaime", "sword", "wine")
#nearest_similarity_cosmul("Arya", "Nymeria", "dragons")