#------------------------------------------ Sentence Similarity----------------------------------------------#
##############################################################################################################
# This script is used to calculate Word2Vec similarity of sentence pairs from the targeted dataset

import gensim
import os
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import pickle
import text_clean
import get_ratings

def sentence_preprocess(sentence):
    #tokenize
    tokens = word_tokenize(sentence)
    #stop-word removal
    stop_words = set(stopwords.words('english'))
    filtered_sentence = [w for w in tokens if not w in stop_words]
    return filtered_sentence

def model_sentence_similarity(s1, s2, model):
    s1 = re.sub('[^a-zA-Z]', ' ', s1 )
    s1 = re.sub(r'\s+', ' ', s1 )
    s2 = re.sub('[^a-zA-Z]', ' ', s2 )
    s2 = re.sub(r'\s+', ' ', s2 )
    #s1 = s1.strip().split()
    s1 = sentence_preprocess(s1)
    #print(s1)
    #s2 = s2.strip().split()
    s2 = sentence_preprocess(s2)
    vecSet_1 = []
    for w in s1:
        try:
          vec = model[w]  
        except KeyError:
            print("not in vocabulary")
            vec = 0
        vecSet_1.append(vec)
    vecSet_2 = []
    for w in s2:
        try:
          vec = model[w]  
        except KeyError:
            print("not in vocabulary")
            vec = 0
        vecSet_2.append(vec)
    vec_1 = np.mean(vecSet_1, axis = 0)
    vec_2 = np.mean(vecSet_2, axis = 0)
    return cosine_similarity(vec_1.reshape(1, -1), vec_2.reshape(1, -1))


def sentence_similarity_dataset_model(Dataset, model):
    sim_list = []
    for i, sentence_pair in enumerate(Dataset):
        print('%d/%d th pair' % (i + 1, len(Dataset)))
        #print(sentence_pair[0])
        #print(sentence_pair[1])
        similarity = model_sentence_similarity(sentence_pair[0], sentence_pair[1], model)
        print(similarity)
        sim_list.append(similarity)
    return sim_list

def initial_clean(text):
    temp1 = text_clean.remove_whitespace(text)
    temp2 = text_clean.compound_split(temp1)
    temp3 = text_clean.remove_punct(temp2)
    temp4 = text_clean.remove_nonalpha(temp3)
    return temp4

def vecsim_snippets(snippets_sp):
    vecsim_list = []
    for pair_no in range(len(snippets_sp)):
        snippetsList_1 = snippets_sp[pair_no][0]
        snippetsList_2 = snippets_sp[pair_no][1]
        sim_list = []
        for no_1 in range(len(snippetsList_1)):
            temp_list = []
            snippet_1 = snippetsList_1[no_1]
            snippet_1 = initial_clean(snippet_1)
            for no_2 in range(len(snippetsList_2)):
                snippet_2 = snippetsList_2[no_2]
                snippet_2 = initial_clean(snippet_2)
                sssim = model_sentence_similarity(snippet_1, snippet_2, model)
                print(sssim)
                temp_list.append(sssim)
            sim_list.append(max(temp_list))
        vecsim_list.append(max(sim_list))
    #
    sim = []
    for i in range(len(vecsim_list)):
        temp = vecsim_list[i][0][0]
        sim.extend([temp])
    return np.array(sim)

if __name__ == "__main__":
    # Load the pretrained model
    model= gensim.models.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)
    
    with open("./dataset/snippets_data_ha_g_full.txt", "rb") as f:
        snippets_sp = pickle.load(f)
    
    sim = vecsim_snippets(snippets_sp)
    # Save the similarity results
    with open("./dataset/snippets_ha_word2vec_g.txt", "wb") as fp:
        pickle.dump(sim, fp)
    
    # Get human judge values
    filename = "./dataset/human_activity.csv"
    ratings = get_ratings.rating(filename)
    for i in sim:
        print(i) 
    # Calculate pearson coefficient
    r = pearsonr(ratings,sim)
    print("r =" + str(r))
    print(np.mean(sim))
    
  
    