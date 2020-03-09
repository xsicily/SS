#------------------------------------------ Sentence Similarity----------------------------------------------#
##############################################################################################################
#-------include all "means like" words by quering Datamuse--------#
# we use stopwds_filtered -list of word lists that is defined above as seed for wordnet
# Implementation: parameters: ml. Number of output results: 1000.

import datamuse
import requests
import json
import csv
import numpy as np
import text_clean
from scipy.stats import pearsonr
import pickle
import get_ratings

# define datamuse search
def datamuse_search(query, num_results, params):
    words = []
    for params in params:
        url = 'https://api.datamuse.com/%s=%s&max=%d' % (params, query, num_results)
        response = requests.get(url)
        results = response.text
        try:
            dataform = str(results).strip("'<>() ").replace('\'', '\"')
            results = json.loads(dataform)
        except:
            print(repr(results))
        words.extend([a['word'] for a in results])
    return words

def new_text_clean(expand_list):
    new_text = text_clean.listAsString(expand_list)
    temp1 = text_clean.remove_whitespace(new_text)
    temp2 = text_clean.compound_split(temp1)
    temp3 = text_clean.tokenizer(temp2)
    temp4 = text_clean.alpha_filter(temp3)
    #temp4 = text_clean.remove_stopwords(temp3)
    #temp5 = text_clean.stemmer(temp4)
    return temp4

def get_morewds(word_list,num_results, params):
    expand_list = []
    new_list = []
    for ss in word_list:      
        words_datamuse = datamuse_search(ss, num_results,params)
        new_list.extend(words_datamuse)
    expand_list.extend(new_list)
    expand_list.extend(word_list)
    return expand_list

def overlap_approxi(hhjsim,diff_wds_list_1, diff_wds_list_2):
    num_intersect_of_diff = int(hhjsim*(len(diff_wds_list_1) + len(diff_wds_list_2)) / (1 + hhjsim))
    return num_intersect_of_diff

def Jsim(wdslist_1, wdslist_2):
    intersection = set(wdslist_1).intersection(set(wdslist_2))
    union = set(wdslist_1).union(set(wdslist_2))
    return len(intersection)/len(union)

def get_djsim(wds_bag_1, wds_bag_2):
    intersection = set(wds_bag_1).intersection(set(wds_bag_2))
    diff_wds_list_1 = set(wds_bag_1) - intersection
    diff_wds_list_2 = set(wds_bag_2) - intersection
    if diff_wds_list_1 == diff_wds_list_2:
        expand_jsim = 1.0
    else:
        expand_list_1 = get_morewds(diff_wds_list_1, 1000, ['words?ml'])
        expand_list_2 = get_morewds(diff_wds_list_2, 1000, ['words?ml'])
        #expand_list_1 = new_text_clean(expand_list_1)
        #expand_list_2 = new_text_clean(expand_list_2)
        hhjsim = Jsim(expand_list_1,expand_list_2)
        num_intersect_of_diff = overlap_approxi(hhjsim,diff_wds_list_1, diff_wds_list_2)###
        expand_jsim = (len(intersection) + num_intersect_of_diff) / (len(set(wds_bag_1)) + len(set(wds_bag_2)) - len(intersection) - num_intersect_of_diff)
    print(expand_jsim)
    return expand_jsim

    # text processing
def snippet_clean(snippet):
    temp1 = text_clean.remove_whitespace(snippet)
    temp2 = text_clean.compound_split(temp1)
    temp3 = text_clean.remove_punct(temp2)
    temp4 = text_clean.remove_nonalpha(temp3)
    temp5 = text_clean.tokenizer(temp4)
    temp6 = text_clean.alpha_filter(temp5)
    temp7 = text_clean.remove_stopwords(temp6)
    return temp7


def snippets_combine(snippets_sp):
    words_bag = []
    for pair_no in range(len(snippets_sp)):
        temp_bag = []
        for list_no in range(len(snippets_sp[pair_no])):
            temp = []
            for snippet in snippets_sp[pair_no][list_no]:
                snippet_cleaned = snippet_clean(snippet)
                temp.extend(snippet_cleaned)
            temp_bag.append(temp)
        words_bag.append(temp_bag)
    return words_bag

def Djsim_snippets(snippets_sp):
    processed_snippets = snippets_combine(snippets_sp)
    sim_list = []
    for pair_no in range(len(processed_snippets)):
        wds_bag_1 = processed_snippets[pair_no][0]
        wds_bag_2 = processed_snippets[pair_no][1]
        sim = get_djsim(wds_bag_1, wds_bag_2)
        sim_list.append(sim)
    return np.array(sim_list)

if __name__ == "__main__":
    # load full snippets for 65 sentence pairs
    with open("./dataset/snippets_data_head_g_full.txt", "rb") as f:
        snippets_sp = pickle.load(f)

    # Calculate similarity
    sim = Djsim_snippets(snippets_sp)

    # Save the similarity results
    with open("./dataset/snippets_head_djsim_g.txt", "wb") as fp:
        pickle.dump(sim, fp)

    # Get human judge values
    filename = "./dataset/head.csv"
    ratings = get_ratings.rating(filename)

    # Calculate pearson coefficient
    r = pearsonr(ratings,sim)
    print("r =" + str(r))