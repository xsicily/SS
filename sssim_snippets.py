#------------------------------------------ Sentence Similarity----------------------------------------------#
##############################################################################################################
# This script is used to calculate Semantic similarity of sentence pairs based on snippets analysis
# Reference:Li, Y., McLean, D., Bandar, Z. A., O'shea, J. D., & Crockett, K. (2006). 
# Source code from: 
# https://github.com/rohanpillai20/Sentence-Similarity-based-on-Semantic-Nets-and-Corpus-Statistics

from scipy.stats import pearsonr
from scipy.stats import spearmanr
import numpy as np
import text_clean
from nltk.tokenize import word_tokenize
import semantic_os as sim
import pickle
import get_ratings


def initial_clean(text):
    temp1 = text_clean.remove_whitespace(text)
    temp2 = text_clean.compound_split(temp1)
    temp3 = text_clean.remove_punct(temp2)
    temp4 = text_clean.remove_nonalpha(temp3)
    return temp4


def SSsim_snippets(snippets_sp):
    SSSim_list = []
    for pair_no in range(len(snippets_sp)):
        snippetsList_1 = snippets_sp[pair_no][0]
        snippetsList_2 = snippets_sp[pair_no][1]
        sssim_list = []
        for no_1 in range(len(snippetsList_1)):
            temp_list = []
            snippet_1 = snippetsList_1[no_1]
            snippet_1 = initial_clean(snippet_1)
            for no_2 in range(len(snippetsList_2)):
                snippet_2 = snippetsList_2[no_2]
                snippet_2 = initial_clean(snippet_2)
                sssim = sim.similarity(snippet_1, snippet_2)
                #sssim = osim.similarity(snippet_1, snippet_2)
                print(sssim)
                temp_list.append(sssim)
            sssim_list.append(max(temp_list))
        SSSim_list.append(max(sssim_list))
    return np.array(SSSim_list)

if __name__ == "__main__":
    # load full snippets for 65 sentence pairs
    with open("./dataset/snippets_data_belief_new_g.txt", "rb") as f:
        snippets_sp = pickle.load(f)

    # Calculate similarity
    sim = SSsim_snippets(snippets_sp)

    # Save the similarity results
    with open("./dataset/snippets_belief_sssim_g.txt", "wb") as fp:
        pickle.dump(sim, fp)
    '''
    for i in sim:
        print(i) 
    # Get human judge values
    filename = "./dataset/head.csv"
    ratings = get_ratings.rating(filename)
    print(np.mean(sim))
    # Calculate pearson coefficient
    r = pearsonr(ratings,sim)
    print("r =" + str(r))
    '''
