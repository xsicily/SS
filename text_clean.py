#------------------- Text processing-------------------------#
#input: sentences
# Text processing
import nltk
import string
import math 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.porter import PorterStemmer
import re
import string

def remove_whitespace(text): #output-sentence
    return text.strip()

def compound_split(text):#output-sentence
    # change compound words to separate words ie. 'conditional-statements' -> 'conditional', 'statements'
    regex = re.compile("[-_'.]")
    trimmed = regex.sub(' ', text)
    return trimmed

# remove punctuations [!”#$%&’()*+,-./:;<=>?@[\]^_`{|}~]
def remove_punct(text):#output-sentence
    exclude = set(string.punctuation)
    s_no_punct = ''.join(ch for ch in text if ch not in exclude)
    return s_no_punct

# remove non-alpha
def remove_nonalpha(text):#output-sentence
    regex = re.compile('[^a-zA-Z]')
    nonAlphaRemoved = regex.sub(' ', text)
    return nonAlphaRemoved

# tokenize
def tokenizer(sentence):#output-words
    return word_tokenize(sentence.lower())

# remove len(word)<=1
def alpha_filter(wordslist):
    words_list = [word for word in wordslist if len(set(word)) > 1]
    return words_list

# remove stopwords
'''set(stopwords.words('english')) from nltk
{‘ourselves’, ‘hers’, ‘between’, ‘yourself’, ‘but’, ‘again’, ‘there’, ‘about’, 
‘once’, ‘during’, ‘out’, ‘very’, ‘having’, ‘with’, ‘they’, ‘own’, ‘an’, ‘be’, 
‘some’, ‘for’, ‘do’, ‘its’, ‘yours’, ‘such’, ‘into’, ‘of’, ‘most’, ‘itself’, 
‘other’, ‘off’, ‘is’, ‘s’, ‘am’, ‘or’, ‘who’, ‘as’, ‘from’, ‘him’, ‘each’, 
‘the’, ‘themselves’, ‘until’, ‘below’, ‘are’, ‘we’, ‘these’, ‘your’, ‘his’, 
‘through’, ‘don’, ‘nor’, ‘me’, ‘were’, ‘her’, ‘more’, ‘himself’, ‘this’, ‘down’, 
‘should’, ‘our’, ‘their’, ‘while’, ‘above’, ‘both’, ‘up’, ‘to’, ‘ours’, ‘had’, 
‘she’, ‘all’, ‘no’, ‘when’, ‘at’, ‘any’, ‘before’, ‘them’, ‘same’, ‘and’, ‘been’, 
‘have’, ‘in’, ‘will’, ‘on’, ‘does’, ‘yourselves’, ‘then’, ‘that’, ‘because’, ‘what’, 
‘over’, ‘why’, ‘so’, ‘can’, ‘did’, ‘not’, ‘now’, ‘under’, ‘he’, ‘you’, ‘herself’, ‘has’, 
‘just’, ‘where’, ‘too’, ‘only’, ‘myself’, ‘which’, ‘those’, ‘i’, ‘after’, ‘few’, ‘whom’, 
‘t’, ‘being’, ‘if’, ‘theirs’, ‘my’, ‘against’, ‘a’, ‘by’, ‘doing’, ‘it’, ‘how’, 
‘further’, ‘was’, ‘here’, ‘than’}'''

def remove_stopwords(word_list):#output-words
    english_stop_words = set(stopwords.words('english'))
    processed_word_list = []
    for word in word_list:
        if word not in english_stop_words:
            processed_word_list.append(word)
    return processed_word_list

def stemmer(word_list):#output-words
    #PorterStemmer
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in word_list]
    return stemmed

def listAsString(list_string):
    one_string = ''
    for idx, word in enumerate(list_string):
        if idx == len(list_string) - 1:
            one_string = one_string + word
        else:
            temp_string = word + ' '
            one_string = one_string + temp_string 
    return one_string

if __name__ == "__main__":
    text = "I like to eat one-half apple."
    result = text_clean(text)
    print(result)
