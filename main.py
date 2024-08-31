# -*- coding: utf-8 -*-
"""
Created on Thu May 16 19:26:32 2024

@author: zidan
"""
import nltk
import lxml.html.clean
from bs4 import BeautifulSoup
import string
import numpy 
import spacy
from nltk.corpus import stopwords



def preprocessing(text):
    stop_words = stopwords.words('spanish')
    
    #obtaining table of associations (the dictionary) to further deleting punctuation
    table = str.maketrans('', '', string.punctuation + '¿' + '¡')
    
    #clean text (with out tags)
    cleantext = BeautifulSoup(text, "lxml").text
    
    #tokenezation
    tokens = nltk.word_tokenize(cleantext)
    tokens = [w.lower() for w in tokens]
    
    #eliminate punctuation symbols
    tokens = [w.translate(table) for w in tokens]
    
    #eliminate all words, containing non-letter characters such '124', 'f344', 'fdfdfdf3g' etc.
    tokens = [word for word in tokens if word.isalpha()]
    
    #eliminate stop-words
    tokens = [w for w in tokens if not w in stop_words]
    
    #eliminating empty elements:
    tokens = [i for i in tokens if i]
    
    #lemmatizing text:
    lem_text = []
    lem_text = lemmatize_text(tokens, table)          
    # f = open('lemmas.txt', 'w', encoding='UTF-8')
    # for i in range (0, len(lem_text)): 
    #         f.write(str(lem_text[i]))
    #         f.write('\n')
    # f.close()
    return lem_text


def lemmatize_text(tokens, table):     #getting lemmatized text
    nlp = spacy.load('es_core_news_md')    
    document = nlp(str(tokens))
    lemmas = []
    # Create a lemmatized version of the original text file
    for token in document:
        # Get the lemma for each token
        lemmas.append(token.lemma_.lower())
    lemmas = [w.translate(table) for w in lemmas]     
    lemmas = [w for w in lemmas if w and w != ' ']   

    for i in range(0, len(lemmas)):
        for j in range(0, len(lemmas[i])):
            if lemmas[i][j] == ' ':
                lemmas[i] = lemmas[i][0:j]
                break
            else:
                continue
    return lemmas


#obtaining the vocabulary referred to full text
def get_full_vocab(lem_text):
    #creating vocabulary
    vocab_list = list(set(lem_text))
    vocab_list.sort()
    print("Number of words in the vocabulary list: ", len(vocab_list))
    print("Total number of lemmas: ", len(lem_text))
    vocab_dict = dict.fromkeys(vocab_list)
    i = 0
    for k in vocab_dict.keys():
        vocab_dict[k] = i
        i+=1
    #print("Количество слов в словаре-диксе: ", len(vocab_dict))
    return vocab_dict


#getting frequancy of a particular word in a lemmatized text
def get_freq(word, lem_text):
    freq = 0
    for i in range(0, len(lem_text)):
        if lem_text[i] == word:
            freq += 1
    return freq

def get_chapter_freqs():
    #length_full_text = len(lem_full_text)
    chapter_freqs = dict.fromkeys(vocab_dict.keys())
    for w in vocab_dict:
        chapter_freqs[w] = get_freq(w, lem_chapter) #/ length_full_text
    return chapter_freqs   
    


def get_back_dist():
    length_full_text = len(lem_full_text)
    back_dist = dict.fromkeys(vocab_dict.keys())
    for w in vocab_dict:
        back_dist[w] = get_freq(w, lem_full_text) / length_full_text
    
        
    
    # tmp_back_dist = dict.fromkeys(vocab_dict.keys())
    # for w in vocab_dict:
    #     tmp_back_dist[w] = get_freq(w, lem_full_text) / length_full_text
    # tmp_back_dist = sorted(tmp_back_dist.items(), key = lambda item: item[1], reverse = True)
    # tmp_back_dist = numpy.array(tmp_back_dist)
    # f = open('background.txt', 'w', encoding='UTF-8')
    # for i in range(0, len(tmp_back_dist)):
    #     f.write(str(tmp_back_dist[i][0]) + ': ' + str(round(float((tmp_back_dist[i][1])),4)) + '\n')
    # f.close()   

    return back_dist


def initialize_topic_dist():
    length_full_text = len(lem_full_text) 
    topic_dist = dict.fromkeys(vocab_dict.keys())
    for w in vocab_dict:
        topic_dist[w] = 1 / length_full_text
    return topic_dist



def estimate_likelihood(chapter_freqs, topic_dist, back_dist, pd, pb):
    L = 0 
    for w in chapter_freqs:
        #prob of appearing of the word (via any of our 2 distributions):
        p_cur_word = pd * topic_dist[w] + pb * back_dist[w]
        L = L + chapter_freqs[w] * numpy.log2(p_cur_word)
    print('Current likelyhood: ' + str(L))
    return L


#executing an iterative algorithm (N - number of iterations)
def do_EM_alg(lem_chapter, vocab_dict, N):
    pd = 0.5 #probability of choosing a particular distribution 
    pb = 0.5
    L_old = -1000000 #inicialization of "old" likelyhood
    old_topic_dist = dict.fromkeys(vocab_dict.keys()) #inicialization of "old" topic distr
    for w in old_topic_dist:
        old_topic_dist[w] = 0
    
    #getting background distribution, based on vocab of whole text:
    back_dist = get_back_dist()
    
    #getting initial values for topic distribution:
    topic_dist = initialize_topic_dist()
    
    #getting frequencies for chapter, based on whole vocab:
    chapter_freqs = get_chapter_freqs()
    
    #P(z = 1 | w)
    p_z_1_w = dict.fromkeys(vocab_dict.keys())
    #P(z = 0 | w)
    p_z_0_w = dict.fromkeys(vocab_dict.keys())
    
    for iteration in range(0, N):
        #E-step (for every word):
        for word in vocab_dict:
            p_z_0_w[word] = (pd * topic_dist[word]) / (pd * topic_dist[word] + pb * back_dist[word]) 
            p_z_1_w[word] = 1 - p_z_0_w[word]
        
        #M-step (for every word):
        sum = 0
        for word in chapter_freqs:
            sum = sum + chapter_freqs[word] * p_z_0_w[word] # getting denominator
        

        # for k in chapter_freqs:
        #     if (chapter_freqs[k] != 0): 
        #         print(str(k) + ' ' + str(chapter_freqs[k]) + ' ' + str(p_z_0_w[k]))
        # print("SUM: " + str(sum))
        
        for word in vocab_dict:   
             topic_dist[word] = (chapter_freqs[word] * p_z_0_w[word]) / sum
                       
        #estimating likelyhood:
        L = estimate_likelihood(chapter_freqs, topic_dist, back_dist, pd, pb)
        
        #if new likelyhood is worse than old one - we have to stop cycle.
        if L < L_old:
            for w in topic_dist:
                topic_dist[w] = old_topic_dist[w]
            print("Stopped at L = " + str(L_old))
            break
        else: 
            L_old = L
            for w in old_topic_dist:
                old_topic_dist[w] = topic_dist[w]
        
        # L = 0 
        # for i in range(0, len(lem_chapter)):
        #     cur_word = lem_chapter[i] #current word in chapter
        #     #prob of appearing of the word (via any of our 2 distributions):
        #     p_cur_word = pd * topic_dist[cur_word] + pb * back_dist[cur_word]
        #     L += numpy.log2(p_cur_word)
        # print('Current likelyhood: ' + str(L))
        
        
    
    final_dist = sorted(topic_dist.items(), key = lambda item: item[1], reverse = True)
    return numpy.array(final_dist)


NN = 200 #number of iterations
f = open('chapter.htm', 'r', encoding='UTF-8')
chapter = f.read()
    
f = open('e990519_mod.htm', 'r', encoding='UTF-8')
full_text = f.read()
f.close()
    
lem_chapter = preprocessing(chapter)
lem_full_text = preprocessing(full_text)
vocab_dict = get_full_vocab(lem_full_text)
final_topic_dist = do_EM_alg(lem_chapter, vocab_dict, NN)

f = open('topics.txt', 'w', encoding='UTF-8')
for i in range(0, len(final_topic_dist)):
    f.write(str(final_topic_dist[i][0]) + ': ' + str(round(float((final_topic_dist[i][1])),4)) + '\n')
f.close()    







