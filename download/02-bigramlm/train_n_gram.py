
# coding: utf-8

# In[26]:

import sys, math
from collections import defaultdict
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[27]:

def gen_corpus_list(f_name):
    with open(f_name, "r", encoding="UTF-8") as f:
        corpus_list = [line.split() for line in f.readlines()]
    for line in corpus_list:
        line.append("<s/>")
        line.insert(0, "<s>")
    return corpus_list


# In[28]:

def gen_n_gram_dict(n,corpus_list):
    n_gram_dict=defaultdict(lambda :0)
    
    for line in corpus_list:
        line_list=[]
        for pos in range(n):
            line_list.append(line[pos:len(line)-n+1+pos])
        n_gram_list=list(zip(*line_list))
        for w in n_gram_list:
            n_gram_dict[" ".join(w)]+=1
    return n_gram_dict


# In[29]:

def calc_sum_token(n_gram_dict):
    return sum(n_gram_dict.values())


# In[35]:

def calc_n_gram_model(n, corpus_list):

    n_gram_dict = gen_n_gram_dict(n, corpus_list)
    n_minus_1_gram_dict = gen_n_gram_dict(n-1, corpus_list)

    for n_gram in n_gram_dict:
        n_minus_1_gram = " ".join(n_gram.split()[:-1])
        P_n_gram = n_gram_dict[n_gram] / float(
            n_minus_1_gram_dict[n_minus_1_gram])
        print(n_gram, "\t", P_n_gram)

    return


# In[38]:

if __name__=="__main__":
    sys.argv="self ../../data/wiki-en-train.word 2".split()
    corpus_list=gen_corpus_list(sys.argv[1])
    calc_n_gram_model(int(sys.argv[2]),corpus_list)


# In[ ]:



