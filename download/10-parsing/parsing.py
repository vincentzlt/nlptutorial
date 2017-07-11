
# coding: utf-8

# In[13]:


from pprint import pprint
from collections import defaultdict
import math


# In[11]:


f_grammar="../../test/08-grammar.txt"
f_input="../../test/08-input.txt"


# In[9]:


nonterms=[]
preterms=defaultdict(list)
for line in open(f_grammar,"r",encoding="utf-8"):
    num_tab=line.count("\t")
    if num_tab==2:
        lhs,rhs,prob=line.split("\t")
        prob=float(prob)
        if " " in rhs:
            rhs1,rhs2=rhs.split(" ")
            nonterms.append((lhs,rhs1,rhs2,prob))
        else:
            preterms[rhs].append((lhs,prob))


# In[12]:


nonterms
preterms


# In[26]:


best_scores=defaultdict(lambda:-1000000000000.0)
best_edges={}
for line in open(f_input,"r",encoding="utf-8"):
    line=line.split()
    for idx, word in enumerate(line):
        l_idx=idx
        r_idx=idx+1
        for lhs, prob in preterms[word]:
            best_scores[(lhs,l_idx,r_idx)]=math.log(prob)
    for j in range(2,len(line)):
        for i in reversed(range(j-2)):
            for k in range(i+1,j-1):
                for sym, lsym, rsym, logprob in nonterms:
                    logprob=math.log(logprob)
                    if best_scores[(lsym,i,k)]>-10e10 and best_scores[(rsym,k,j)]>-10e10:
                        my_lp=best_scores[(lsym,i,k)]+best_scores[(rsym,k,j)]+log_prob
                        if my_lp>best_scores[(sym,i,j)]:
                            best_scores[(sym,i,j)]=my_lp
                            best_edges[(sym,i,j)]=((lsym,i,k),(rsym,k,j))
best_scores
best_edges


# In[24]:


10e10

