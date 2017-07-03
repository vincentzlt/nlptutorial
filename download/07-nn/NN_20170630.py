
# coding: utf-8

# In[1]:


import numpy as np
from pprint import pprint
from time import time

# In[2]:


def gen_ids(f_names):
    words=[]
    labels=[]
    for line in open(f_names,"r",encoding="utf-8"):
        label,sentence=line.split("\t")
        labels.append(int(label))
        for w in sentence.split():
            words.append(w)
    words=set(words)
    w_ids={}
    
    for w in words:
        w_ids[w]=len(w_ids)
    sentences=[]
    for line in open(f_names,"r",encoding="utf-8"):
        label,sentence=line.split("\t")
        sent=np.zeros(len(w_ids))
        for w in sentence.split():
            sent[w_ids[w]]+=1
        sentences.append(sent)
    return w_ids,sentences,labels


# In[3]:


def gen_sent_phi(sentence,w_ids):
    sent=np.zeros(len(w_ids))
    for w in sentence.split():
        if w in w_ids:
            sent[w_ids[w]]+=1
    return sent


# In[90]:


class NN():
    def __init__(self, input_arrays, derr_out, network_dims):
        """=========input==============
        input_arrays:
            list, contains input arrays
        derr_out:
            list of np.array
        network_dims:
            list, contains dims of each layers
        """
        assert len(derr_out) == network_dims[-1]

        self.network_len = len(network_dims)
        self.input_len = len(input_arrays)

        self.input_arrays = input_arrays
        self.derr_out = derr_out
        self.network_dims = network_dims

        dim_input_arrays = [len(array) for array in input_arrays]
        input_weights = list(np.random.rand(dim_prev, network_dims[0])
                         for dim_prev in dim_input_arrays)
        dinput_weights=list(np.zeros((dim_prev, network_dims[0]))
                         for dim_prev in dim_input_arrays)
        
        self.weights = []
        self.dinput_arrays=[None]*self.input_len
        self.neurons_net = []
        self.neurons_bias_weight = []
        self.neurons_out = []
        self.neurons_dout = []
        self.neurons_dnet = []
        self.dweights=[]

        for i in range(self.network_len):
            if i == 0:
                self.weights.append(input_weights)
                self.dweights.append(dinput_weights)
                dim_prev = network_dims[i]

            else:
                weight = np.random.rand(dim_prev, network_dims[i])
                self.weights.append(weight)
                self.dweights.append(np.zeros((dim_prev, network_dims[i])))
                dim_prev = network_dims[i]
            self.neurons_bias_weight.append(np.array([1]))
    
    def set_input_array(self,input_arrays):
        self.input_arrays=input_arrays
    def set_labels(self,labels):
        assert self.input_len==len(labels)
        self.labels=labels
    
    def ff_one(self):
        for i in range(self.network_len):
            self.neurons_out.append(np.zeros(self.network_dims[i]))
            self.neurons_net.append(np.zeros(self.network_dims[i]))
            self.neurons_dnet.append(np.zeros(self.network_dims[i]))
            self.neurons_dout.append(np.zeros(self.network_dims[i]))
            
            if i == 0:
                # w_h*h_t-1+w_x*x+b
                self.neurons_net[i] = sum(
                    np.dot(input_array, weight)
                    for input_array,weight in zip(self.input_arrays
                    ,self.weights[i])) + self.neurons_bias_weight[i]
                self.neurons_out[i]=np.tanh(self.neurons_net[i])
            else:
                self.neurons_net[i]=np.dot(self.neurons_out[i-1],self.weights[i])+self.neurons_bias_weight[i]
                self.neurons_out[i]=np.tanh(self.neurons_net[i])

    def bk_one(self):
        for i in reversed(range(self.network_len)):
            
            if i==self.network_len-1:
                self.neurons_dout[i]=self.derr_out
                self.neurons_dnet[i]=1-self.neurons_net[i]**2
                self.dweights[i]=np.outer(self.neurons_out[i-1],self.neurons_net[i])
            elif i==0:
                self.neurons_dout[i]=np.dot(self.neurons_net[i+1],self.weights[i+1].T)
                self.neurons_dnet[i]=1-self.neurons_net[i]**2
                for j in range(self.input_len):
                    self.dweights[i][j]=np.outer(self.input_arrays[j],self.neurons_net[i])
                    self.dinput_arrays[j]=np.dot(self.neurons_net[i],self.weights[i][j].T)
            else:
                self.neurons_dout[i]=np.dot(self.neurons_net[i+1],self.weights[i+1].T)
                self.neurons_dnet[i]=1-self.neurons_net[i]**2
                self.dweights[i]=np.outer(self.neurons_out[i-1],self.neurons_net[i])
            

    def update_weight(self,lrate=0.01):
        for i in range(self.network_len):
            if i==0:
                for j in range(self.input_len):
                    self.weights[i][j]+=lrate*self.dweights[i][j]
            else:
                self.weights[i]+=lrate*self.dweights[i]
            self.neurons_bias_weight+=lrate*self.neurons_dnet[i]


# In[91]:


w_ids,arrays,labels=gen_ids("../../test/03-train-input.txt")


# In[92]:


nn=NN([arrays[0]],[np.array([labels[0]])],[2,1])


# In[93]:


for idx,(array,label) in enumerate(zip(arrays,labels)):
    if idx==100:
        print(label,nn.neurons_out[-1])
    nn.set_input_array((array,))
    nn.set_labels(np.array([label]))
    nn.ff_one()
    nn.bk_one()
    nn.update_weight()


# In[ ]:





# In[251]:


'''test_file="../../data/titles-en-test.word"
with open("my-answer","w",encoding="utf-8") as f:
    for sent in open(test_file,"r",encoding="utf-8"):
        f.write("{}\t{}".format(nn.predict(sent),sent))
'''

# In[ ]:





# In[ ]:




