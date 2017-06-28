
import numpy as np
from pprint import pprint
import pdir


def gen_ids(f_names):
    words = []
    labels = []
    for line in open(f_names, "r", encoding="utf-8"):
        label, sentence = line.split("\t")
        labels.append(int(label))
        for w in sentence.split():
            words.append(w)
    words = set(words)
    w_ids = {}

    for w in words:
        w_ids[w] = len(w_ids)
    sentences = []
    for line in open(f_names, "r", encoding="utf-8"):
        label, sentence = line.split("\t")
        sent = np.zeros(len(w_ids))
        for w in sentence.split():
            sent[w_ids[w]] += 1
        sentences.append(sent)
    return w_ids, sentences, labels

class NN():
    from pprint import pprint

    def __init__(self, input_list, labels, network_dims):
        assert len(input_list) == len(labels)

        self.input_list = input_list
        self.labels = labels
        self.network_neuron_outputs = []
        self.network_neuron_nets = []
        self.network_neuron_bias=[]

        self.network_neuron_derr_net = []
        self.network_neuron_dnet_weight = []

        self.network_weights = []

        dim_prev = len(input_list[0])
        for i in range(len(network_dims)):
            dim = network_dims[i]
            self.network_weights.append(np.random.rand(dim_prev, dim) / 10)
            self.network_neuron_bias.append(np.random.rand(dim) / 10)
            self.network_neuron_outputs.append(np.zeros((dim)))
            self.network_neuron_nets.append(np.zeros((dim)))
            self.network_neuron_derr_net.append(np.zeros((dim)))
            self.network_neuron_dnet_weight.append(np.zeros((dim)))
            dim_prev = dim

        self.network = (self.network_neuron_outputs, self.network_neuron_nets,
                        self.network_neuron_derr_net,
                        self.network_neuron_dnet_weight)

    def print_network(self):
        for w in self.__dict__:
            if w.startswith("network"):
                print(w)
                pprint(self.__dict__[w])
                print()

    def ff_one(self, input_array):
        self.input_array = input_array
        for idx in range(len(self.network_weights)):
            if idx == 0:
                outputs_prev = input_array
            else:
                outputs_prev = self.network_neuron_outputs[idx - 1]
            self.network_neuron_nets[idx] = np.dot(outputs_prev,
                                                   self.network_weights[idx])+1*self.network_neuron_bias[idx]
            self.network_neuron_outputs[idx] = np.tanh(
                self.network_neuron_nets[idx])

    def bk_one(self, label):
        err = label - self.network_neuron_outputs[-1]
        for i in reversed(range(len(self.network_neuron_outputs))):
            out = self.network_neuron_outputs[i]
            net = self.network_neuron_nets[i]
            if i != 0:
                out_prev = self.network_neuron_outputs[i - 1]
            else:
                out_prev = self.input_array
            w = self.network_weights[i]
            if i == len(self.network_neuron_outputs)-1:
                derr_out = self.network_neuron_outputs[-1]-label
            else:
                derr_out = np.dot(self.network_neuron_derr_net[
                                  i + 1], self.network_weights[i + 1].T)
            dout_net = 1 - out**2
            dnet_weight = np.outer(out_prev, net)

            self.network_neuron_derr_net[i] = derr_out * dout_net

            self.network_neuron_dnet_weight[i] = self.network_neuron_derr_net[i]*dnet_weight
                
    def update_weight(self, lrate=0.01):
        for i in range(len(self.network_weights)):
            self.network_weights[i]-=self.network_neuron_dnet_weight[i]*lrate
            self.network_neuron_bias[i]-=self.network_neuron_derr_net[i]*lrate
    def train(self, epoch=10,lrate=0.01):
        for i in range(epoch):
            for input_array,label in zip(self.input_list,self.labels):
                self.ff_one(input_array)
                self.bk_one(label)
                self.update_weight(lrate=lrate)


if __name__=="__main__":
    w_ids, sentences, labels = gen_ids("../../test/03-train-input.txt")
    nn = NN(sentences, labels, (4, 1))
    nn.train(epoch=100000,lrate=0.5)
    nn.ff_one(nn.input_list[0])
    print(nn.network_neuron_outputs[-1])
    nn.ff_one(nn.input_list[1])
    print(nn.network_neuron_outputs[-1])
    input()
