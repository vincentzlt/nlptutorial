#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys,math
from collections import defaultdict


def preprocess_corpus(f_name):
    """
    input:  a file name
    ouput:  a list of lines in the corpus
            each line is a list"""

    try:
        with open(f_name, "r", encoding="UTF-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print("file not found. try again")
    except:
        print(sys.exc_info())

    output = [(i + " </s>").split() for i in lines]

    return output


def calc_w_uni_P(w, corpus_list,g,vocab_number):
    sum_token = sum([sum(i) for i in corpus_list])
    sum_w = sum([sum(filter(lambda w_:w_=w, i)) for i in corpus_list])
    output=g*sum_w/float(sum_token)+(1-g)/float(n)
    return output

def calc_entropy(input_list,corpus_list):
    sum_token=sum([sum(i) for i in corpus_list])


if __name__ == "__main__":
    if sys.argv[1]:
        preprocess_corpus(sys.argv[1])
    else:
        print("please enter a file name.")
