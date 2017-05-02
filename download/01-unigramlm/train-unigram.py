#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
from collections import defaultdict

def preprocess_corpus(f_name):
    """
    input:  a file name
    ouput:  a list of lines in the corpus
            each line is a list"""
    
    try:
        with open(f_name,"r",encoding="UTF-8") as f:
            lines=f.readlines()
    except FileNotFoundError:
        print("file not found. try again")
    except:
        print(sys.exc_info())
    
    output=[(i+" </s>").split() for i in lines]

    return output


if __name__=="__main__":
    if sys.argv[1]:
        preprocess_corpus(sys.argv[1])
    else:
        print("please enter a file name.")


    


