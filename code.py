############################################################
# Imports
############################################################

# Include your imports here, if any are used.
import collections
from collections import deque
import email as emails
from email import message_from_file
from email import iterators
from email import charset
import math as mt
import heapq
import os
############################################################
# Section 1: Spam Filter
############################################################

def load_tokens(email_path):
    # source: https://www.programcreek.com/python/example/2579/email.message_from_file
    try:
        with open(email_path,"r",encoding='utf-8') as file:
            message=message_from_file(file)
    except:
        print('file not found')

    mess=iterators.body_line_iterator(message)
    email_message=deque()
    for i in mess:
        email_message.append(i.split())
    email_t=[]
    while email_message:
        email_t.extend(email_message.popleft())
    return(email_t)

def log_probs(email_paths, smoothing):
    word_token = collections.OrderedDict() #dict()
    for path in email_paths:
        tokens = load_tokens(path)
        #word count
        for element in tokens:
            word_token[element] = word_token.get(element,0) + 1
    total_count = 0
    vocab_length = 0
    for i in word_token:
        total_count += word_token[i]
    vocab_length=len(word_token)
    total_count=total_count*1.0
    vocab_length=vocab_length*1.0
    # store log values for each word
    d=total_count + (smoothing*(vocab_length+1.0))
    for key in word_token.keys():
        word_token[key] = mt.log((word_token[key]+smoothing)/d)

    word_token["<UNK>"] = mt.log((smoothing/(total_count + (smoothing*(vocab_length+1.0)))))
    return word_token

class SpamFilter(object):

    def __init__(self, spam_dir, ham_dir, smoothing):
        spam_paths=deque()
        for r, d, f in os.walk(spam_dir):
            for file in f:
                spam_paths.append(os.path.join(r, file))
        spam_files=len(spam_paths)*1.0
        ham_paths=deque()
        for r, d, f in os.walk(ham_dir):
            for file in f:
                ham_paths.append(os.path.join(r, file))
        ham_files=len(ham_paths)*1.0
        self.spam_log=log_probs(spam_paths, smoothing)
        self.ham_log=log_probs(ham_paths, smoothing)
        log_d=mt.log(spam_files+ham_files)
        self.spam_prob = mt.log(spam_files)-log_d
        self.spam_prob_n = mt.log(ham_files)-log_d


    def is_spam(self, email_path):
        train_tokens = load_tokens(email_path)
        spam_prob=self.spam_prob
        spam_prob_n=self.spam_prob_n
        for word in train_tokens:
            if word in self.spam_log.keys():
                spam_prob+=self.spam_log[word]
            else:
                spam_prob+=self.spam_log['<UNK>']
            if word in self.ham_log.keys():
                spam_prob_n+=self.ham_log[word]
            else:
                spam_prob_n+=self.ham_log['<UNK>']
        if spam_prob>spam_prob_n:
            return True
        else:
            return False

    def most_indicative_spam(self, n):
        spam_ind=deque()
        for i in self.spam_log:
            if i in self.ham_log.keys():
                spam_ind.append(((self.spam_log[i] - mt.log(mt.exp(self.spam_log[i])*mt.exp(self.spam_prob) +
                                                            mt.exp(self.ham_log[i])*mt.exp(self.spam_prob_n))),i))
        ind_word = heapq.nlargest(n,spam_ind)
        return [word for (num,word) in ind_word]

    def most_indicative_ham(self, n):
        ham_ind=deque()
        for i in self.ham_log:
            if i in self.spam_log.keys():
                ham_ind.append(((self.ham_log[i] - mt.log(mt.exp(self.spam_log[i])*mt.exp(self.spam_prob) +
                                                          mt.exp(self.ham_log[i])*mt.exp(self.spam_prob_n))),i))
        ind_word = heapq.nlargest(n,ham_ind)
        return [word for (num,word) in ind_word]