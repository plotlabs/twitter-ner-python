
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from collections import namedtuple, defaultdict


# In[2]:


sns.set_context("poster")
sns.set_style("ticks")


# In[3]:


DATA_DIR="data/data/"
CLEANED_DIR="data/cleaned/"

Tag = namedtuple("Tag", ["token", "tag"])

def load_sequences(filename, sep="\t", notypes=False):
    tag_count = defaultdict(int)
    sequences = []
    with open(filename) as fp:
        seq = []
        for line in fp:
            line = line.strip()
            if line:
                line = line.split(sep)
                if notypes:
                    line[1] = line[1][0]
                tag_count[line[1]] += 1
                #print line
                seq.append(Tag(*line))
            else:
                sequences.append(seq)
                seq = []
        if seq:
            sequences.append(seq)
    return sequences, tag_count

def write_sequences(sequences, filename, sep="\t", to_bieou=True):
    with open(filename, "wb+") as fp:
        for seq in sequences:
            if to_bieou:
                seq = to_BIEOU(seq)
            for tag in seq:
                print >> fp, sep.join(tag)
            print >> fp, ""                
                
def count_phrases(ptype="movie"):
    phrase_counts = defaultdict(int)
    check_tag = ptype
    for seq in sequences:
        phrase = ""
        for tag in seq:
            if not phrase and tag.tag == "B-%s" % check_tag:
                phrase = tag.token
                continue
            if tag.tag == "I-%s" % check_tag:
                phrase += " %s" % tag.token
                continue
            if phrase:
                phrase_counts[phrase] += 1
                phrase = ""
    return phrase_counts


def phrase_to_BIEOU(phrase):
    l = len(phrase)
    new_phrase = []
    for j, t in enumerate(phrase):
        new_tag = t.tag
        if l == 1:
            new_tag = "U%s" % t.tag[1:]
        elif j == l-1:
            new_tag = "E%s" % t.tag[1:]
        new_phrase.append(Tag(t.token, new_tag))
    return new_phrase

def to_BIEOU(seq, verbose=False):
    # TAGS B I E U O
    phrase = []
    new_seq = []
    for i, tag in enumerate(seq):
        if not phrase and tag.tag[0] == "B":
            phrase.append(tag)
            continue
        if tag.tag[0] == "I":
            phrase.append(tag)
            continue
        if phrase:
            if verbose:
                print "Editing phrase", phrase
            new_phrase = phrase_to_BIEOU(phrase)
            new_seq.extend(new_phrase)
            phrase = []
        new_seq.append(tag)
    if phrase:
        if verbose:
            print "Editing phrase", phrase
        new_phrase = phrase_to_BIEOU(phrase)
        new_seq.extend(new_phrase)
        phrase = []
    return new_seq


# In[4]:


sequences, tag_count = load_sequences("data/data/train", sep="\t")
write_sequences(sequences, "data/cleaned/train.tsv", to_bieou=False)


# In[5]:


sequences, tag_count = load_sequences("data/data/dev", sep="\t")
write_sequences(sequences, "data/cleaned/dev.tsv", to_bieou=False)


# In[6]:


sequences, tag_count = load_sequences("data/data/dev_2015", sep="\t")
write_sequences(sequences, "data/cleaned/dev_2015.tsv", to_bieou=False)


# In[7]:


sequences, tag_count = load_sequences("data/data/test.txt", sep="\t")
write_sequences(sequences, "data/cleaned/test.tsv", to_bieou=False)


# In[8]:


sequences, tag_count = load_sequences("data/cleaned/train.tsv", sep="\t")


# In[9]:


len(sequences)


# In[10]:


sum(len(seq) for seq in sequences)


# In[11]:


sequences[-1]


# In[12]:


tag_count


# In[13]:


phrase_counts = count_phrases(ptype="movie")
phrase_counts


# In[14]:


phrase_counts = count_phrases(ptype="facility")
phrase_counts


# In[15]:


phrase_counts = count_phrases(ptype="company")
phrase_counts


# In[16]:


phrase_counts = count_phrases(ptype="musicartist")
phrase_counts


# In[17]:


phrase_counts = count_phrases(ptype="other")
phrase_counts


# In[18]:


to_BIEOU(sequences[1])


# In[19]:


write_sequences(sequences, "data/cleaned/train.BIEOU.tsv", to_bieou=True)


# In[20]:


sequences, tag_count = load_sequences("data/cleaned/dev.tsv", sep="\t")
write_sequences(sequences, "data/cleaned/dev.BIEOU.tsv")


# In[21]:


sequences, tag_count = load_sequences("data/cleaned/dev.tsv", sep="\t")


# In[22]:


for seq in sequences:
    if seq[0].token == "Happy":
        print seq
        break

