

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from collections import namedtuple, defaultdict


sns.set_context("poster")
sns.set_style("ticks")


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


sequences, tag_count = load_sequences("data/cleaned/train.tsv", sep="\t")

write_sequences(sequences, "data/cleaned/train.BIEOU.tsv", to_bieou=True)

sequences, tag_count = load_sequences("data/cleaned/dev.tsv", sep="\t")
write_sequences(sequences, "data/cleaned/dev.BIEOU.tsv")


sequences, tag_count = load_sequences("data/cleaned/dev.tsv", sep="\t")


