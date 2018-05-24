
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from collections import namedtuple, defaultdict, Counter


# In[2]:


sns.set_context("poster")
sns.set_style("ticks")


# In[3]:


Tag = namedtuple("Tag", ["token", "tag"])
BEGIN_TAG = Tag("^", "<BOS>")
END_TAG = Tag("$", "<EOS>")

def load_sequences(filename, sep="\t", notypes=False):
    sequences = []
    prev_char_tag = "O"
    with open(filename) as fp:
        seq = []
        for line in fp:
            line = line.strip()
            if line:
                if not seq:
                    seq = [BEGIN_TAG]
                line = line.split(sep)
                if notypes:
                    line[1] = line[1][0]
                seq.append(Tag(*line))
            else:
                seq.append(END_TAG)
                sequences.append(seq)
                seq = []
        if seq:
            seq.append(END_TAG)
            sequences.append(seq)
    return sequences
    


# In[11]:


train = load_sequences("data/cleaned/train.BIEOU.tsv", notypes=True)
test = load_sequences("data/cleaned/dev.BIEOU.tsv", notypes=True)


# In[13]:


labels = sorted(Counter(t.tag for seq in train for t in seq).keys(), key=lambda x: x.split("-")[-1])
labels


# In[14]:


labels = ["MASK"] + labels
label2id = {l: i for i, l in enumerate(labels)}
label2id


# In[21]:


vocab = sorted(Counter(t.token for seq in train for t in seq).keys())
vocab = ["<MASK>", "<OOV>"] + vocab
word2id = {l: i for i, l in enumerate(vocab)}
print len(vocab)


# In[18]:


from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, TimeDistributed, Merge, merge
from keras.callbacks import TensorBoard, ModelCheckpoint


# In[43]:


INPUT_DIM=len(vocab)
EMBED_DIM=64
MAXLENGTH=50
NUM_LABELS=len(label2id)


# In[22]:


X_train = sequence.pad_sequences([[word2id.get(t.token, word2id["<OOV>"])
                                   for t in seq] for seq in train], maxlen=MAXLENGTH)
X_test = sequence.pad_sequences([[word2id.get(t.token, word2id["<OOV>"])
                                  for t in seq] for seq in test], maxlen=MAXLENGTH)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)


# In[23]:


Y_train = sequence.pad_sequences([[label2id[t.tag] for t in seq] for seq in train], maxlen=MAXLENGTH)
Y_test = sequence.pad_sequences([[label2id[t.tag] for t in seq] for seq in test], maxlen=MAXLENGTH)

train_sample_weight = Y_train != 0
test_sample_weight = Y_test != 0

Y_train = np.expand_dims(Y_train, -1)
Y_test = np.expand_dims(Y_test, -1)


print('y_train shape:', Y_train.shape)
print('y_test shape:', Y_test.shape)

print('train_sample_weight shape:', train_sample_weight.shape)
print('test_sample_weight shape:', test_sample_weight.shape)


# In[44]:


num_layers=3
lstm_units=32
dense_units=32
dropout_p=0.5

input_layer = Input(shape=(MAXLENGTH,), dtype='int32')
emb = Embedding(input_dim=INPUT_DIM, output_dim=EMBED_DIM,
                           input_length=MAXLENGTH, mask_zero=True,
                dropout=dropout_p)(input_layer)
# apply forwards LSTM
forward_lstms = []
prev_layer = emb
for i in xrange(num_layers):
    fw_lstm = LSTM(lstm_units, return_sequences=True,
                   dropout_U=dropout_p, dropout_W=dropout_p)(prev_layer)
    prev_layer = fw_lstm
    forward_lstms.append(fw_lstm)

# apply backwards LSTM
backward_lstms = []
prev_layer = emb
for i in xrange(num_layers):
    bw_lstm = LSTM(lstm_units, return_sequences=True,
                   go_backwards=True, dropout_U=dropout_p, dropout_W=dropout_p)(prev_layer)
    prev_layer = bw_lstm
    backward_lstms.append(bw_lstm)

merged_layer = merge([fw_lstm, bw_lstm], mode="concat")
dropout = Dropout(p=dropout_p)(merged_layer)
dense = TimeDistributed(Dense(dense_units, activation='tanh'))(dropout)
dense2 = TimeDistributed(Dense(dense_units, activation='tanh'))(dense)
out = TimeDistributed(Dense(NUM_LABELS, activation='softmax'))(dense2)


# In[45]:


model = Model(input=input_layer, output=out)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', sample_weight_mode="temporal",  metrics=['accuracy'])
tb_callback = TensorBoard(log_dir='./tb_logs', histogram_freq=0, write_graph=True)
mc_callback = ModelCheckpoint("models/LSTM1_word.hdf5", monitor='val_loss', save_best_only=True)


# In[ ]:


model.fit(X_train, Y_train, nb_epoch=30, batch_size=5,
          sample_weight=train_sample_weight,
         validation_data=(X_test, Y_test, test_sample_weight),
         callbacks=[tb_callback, mc_callback])


# In[47]:


model.history.history.keys()


# In[49]:


predictions = model.predict(X_train)


# In[50]:


predictions.argmax(axis=-1)[1]


# In[51]:


Y_train[1].flatten()


# In[52]:


predictions.argmax(axis=-1)[1] == Y_train[1].flatten()


# In[53]:


predictions_val = model.predict(X_test)


# In[54]:


predictions_val.argmax(axis=-1)[0]


# In[55]:


Y_test[0].flatten()


# In[57]:


from sklearn.metrics import classification_report, confusion_matrix


# In[58]:


confusion_matrix(Y_train.flatten(),
                            predictions.argmax(axis=-1).flatten(),
                labels=range(len(labels)))


# In[59]:


confusion_matrix(Y_train.flatten()[Y_train.flatten() != 0],
                            predictions.argmax(axis=-1).flatten()[Y_train.flatten() != 0],
                labels=range(1,len(labels)))

