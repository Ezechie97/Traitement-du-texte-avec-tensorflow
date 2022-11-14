#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


# In[3]:


text= open("C:/Users/33656/Desktop/LIVRES DE VACANCES DATASCIENCES/Deep_Learning_TF_2_Notebooks_et_Datasets (1)/Deep_Learning_TF_2_Notebooks_et_Datasets/06-NLP-et-Donne╠ües-Textuelles/shakespeare.txt",'r').read()


# In[4]:


print(text[:500])


# In[5]:


# Comprendre les caractères Uniques


# In[6]:


# Extraction des caractères des caractères uniques dans le fichier
vocab = sorted(set(text))
print(vocab)
len(vocab)


# In[7]:


#Traitement du texte


# In[8]:


for pair in enumerate(vocab):
    print(pair)


# In[9]:


#indexation par les caractère
char_to_ind={char:ind for ind,char in enumerate(vocab)}
char_to_ind


# In[10]:


#indexation par les chiffres
ind_to_char=np.array(vocab)


# In[11]:


encoded_text=np.array([char_to_ind[c] for c in text])
encoded_text.shape


# In[12]:


encoded_text[:500]


# In[13]:


text[:500]


# In[14]:


#Création de batches


# In[15]:



seq_len=120
total_num_seq=len(text)//(seq_len+1)
total_num_seq


# In[16]:


# séquences d'entraînement
char_dataset=tf.data.Dataset.from_tensor_slices(encoded_text)

for i in char_dataset.take(500):
    print(ind_to_char[i.numpy()])


# In[17]:


sequences=char_dataset.batch(seq_len+1, drop_remainder=True)


# In[18]:


def create_seq_targets(seq):
    input_txt=seq[:-1]
    target_txt=seq[1:]
    return input_txt, target_txt


# In[19]:


dataset=sequences.map(create_seq_targets)
dataset


# In[20]:


for input_txt, target_txt in dataset.take(1):
    print(input_txt.numpy())
    print(''.join(ind_to_char[input_txt.numpy()]))
    print('\n')
    print(target_txt.numpy())
    print(''.join(ind_to_char[target_txt.numpy()]))


# In[21]:


batch_size=128


# In[22]:


buffer_size=10000


# In[23]:


dataset=dataset.shuffle(buffer_size).batch(batch_size,drop_remainder=True)
dataset


# In[24]:


vocab_size=len(vocab)


# In[25]:


embed_dim=64


# In[26]:


rnn_neurons=1026


# In[27]:


from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Embedding,Dropout,GRU


# In[28]:


def sparse_cat_loss(y_true,y_pred):
    return sparse_categorical_crossentropy(y_true,y_pred, from_logits=True)


# In[29]:


def create_model(vocab_size, embed_dim, rnn_neurons, batch_size):
    model = Sequential()
    model.add(Embedding(vocab_size, embed_dim,batch_input_shape=[batch_size, None]))
    model.add(GRU(rnn_neurons,return_sequences=True,stateful=True,recurrent_initializer='glorot_uniform'))
    # Couche Finale Dense de Prédiction
    model.add(Dense(vocab_size))
    model.compile(optimizer='adam', loss=sparse_cat_loss) 
    return model


# In[30]:


model=create_model(vocab_size=vocab_size,
                  embed_dim=embed_dim,
                  rnn_neurons=rnn_neurons,
                  batch_size=batch_size)


# In[31]:


model.summary()


# In[32]:


for input_example_batch, target_example_batch in dataset.take(1):

  # Prédire sur un lot aléatoire
  example_batch_predictions = model(input_example_batch)

  # Afficher les dimensions des prédictions
  print(example_batch_predictions.shape, " <=== (batch_size, sequence_length, vocab_size)")


# In[33]:


sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)


# In[34]:


sampled_indices


# In[35]:


sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()


# In[36]:


sampled_indices


# In[37]:


print("Compte tenu de la séquence d'entrée : \n")
print("".join(ind_to_char[input_example_batch[0]]))
print('\n')
print("Prochain caractère prédit : \n")
print("".join(ind_to_char[sampled_indices ]))


# In[38]:


epochs = 30


# In[ ]:


model.fit(dataset,epochs=epochs)


# In[ ]:


# Génération du text


# In[ ]:


model.save('shakespeare_gen.5')


# In[ ]:


from tensorflow.keras.models import load_model


# In[ ]:


model=create_model(vocab_size,embed_dim,rnn_neurons,batch_size=1)
model.load_weights('shakespeare_gen.5')
model.build(tf.tensorShape([1,None]))


# In[ ]:


model.summary()


# In[ ]:


def generate_text(model, start_seed,gen_size=100,temp=1.0):
  '''
  model: Modèle Entraîné pour générer du texte
  start_seed: Seed initial du texte sous forme de chaîne de caractères
  gen_size: Nombre de caractères à générer

  L'idée de base de cette fonction est de prendre un texte de départ, de le formater de manière à
  qu'il soit dans le bon format pour notre réseau, puis bouclez la séquence à mesure que
  nous continuons d'ajouter nos propres caractères prédits. Similaire à notre notre travail au sein 
  des problèmes de séries temporelles avec le RNN.
  '''

  # Nombre de caractères à générer
    num_generate = gen_size

  # Vectorisation du texte du seed de départ
    input_eval = [char_to_ind[s] for s in start_seed]

  # Étendre les dimensions pour correspondre à la forme du format de batch
    input_eval = tf.expand_dims(input_eval, 0)

  # Liste vide pour contenir le texte généré
    text_generated = []

  # La température a un effet aléatoire sur le texte qui en résulte
  # Le terme est dérivé de l'entropie/thermodynamique.
  # La température est utilisée pour affecter la probabilité des caractères suivants.
  # Probabilité plus élevée == moins surprenante/ plus attendue
  # Une température plus basse == plus surprenante / moins attendue
    temperature = temp

  # Ici batch size == 1
     model.reset_states()
    for i in range(num_generate):

      # Générer des prédictions
        predictions = model(input_eval)

      # Supprimer la dimension de la forme du batch
        predictions = tf.squeeze(predictions, 0)

      # Utilisez une distribution catégorielle pour sélectionner le caractère suivant
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # Passez le caractère prédit pour la prochaine entrée
        input_eval = tf.expand_dims([predicted_id], 0)

      # Transformer à nouveau en lettre de caractère
        text_generated.append(ind_to_char[predicted_id])
return (start_seed + ''.join(text_generated))


# In[ ]:


print(generate_text(model,"flower",gen_size=1000))


# In[ ]:




