#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf


# In[2]:


def load_embeddings(file_path):
    # load pre-trained embeddings file
    embeddings_index = {}
    with open(file_path,'r', encoding='UTF-8') as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs
    return embeddings_index


# In[18]:


def load_images(file_path, batch_size = 32,img_height = 160,img_width = 160):
    import pathlib
    data_dir = pathlib.Path(file_path)
    image_count = len(list(data_dir.glob('*/*.jpg')))
    labels=[i+1 for i in range(image_count)]
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir, labels=labels,subset=None,seed=123,image_size=(img_height, img_width),batch_size=batch_size)
#     val_ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir, labels=labels, validation_split=0.2,subset="validation",seed=123,image_size=(img_height, img_width),batch_size=batch_size)
    return train_ds
# def normalize_images(ds):
#     normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
#     normalized_ds = ds.map(lambda x, y: (normalization_layer(x), y))
#     image_batch, labels_batch = next(iter(normalized_ds))


# In[32]:


def load_attributes(file_path):
    with open(file_path,'r', encoding='UTF-8') as f:
        image_count = int(f.readline())
        raw_cats = f.readline().split()
        attributes=np.zeros([image_count+1,len(raw_cats)])
        cats = [s.lower().split('_') for s in raw_cats]
        cats[0] = ['5', 'o\'clock', 'shadow']
        i=0
        for line in f:
            i+=1
            image, labels = line.split(maxsplit=1)
            labels = np.fromstring(labels, "f", sep=" ")
            attributes[i]=labels
    return cats, attributes


# In[31]:


if __name__ == '__main__':
    d = load_embeddings("/git-repos/latent-space-arithmetic/dataset/embedding/glove.6B.50d.txt")
    print(d['o'])
    train_ds = load_images("/git-repos/latent-space-arithmetic/dataset/")
    load_attributes("/git-repos/latent-space-arithmetic/dataset/Anno/list_attr_celeba.txt")
    import matplotlib.pyplot as plt

    for images, labels in train_ds.take(1):
        print(labels)
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.axis("off")
