#!/usr/bin/env python
# coding: utf-8

# In[6]:


import streamlit as st
import pickle
from sentence_transformers import SentenceTransformer, util


# In[7]:


pickled_model_similarity = pickle.load(open("model_similarity.pkl","rb"))


# In[8]:


def calculate_similarity(input_data_1,input_data_2):
    similarity_1 = pickled_model_similarity.encode(input_data_1)
    similarity_2 = pickled_model_similarity.encode(input_data_2)
    similarity_value = util.cos_sim(similarity_1,similarity_2)
    return similarity_value


# In[9]:


def main():
    st.title('Sentence Similarity Calculator')

    sentence1 = st.text_input('Enter Sentence 1')
    sentence2 = st.text_input('Enter Sentence 2')

    if st.button('Calculate Similarity'):
        similarity_score = calculate_similarity(sentence1, sentence2)

        st.write('Sentence 1:', sentence1)
        st.write('Sentence 2:', sentence2)
        st.write('Similarity Score:', similarity_score.item())


# In[10]:


if __name__ == '__main__':
    main()


# In[ ]:




