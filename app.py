#!/usr/bin/env python
# coding: utf-8

# In[2]:



import numpy as np
import pickle



import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")


st.title('Resume Parser')


'##### Instructions:- Select text from LinkedIn Summary Section Only '

txt = st.text_input('Resume summary to Analyze', '''
Been a CS Graduate, I am a Data Science enthusiast 
with a good understanding of Machine Learning algorithms 
Techniques such as Regression, Classification, and Clustering, along with 
Computer Vision and NLP. Self Thaught Python Programmer. 
Has a good understanding of SQL and also has good Data Visualisation skills, 
and has worked with tools like Tableau and Python libraries such as Numpy and Pandas,
Scikit learn, Matplotlib and Seaborn. 
I'm highly passionate about Data Science and intending to pursue a career in Big data/Data Scientist. 
github link - https://github.com/pramod1998-hash
     ''')

model = pickle.load(open('modelf.pkl', 'rb'))



st.write('**Summary Submitted**:\n', txt)

int_features = [txt]
#st.write(int_features)
docs=list(model.pipe(int_features))
lst=[]        
for doc in docs:        
    for entity in doc.ents:
        #st.write((entity.text,entity.label_))
        lst.append((entity.text,entity.label_))

output = lst

df = pd.DataFrame(
    lst)

st.write("### OUTPUT:-")
st.dataframe(df)

st.write("*Note: So far our AI has been train to extract only skill and experience. This AI can be trained further for extraction required information* ")
#st.write(output)





