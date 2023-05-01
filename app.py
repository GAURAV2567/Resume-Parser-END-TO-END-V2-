#!/usr/bin/env python
# coding: utf-8

# In[2]:



import numpy as np
import joblib



import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk

from annotated_text import annotated_text

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
github
     ''')

model = joblib.load(open('modelf.pkl', 'rb'))



st.write("**Summary Submitted**:\n", txt)

int_features = [txt]
#st.write(int_features)
docs=list(model.pipe(int_features))
lst=[]       
annot_lst = [] 
for doc in docs:   
    prev_Start = 0     
    for entity in doc.ents:     
        annot_lst.extend([str(doc[prev_Start:entity.start])])
        annot_lst.append((entity.text,entity.label_))

        lst.append((entity.text,entity.label_))
        prev_Start = entity.end
    annot_lst.extend([str(doc[prev_Start:])])

output = lst



df = pd.DataFrame(
    lst)

st.write("### OUTPUT:-")
annotated_text(annot_lst)
st.dataframe(df)

st.write("*Note: So far our AI has been train to extract only skill and experience. This AI can be trained further for extraction required information* ")
#st.write(output)





