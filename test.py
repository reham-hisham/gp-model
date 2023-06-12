# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 02:34:46 2023

@author: user
"""

from fastapi import Request
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from pyresparser import ResumeParser
from docx import Document
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import joblib
import spacy
from sklearn.neighbors import NearestNeighbors

import os
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
print(os.getcwd())
app = FastAPI()
import cloudinary
from cloudinary import utils 
import urllib.request
from pydantic import BaseModel
#spacy_model = "en_core_web_sm"
#nlp = spacy.load(spacy_model)
 #Configure Cloudinary with your credentials
cloudinary.config(
    cloud_name= "dixwpsf3m",
    api_key= "566711157714741",
    api_secret= "UpTlpJSQstQCQXH3q6b4HLhVC6M" 
)
import nltk
import os



def download_cv(public_id: str, filename: str):
     #Generate the URL for the CV file
     cv_url = cloudinary.utils.cloudinary_url(public_id)[0]
    
        # Download the CV file
     url, _ = urllib.request.urlretrieve(cv_url, filename)
     return filename    
    #Download the CV file





         
@app.post("/testing")
async def testing_resumes(testingdata : Request):
    body = await testingdata.json()
    resume_skills={}
    cvfile = None
    print("hna 1")
    print("datattatattta" ,body["users"])
    for user in body["users"]:
        print(user)
        try:
            doc = Document()
          
            with open(user, 'r') as file:
                doc.add_paragraph(str(file.read()))
            doc.save("text.docx")
            data = ResumeParser('text.docx').get_extracted_data()
            resume_skills[user] = data['skills']
            print("hna 3")

        except: 
            data = ResumeParser(user).get_extracted_data()
            resume_skills[user] = data['skills']
            
            
            


            

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(list(resume_skills.items()), columns=['Resume', 'Skills'])

    # Print the DataFrame
    print(df)

    # Save the DataFrame to a CSV file
    df.to_csv('resume_skills.csv', index=False)

    p_stemmer = PorterStemmer()
    the_skills=[]
    the_skills.extend(body["skills"])
    skills = []
    skills.append(' '.join(p_stemmer.stem(word.lower()) for word in the_skills))
    print(skills)
    dataset_path = "resume_skills.csv"
    
    jobs = pd.read_csv(dataset_path)
    print(jobs.head())
    
    
    my_stopwords  = set(stopwords.words('english'))
    
    # Define the tokenizer function
    def my_tokenizer(text):
        # Tokenize the text using NLTK's word_tokenize function
        tokens = word_tokenize(text)
    
        # Remove stop words and words with length less than 3
        tokens = [word for word in tokens if word not in my_stopwords and len(word) > 2]
    
        # Lemmatize the words
        tokens = [p_stemmer.stem(word.lower()) for word in tokens]
    
        return tokens
    
    jobs['skills'] = jobs['Skills'].apply(my_tokenizer).apply(' '.join)
    test = (jobs['skills'].values.astype('U')) # unicode string
    
    
    def generate_ngrams(text, n=1):
        # Convert to lowercase
        text = text.lower()
    
        # Replace special characters with spaces
        special_chars = [")", "(", ".", "|", "[", "]", "{", "}", "'"]
        for char in special_chars:
            text = text.replace(char, " ")
    
        # Replace '&' with 'and'
        text = text.replace("&", "and")
    
        # Replace multiple spaces with a single space
        text = re.sub("\s+", " ", text)
    
        # Tokenize into words
        words = text.split()
    
        # Create n-grams
        ngrams = []
        for i in range(len(words) - n + 1):
            ngrams.append(" ".join(words[i:i+n]))
    
        return ngrams
    
    
    vectorizer = TfidfVectorizer(min_df=1, analyzer=generate_ngrams, lowercase=False)
    tfidf = vectorizer.fit_transform(skills)
    
    # "analyzer" (a function that is applied to each document to extract the features)
    
    # Load the saved model using joblib
    print(os.getcwd())
    nbrs = joblib.load(r"saved_model.joblib")
    
    
    
    def getNN(query):
        queryTFIDF_ = vectorizer.transform(query)
        query_dense = queryTFIDF_.toarray()
        distances,indices = nbrs.kneighbors(query_dense)
        return distances, indices
    
    
      # The distances represent the distance between each query vector and its nearest neighbor in the training set
      # the indices represent the index of the nearest neighbor in the training set.
      
      
    distances, indices = getNN(test)
    # Convert distances to similarities
    similarities = 1 - distances
    matches = []
    
    
    for i,j in enumerate(indices):
        sim=round(similarities[i][0],2)
      
        temp = [sim]
        matches.append(temp)
        
    matches = pd.DataFrame(matches, columns=['Match confidence'])
    
    print(matches.head())
    
    # Combine the match confidence with the jobs dataset
    jobs_plus = pd.concat([jobs, matches], axis=1)
    
    # Sort the jobs dataset by match confidence
    jobs_plus = jobs_plus.sort_values('Match confidence', ascending=False)
    
    # Print the top 10 jobs with the highest match confidence
    return (jobs_plus[['Resume', 'Match confidence']].head(10))
    
    # reset_index() resets the index of the resulting dataframe to start from 0





