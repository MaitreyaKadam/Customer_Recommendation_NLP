#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Milestone I Natural Language Processing
# ## Task 2&3
# #### Student Name: Maitreya Milind Kadam
# #### Student ID: S4087536
# 
# 
# Environment: Python 3 and Jupyter notebook
# 
# Libraries used: please include all the libraries you used in your assignment, e.g.,:
# * pandas
# * re
# * numpy
# * scipy.sparse
# * sklearn.feature_extraction.text
# * sklearn.model_selection
# * sklearn.linear_model
# * gensim.models.fasttext
# 
# ## Introduction
# In this task, the main focus was to generate various feature representations of review description using models like Bag-of-Words, unweighted, and TF-IDF weighted using FastText. The count vector representation is also stored under the file name count_vectors.txt. These representations are then used to train and evaluate a Logistic Regression classifier to determine the most effective language model for predicting recommendations.
# 
# Furthermore, the task explores the impact of adding some extra information, specifically just the review title, or the review title and review description both. Initially, separate feature sets are generated for the review title and the feature sets for review description were already created.  So, using the feature sets for the review title the Logistic Regression model is trained and evaluated. Similary, the feature sets for the review title and the review description are combined using horizontal stacking and then the Logistic Regression model is trained and evaluated to assess the performance of the model on different feature sets.
# 

# ## Importing libraries 

# In[1]:


import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from scipy.sparse import hstack
from gensim.models.fasttext import FastText


# ## Importing the data from processed.csv and vocab.txt

# In[2]:


rev_data=pd.read_csv('processed.csv')
processed_review=rev_data['Processed Review Text'].astype(str)


# In[3]:


with open('vocab.txt','r') as f:
    vocab_list=[line.strip().split(':')[0]for line in f] #since the vocab.txt contains words:index and we just need words for the generating the Count Vector
    


# ## Task 2. Generating Feature Representations for Clothing Items Reviews
# 
# ### Bag-of-words model:
# Inorder to generate the Count Vector representation for each clothing review, the Bag-of-Words model converts the text into numerical vectors just by counting the word occurrences. Firstly, the CountVectorizer was initialized with the vocab_list that was created in task 1. The fit_transform() method then processed the processed_review text that is basically the processed reviews that we did in task 1. Eventually, it creates a matrix where each row is a review and each column represents a word from the vocabulary. The value present in the matrix indicates the frequency of that word in the review.

# In[4]:


#Generating Count Vectors
c_vectors=CountVectorizer(analyzer='word',vocabulary=vocab_list) #Initializes a CountVectorizer object using the above defined vocabulary list for Review Text(description) 
countvecs=c_vectors.fit_transform(processed_review) #learns the vocabulary from the data and transforms the 'Review Text' into a document-term matrix
countvecs.shape #Each row in 'countvecs' represents a review, and each column represents a word in the vocabulary


# The above output describes the shape of the resulting matrix. The values present in the matrix are nothing but the frequency of that word in the review.

# ### FastText Language Model
# 
# A FastText model is implemented using Gensim to generate word embeddings for the processed review text. Each word is represented as an n-gram of characters, which allows it to handle out-of-vocabulary words. For model is trained on the tokenized reviews to learn vector representation for words. These word embeddings are further averaged to create document-level embeddings, both unweighted and weighted by TF-IDF scores, to represent the meaning of each review.

# In[5]:


processed_review=processed_review.tolist() #Converts the 'processed_review
tk_review=[review.split() for review in processed_review] #it's a list of list where each inner list contains the tokens for a corresponding review.


# In[6]:


model_FT=FastText(vector_size=100,min_count=2) #Initializes a FastText model object from the gensim library.
model_FT.build_vocab(corpus_iterable=tk_review) #Builds the vocabulary of the FastText model based on the tokens present in the 'tk_review'
model_FT.train(corpus_iterable=tk_review,total_examples=len(tk_review),epochs=model_FT.epochs) #Trains the FastText model on the 'tk_review' corpus.
model_FT_wv=model_FT.wv #It is a variable that can be used to retrieve the word embeddings learned by the FastText model.


# In[7]:


#The below function is to get the unweighted average FastText Vectors
# processed_review is a string containing the preprocessed review text(or title, or combined text)
# word_vector contains the learned word embeddings from the FastText model
def unweighted_FT(processed_review,word_vector):
    w_vector=[word_vector[word] for word in processed_review.split() if word in word_vector] #Gets the word vectors for each word present in review and vocab.
    if not w_vector: #this if-statement checks if any word in the review is present in the vocab
        return np.zeros(word_vector.vector_size) #returns a zero vector if the word is not present in the vocab
    return np.mean(w_vector,axis=0) # returns the element-wise mean of the word vector if the word is present in the vocab.


# In[8]:


#Calculating TF-IDF weights
tfidf_vec=TfidfVectorizer() #Initializes a TfidVectorizer object
tfidf_vec.fit_transform(processed_review) #the object learns the vocabulary and calculates IDF scores from the processed_review
w_to_tfidf=dict(zip(tfidf_vec.get_feature_names_out(),tfidf_vec.idf_)) #Creates a dictionary mapping which maps words to their calculated IDF weights.


# In[9]:


#Function to get the TF-IDF weighted average FastText vector--> we will use this function in task3 as well for title and title+description
def tfidf_weighted(processed_review,word_vector,tfidf_wts):
    wt_word_vectors=[] #an empty list to store the weighted word vectors.
    for word in processed_review.split(): #The .split() method separates each word and the for loop iterates through each word in the processed_review
        if word in word_vector: #Checks if the word exists in the FastText word vector vocabulary
            tfidf_wt=tfidf_wts.get(word,0) #Gets the TF-IDF weight for the word, the default is 0 if it is not found
            wt_word_vectors.append(tfidf_wt*word_vector[word]) #multiplies the word vector by it's TF-IDF weight and appends to the list using the .append() method
    if not wt_word_vectors: #If statement checks if the list of weighted word vectors is empty or not.
        return np.zeros(word_vector.vector_size) #returns a zero vector if no words from the review are in the FastText vocabulary.
    return np.mean(wt_word_vectors,axis=0) #returns the calculated element-wise mean of the weighted word vector.


# In[10]:


#Generating the vectors for each processed review
unweight_vector=[] #list to store unweighted FastText vectors
tfidf_weight_vector=[] #list to store TF-IDF weighted FastText vectors
for rev in processed_review: #Iterates through each processed review in the 'processed_review' list
    unweight_vector.append(unweighted_FT(rev, model_FT_wv)) #Calculates the unweighted FastText vector for the current review and appends using the .append() to the list
    tfidf_weight_vector.append(tfidf_weighted(rev, model_FT_wv, w_to_tfidf)) #Calculates the TF-IDF weighted FastText vector for the current review and appends it to the list.


# ### Saving outputs
# 
# The count vector representation created is stored in the count_vectors.txt as per the required format. 
# - count_vectors.txt

# In[11]:


w_t_i={word:index for index,word in enumerate(vocab_list)} #Creates a mapping from word to index using vocab_list
out_lines=[] #A list for storing the output lines that are to be stored in the count_vectors.txt
for ind,row in rev_data.iterrows(): #Iterates through each row of the processed dataset i.e processed.csv
    word_index=row['Clothing ID'] #Extracts the 'Clothing ID' that will act as the word_index
    proc_text=row['Processed Review Text'] #Extracts the processed review text.
    repr_list=[] #A list for storing the current review's representation
    if isinstance(proc_text,str): #The if-statement checks if the processed text is a string.
        count_w={} #stores the word counts
        for word in proc_text.split(): #The for-loop iterates through each word in the processed review text.
            if word in w_t_i: #checks if the word is present in the mapping that we created earlier.
                count_w[word]=count_w.get(word,0)+1 #If yes, the count of the word is increamented.
        for word,word_freq in count_w.items(): #This for-loop iterates through count_w that stores the words along with their respective frequencies.
            wordi_ind=w_t_i[word] #Gets the word integer index for every word
            repr_list.append(f'{wordi_ind}:{word_freq}') #Appends the word index along with the frequency to the representation list.
            
    repr_str=','.join(repr_list) #joinrs the representation list with commas as it is required in the given format.
    out_lines.append(f'#{word_index},{repr_str}') #Appends the word index along with representation string separated by a comma as per the required format.
    
with open('count_vectors.txt','w') as f: #Opens a new file named count_vectors.txt in write mode
    for line in out_lines: #the for-loop iterates through each output line that is to be stored in the .txt file
        f.write(line+'\n') #writes each line to the file with a newline so that every line is on a newline.


# ## Task 3. Clothing Review Classification

# For Task 3, there are 2 sub-tasks i.e. Q1 and Q2 so firstly Q1 will be addressed and then Q2.
# 
# Q1: For Q1, a Logistic Regression model was used based on the feature representation generated in Task 2 i.e. Bag-of-Words, Unweighted and TF-IDF weighted. Once the Logistic Regression model was trained on the different feature representations, it was evaluated using K-Fold cross-validation to obtain mean accuracy and weighted F1-score for each feature set. By comparing these evaluation metrics, the language model that yiels the highest score with the Logistic Regression classifier can be selected as the most effective for this task.
# 
# ### For Review Text (Description)

# In[12]:


# Q1
unweight_vector=np.array(unweight_vector) #Converts the list of unweighted FastText vectors to a numpy array
tfidf_weight_vector=np.array(tfidf_weight_vector) #Converts the list of TF-IDF weighted FastText vectors to a numpy array


# In[13]:


y=rev_data['Recommended IND'] #the target variable is the 'Recommended IND'
log_model=LogisticRegression(max_iter=800) #Initializes a Logistic Regression model.
#The function defined below evaluates the classification model using KFold cross validation. It can be used for any classification model.
def eval_model(X,y,model_name,name):
    kfoldvalidate = KFold(n_splits=5, random_state=42, shuffle=True) #A 5-fold cross-validation is initialized
    acc_score = cross_val_score(model_name, X, y, cv=kfoldvalidate, scoring='accuracy') #Calculates the accuracy score for each fold
    f1score = cross_val_score(model_name, X, y, cv=kfoldvalidate, scoring='f1_weighted') #Calculates the weighted-F1 score for each fold.
    print(f"{name}")
    print(f"Mean Accuracy: {acc_score.mean():.4f}")
    print(f"Mean Weighted F1-score: {f1score.mean():.4f}\n")
    return acc_score.mean(),f1score.mean() #returns the mean accuracy score and the mean f1 score


# In[14]:


# Perform evaluation for each feature set
print('Evaluation of Logistic Regression Model for Review Text(Description)')
bow_eval_descrip=eval_model(countvecs, y, log_model, "Bag-of-Words (CountVectorizer)") #Evaluates the classification model with Bag-of-Words features
unweight_eval_descrip=eval_model(unweight_vector, y, log_model, "Unweighted FastText Embeddings") #Evaluates the classification model with unweighted FastText features
tfid_eval_descrip=eval_model(tfidf_weight_vector, y, log_model, "TF-IDF Weighted FastText Embeddings") #Evaluates the classification model with TF-IDF weighted FastText features


# From the above output, the Logistic Regression model performed the best with Bag-of-Words features achieving the highest mean accuracy of 0.8753 and F1-score of 0.8686. This suggests that the frequency of individual words in the review description was more predictive of the recommendation than the unweighted and weighted TF-IDF embeddings captured by FastText.
# 
# 
# Q2: For Q2, spme extra information is explored to check if it boosts up the accuracy of the model or not. 3 experiments are performed to build and compare the performance of classification models considering 1) only the review title, 2) only the review description(this I have already done above), and 3) both title and review description. 
# 
# ## For Just The Title
# 
# In this task, just the title has to be used so a new vocabulary list has to be created as it is a new column and the previous vocabulary was suitable for the processed review text column. Firstly, the data from the review title column was tokenized and pre-processed using the regex pattern that was used for review desccription. Once the tokenisation was done, the vocabulary was created and the respective Bag-of-Words, unweighted, and weighted TF-IDF embeddings were created.

# In[15]:


#So now we will check if using just the title of the review makes any change in the accuracy of the model.
#For that we will have to pre-process the title column and create a new vocabulary for that column
pattern_title=r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"
#Tokenization using regex
def token_regex(review_text):
    if isinstance(review_text,str): #Checks if the input from the title column is a string.
        tk_review=re.findall(pattern_title,review_text) #Extracts all the matches and generates tokens
        return tk_review #Returns the list of tokens found
    return [] #If the input from the title column is not a string, it returns an empty list.
tokenized_review=rev_data['Title'].apply(token_regex).tolist() #calls the above function and tokenizes each entry in the title column.
lowercase_rev=[] #An empty-list to store all the tokens which are converted to lowercases.
for tk_list in tokenized_review:  #for-loop iterates through each tokenized review 
    lwcase_token=[tk.lower() for tk in tk_list] #Converts each token into lowercase
    lowercase_rev.append(lwcase_token) #Appends the lowercase tokens to the list that we declared earlier.


# In[16]:


#creating the vocab list for just the TITLE:
vocab_title=set() #An empty set to store unique words for the title column.
for tk in lowercase_rev: #Iterates through each lowercase token
    vocab_title.update(tk) #adds all the lowercase tokens to the vocab_title(duplicates are automatically handled by the set)
sort_vtitle=sorted(list(vocab_title)) #converts the vocab_title set into a list and uses the sorted() function to sort it alphabetically.


# In[17]:


#Generating Cfeatures for Title column
title=rev_data['Title'].astype(str) #ensures all the entries in the 'title' column is treated as strings.
title_vector=CountVectorizer(analyzer='word',vocabulary=sort_vtitle) #Initializes the CountVectorizer object on the vocabulary that is made for just the 'Title' column
t_features=title_vector.fit_transform(title)  # bag of words vector for title of the reviews
t_features.shape #Displays the shape of the resulting Bag-of-Words feature matrix for the count vectors for the 'Title' column.


# In[18]:


#Generating Features for Title column using FastText
title=title.tolist() #Converts the 'title' series to a list
title_rev=[rev.split() for rev in title] #tokenize each title into a list of words.
modelt=FastText(vector_size=100,min_count=2) #Initialize a FastText model.
modelt.build_vocab(corpus_iterable=title_rev) #Building the vocab for the model based on 'Title' tokens
modelt.train(corpus_iterable=title_rev,total_examples=len(title_rev),epochs=modelt.epochs) #trains the FastText model on the tokens from the title column.
modelt_wv=modelt.wv #access the word vectors from the trained FastText model for the 'title' column.
#Calculating TF-IDF weights and unweighted vectors
tfidf_title=TfidfVectorizer() #Initializes a TfidVectorizer.
tfidf_title.fit_transform(title) #learns the vocabulary and calculates the IDF score from the 'title' column
t_to_tfidf=dict(zip(tfidf_title.get_feature_names_out(),tfidf_title.idf_)) #a dictionary is created for mapping words from the title vocabulary to their TF-IDF weights
unweight_vector_t=[] #unweighted vectors for just the title.
tfidf_t_vector=[] #tfidf vectors for just the title.
for rev in title: #the for-loop iterates through each title in the 'title' list.
    unweight_vector_t.append(unweighted_FT(rev,modelt_wv)) #calculates and appends the unweighted FastText vector for the title column
    tfidf_t_vector.append(tfidf_weighted(rev,modelt_wv, t_to_tfidf)) #the TF-IDF weighted FastText vector is calculated for the title column and it is appended.


# In[19]:


unweight_vector_t=np.array(unweight_vector_t) #the list of unweighted FastText title vectors is converted to a numpy array
tfidf_t_vector=np.array(tfidf_t_vector) #the list of TF-IDF weighted FastText title vectors is converted to a numpy array
y=rev_data['Recommended IND'] #the target variable is Recommended IND.
log_t_mod=LogisticRegression(max_iter=800)  #Initializes a logistic regression model
# Perform evaluation for each feature set
print('Evaluation of Logistic Regression Model for Title')
bow_eval_title=eval_model(t_features, y, log_t_mod, "Bag-of-Words (CountVectorizer)") #Evaluates the Log Reg model using Bag-of-words features from title column
unweight_eval_title=eval_model(unweight_vector_t, y, log_t_mod, "Unweighted FastText Embeddings") #using unweighted FastText embeddings the logistic regression model is evaluated.
tfid_eval_title=eval_model(tfidf_t_vector, y, log_t_mod, "TF-IDF Weighted FastText Embeddings") #the TF-IDF weighted FastText embeddings are used to evaluate the logistic regression model.


# From the above output, using just the review titles on the Logistic Regression model revelead that the Bag-of-Words outperformed the Bag-of-Words from the review description. The logistic regression model achieved a mean-accuracy of 0.885 on using Bag-of-Words for review title as compared to a mean accuracy of 0.8753 on using Bag-of-Words for review description. The weighted F1-score was also improved on using review title instead of review description. This strongly indicates that the specific words present in the titles were captured by Bag-of-Words. The simplicity of titles likely favor the direct word count approach for this classification task.
# 
# ## For the Title along with the Description
# 
# To check if combining Title along with the description has any affect on the accuracy of the model, the separated features that were created fro the title and the review description were combined using horizontal stacking of the respective sparse matrices. This combined representation allows the model to learn from information that is present in both the title and the review description.

# In[20]:


log_q2=LogisticRegression(max_iter=800) #Initializes a Logistic Regression model
print('Evaluation of Logistic Regression Model for (Title+Review Text)')
#Combining Bag-of-Words features
combined_features_bow = hstack([t_features, countvecs]) #Horizontally stacks the Bag-of-words features of the title and the review description
bow_eval_combine=eval_model(combined_features_bow, y, log_q2, "Bag-of-Words (CountVectorizer)") #evaluates the Logistic Regression model using the combined Bag-of-words features.
#Combining Unweighted FastText features
combined_features_ft_unweighted = np.hstack([unweight_vector_t, unweight_vector]) #Horizontally stacks the unweighted FastText embeddings of the title and the review description
unweight_eval_combine=eval_model(combined_features_ft_unweighted, y, log_q2, "Unweighted FastText") #the classification model is evaluated using the combined unweighted FastText embeddings.
#Combining TF-IDF Weighted FastText features
combined_features_ft_tfidf = np.hstack([tfidf_t_vector, tfidf_weight_vector]) #Horizontally stacks the TF-IDF weighted FastText embeddings of the title and the review description
tfid_eval_combine=eval_model(combined_features_ft_tfidf, y, log_q2, "TF-IDF Weighted FastText") #the classification model is evaluated using the combined TF-IDF weighted embeddings.


# From the above output, it is very clear that combining title and review description has improved the performance of the logistic regression model. The model achieved the highest accuracy of 0.9008 and F1-score of 0.8980 with the combined Bag-of-words features. This score is greater than the score obtained for just the title and just the review description. This suggests that using word counts from both the title and the review description provides the most effective information for prediction recommnedations. The unweighted and TF-IDF weighted embeddings also showed improvement when combined, but they still underperformed compared to the Bag-of-Words features.

# ## Summary
# 
# In conclusion, the Logistic Regression model performed best with Bag-of-Words features for both review descriptions and titles. Notably, the review title yielded slightly better results than review description. However, the highest accuracy was achieved by combining Bag-of-words features from both title and description. This suggests that using word frequencies from both the review title and review description provides the most predictive information for recommendations. The unweighted and TF-IDF weighted FastText embeddings consistently underperformed compared to the Bag-of-Words features.
