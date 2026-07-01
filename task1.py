#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Milestone I Natural Language Processing
# ## Task 1. Basic Text Pre-processing
# #### Student Name: Maitreya Milind Kadam
# #### Student ID: S4087536
# 
# 
# Environment: Python 3 and Jupyter notebook
# 
# Libraries used:
# * pandas
# * re
# * nltk.FreqDist
# 
# ## Introduction
# In this task, the main focus is on to perform the basic text pre-processing steps which includes, tokenization, removing most/least frequent words and stop words. Firstly, the information about the review i.e Review Text was extracted from the csv. Once it was extracted, the regex pattern that was provided was applied onto each row of that review. Upon applying the regrex pattern each clothing review was tokenized and stored into a list. Each list represents 1 row(Review Text).
# 
# Moving on, the next sub-task was to convert all the tokens to lowercase in which the .lower() method was used on every token present in the list so as to convert all the tokens into lowercase and maintain a consistent format. The next sub-task was to remove all the tokens that had a length less than 2 for which the len() method was used on every token and the tokens that had a length less than 2 were removed. Next, the stopwords were to removed from the provided stopwords list i.e. stopwords_en.txt. Upon successfull deletion of the stopwords from the list of tokens the term frequency was calculated. Term frequency basically means how many times a specific word appears in a row. Once the term frequency was calculated, the words that appeared just once in the document were removed i.e. words having term frequency equal to 1. Moving on, the document frequency was calculated. Document frequency means the number of rows that contained a specific word, regardless of how many times it appears within each document. Upon calculating the document frequency the top 20 most frequent words were acquired using the .most_common() function. These top 20 most frequent words were then excluded from the tokens list.
# 
# Once all these pre-processing steps were completed. The processed data is saved into the processed.csv file. In addition to that, the vocabulary of the cleaned review text is saved in a txt file which is named as vocab.txt.

# ## Importing libraries 

# In[1]:


import pandas as pd
import re
from nltk import FreqDist


# ### 1.1 Examining and loading data
# The data provided i.e. assignment3.csv was loaded using the .read_csv() function and then the first two rows of the dataset were displayed to examine the initial structure of the dataset.

# In[2]:


#Inspection of data
rev_data=pd.read_csv('assignment3.csv') #the .read_csv() reads the data from the assignment3.csv into a pandas Dataframe for further pre-processing.
rev_data.head(2) #we can inspect the first few rows using the .head() function


# In[3]:


rev_data['Review Text'] #inspecting the entries present in the 'Review Text' column which is to be pre-processed further.


# In[4]:


rev_data.info() #The .info() method displays the overall information of the dataframe including datatype of every column.


# The above output displays the overall information of the pandas dataframe including the number of rows and the datatype of each column present in the dataframe. There are 19,662 rows present in the dataframe.

# ### 1.2 Pre-processing data
# Inorder to pre-process the data, I have performed various steps involding tokenization, converting the tokens to lowercase, removing words having length greater than 2, removing words from the stopwords, removing words based on their term frequency, removing top 20 most common words according to the document frequency.

# ### 1) Tokenization
# - For tokenization, the given regex pattern was used to tokenize every review present in the review text. Once the regex pattern was applied onto all the reviews, they were tokenized and stored in a list.

# In[5]:


pattern=r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?" #the regex pattern which is to be followed for tokenization


# In[6]:


#I am defining a function below so as to tokenize every entry in the Review Text
def token_regex(review_text):
    if isinstance(review_text,str): #the if-statement checks if the input string from the column 'Review Text' is string or not
        tk_review=re.findall(pattern,review_text) #uses above defined regex pattern to find all the tokens
        return tk_review #returns the list of tokens found from the column
    return [] #returns an empty list if the input is not a string.


# The above function was used so that every review is tokenized by applying the regex pattern and then the created tokens are stored in a list.

# In[7]:


#Sub-task of Task 1 i.e Tokenize each clothing review
tokenized_review=rev_data['Review Text'].apply(token_regex).tolist() # .apply() function applies the above function to each individual review description
#once the function is applied the resulting series of tokens is converted into a list()


# In[8]:


#Not required line --> Just for checking the output
for i in range(2): #the for-loop is used to iterate through the first 2 list of the tokenized reviews
    print(f"Review {i+1} tokens: {tokenized_review[i]}") #Prints the list of tokenized reviews


# From the above output, you can see that the reviews have been tokenized and each token is stored in the list i.e. each list represents 1 review.
# 
# ### 2) Conversion to Lowercase
# - For converting the tokens to lowercase, a for-loop was used to iterate through every token present in the list and on each token a .lower() function was used so to convert the token to a lowercase.

# In[9]:


#The below block of code is the sub-task 3 which is to convert all words into lowercase
lowercase_rev=[] #a list for storing the tokenized reviews which are converted to lowercase.
for tk_list in tokenized_review: #the for-loop iterates through the tokenized_review list which was created above.
    lwcase_token=[tk.lower() for tk in tk_list] #a new list is created which contains the lowercase version of each token
    lowercase_rev.append(lwcase_token) #the list of lowercase tokens for each review is appended to the above created list.


# In[10]:


#Not Required Line---> just for checking the output
for i in range(2): #the for-loop is used to iterate through the first 2 list of the lowercase tokenized reviews
    print(f"Lowercase Tokens of Review {i+1}: {lowercase_rev[i]}") #Prints the list of lowercase tokenized reviews


# According to the above output, all the tokens were converted to lowercase using the .lower() function on each of the tokens present in the list.
# 
# ### 3) Removing words with length less than 2
# - The len() function was used to run a check on the length of each tokens and the tokens having length less than 2 were removed and tokens having length greater than equal to 2 were kept as it is.

# In[11]:


#Subtask 4 removing words with a length less than 2
filter_rev=[] #A list is created for storing only those tokens which have length greater than 2
for tk_list in lowercase_rev: #iterates through every token in lowercase_rev which contains tokenized words in lowercase
    fil_tk=[tk for tk in tk_list if len(tk)>=2] #filters the tokens which have length greater than 2
    filter_rev.append(fil_tk) #the lists of each review are then appended to the above created list.


# In[12]:


#Not required --> just for checking if the operation is performed correctly.
for i in range(2): #the for-loop is used to iterate through the first 2 list of the tokenized reviews that have length greater than 2.
    print(f"Filtered tokens of Review {i+1}: {filter_rev[i]}") #Prints the list of filtered out tokenized reviews


# It can be seen from the above output that the tokens having length less than 2 were removed and the filtered tokens were kept as it is. An if-statement was used to run a check on the length of the tokens.
# 
# ### 4) Removing Stopwords
# - The stopwords were provided in a text file named as stopwords_en.txt and those words were extracted by reading the .txt file using the 'with' keyword. Once these stopwords were obtained the token list was checked if there are any stopwords present in that list and if found, they were removed.

# In[13]:


#Subtask 5 removing stopwords from the provided stopwords list
with open('stopwords_en.txt','r') as f: #opens the stopwords list and to access it 'f' is defined.
    stop_words=[eword.strip() for eword in f] #for each word in the stopword list, extra spaces are removed using .split() and the cleaned word is stored.
stop_remove_tk=[] #list for storing the tokens after removing the stopwords.
for tk_list in filter_rev: #the for-loop iterates through the filtered tokens having length greater than 2
    filter_tk=[tk for tk in tk_list if tk not in stop_words] #the tokens are filtered out after removing the tokens that are also present in the stopwords_en.txt file.
    stop_remove_tk.append(filter_tk) #the reviews are appended to the list from which the stopwords are removed.


# In[14]:


for i in range(2): #the for-loop is used to iterate through the first 2 list of the tokenized reviews after removing the stopwords.
    print(f"Review {i+1} after removing stop-words: {stop_remove_tk[i]}") #Prints the tokenized reviews from which the stopwords are removed


# The stopwords present in the stopwords_en.txt file were filtered out from the list of tokens as you can see from the above output.
# 
# ### 5) Term Frequency and removing words that appear only once in the document collection
# - A for-loop was used to iterate through every token present in the list, once the token was encountered it's term frequency was calculated based on how many times each word appeared across all reviews. The calculated term frequency of every token was stored in a dictionary.

# In[15]:


term_freq={} #variable for storing the frequency of each word. So the dict will look like word:word_count
for tk_list in stop_remove_tk: #iterates through the list of filtered out tokens i.e tokens after removing the stopwords.
    for tk in tk_list: #the for-loop iterates through each individual token within the current token list.
        term_freq[tk]=term_freq.get(tk,0)+1 #increaments the count of the token if the token is already present in the term_freq. Bydefault at the start it is 0.

once_word={word for word, freq_count in term_freq.items() if freq_count==1}  #this is infrequent words i,e words that appear only once

freq_filter_tk=[] #List for storing tokens after filtering out based on their term frequency. Only those tokens that have term frequency greater than 1 are stored.
for tk_list in stop_remove_tk: #the for-loop iterates through each list of tokens.
    freq_tok=[tk for tk in tk_list if tk not in once_word] #filters out and stores only those tokens that have term frequency greater than 1 i.e. tokens having term frequency=1 are removed.
    freq_filter_tk.append(freq_tok) #the words that have freq greater than 1 are appended.


# In[16]:


for i in range(2): #the for-loop is used to iterate through the first 2 reviews after filtering out them based on their term frequency
    print(f"Review {i+1} Tokens (Frequency Filtered): {freq_filter_tk[i]}") #Prints the tokenized reviews from which the words having term frequency only 1 are removed.


# From the calculated term frequency, tokens appearing only once in the document were removed from the token list and the rest of the token list was displayed as an output above.
# 
# ### 6) Document Frequency and removing the Top 20 most frequent words based on document frequency.
# 
# - The document frequency was calculated for each word, i.e. counting how many reviews contained that word. Then, the top 20 most frequent words based on this document frequency were identified using the .most_common() method. These top 20 words were removed from each review's list of tokens as required.

# In[17]:


doc_freq=FreqDist() #Creates a Frequency Distribution object to calculate document frequency of each word.
for tk_list in freq_filter_tk: #iterates through the list of frequency filtered tokens i.e after the term frequency operation
    unique_tk=set(tk_list) #the set of unique tokens within the current review.
    for tk in unique_tk: #iterates through each unique token.
        doc_freq[tk]+=1 #increaments the document frequency count for the current token.
        
most_freq=doc_freq.most_common(20) #exacts the top 20 most common words depending on their document frequency using the .most_common() function
top_freq_tk={word for word, freq in most_freq} #a set is created containing only the words from the top 20 most frequent words based on document frequency.
print("\nDocument Frequency of the top 20 most frequent words:")
for word, frequency in most_freq: #Iterates through the top 20 most common words along with their document frequency.
    print(f"{word}: {frequency}") #prints every word and it's document frequency


# The top 20 most frequent words are displayed above. These words are to be removed from every review's list of token, which is done in the below block of code.

# In[18]:


#Removing the Top 20 most frequent words
doc_filter_tk=[] #A list is created to store tokens after removing the top 20 most frequent words.
for tk_list in freq_filter_tk: #the for-loop iterates through each list of tokens
    filtered_tk=[tk for tk in tk_list if tk not in top_freq_tk] #creates a new list after removing the top 20 most frequent words
    doc_filter_tk.append(filtered_tk) #each list representing a review is appended to the above declared list.


# In[19]:


#doc_filter_tk contains the newly filtered words i.e. top 20 most frequent words removed based on document frequency.
for i in range(2): #the for loop iterates through the first 2 tokenized reviews from which the top 20 most frequent words are removed.
    print(f"Review {i+1} Tokens (Document Filtered): {doc_filter_tk[i]}") #Prints the first 2 review tokens.


# From the calculated document frequency, the top 20 most common words were removed from the token list and the rest of the token list was displayed as an output above.

# ## Saving required outputs
# 
# Once the reviews were processed, a vocabulary is created in the below block of code in the required format. This vocabulary is stored under the file named as vocab.txt.

# In[20]:


vocab=set() #a set is initialized so as to create the vocabulary
for tk_list in doc_filter_tk: #the for loop iterates through each list of tokenized reviews after performing all the pre-processing steps
    vocab.update(tk_list) #adds all the tokens from the current list to the set of vocabulary words. duplicates are automatically handled
    
#Converts the set to a sorted list
sort_vocab=sorted(list(vocab)) #converts the 'vocab' set into a list and then sorts it alphabetically using the sorted() function

word_line_vocab=[] #creates the content for the file vocab.txt in the format word_string:word_int_index
for word_int_index,word_string in enumerate(sort_vocab): #the for-loop iterates through the sorted vocabulary accessing both the index and the word.
    word_line_vocab.append(f'{word_string}:{word_int_index}') #a string is created in the desired format word_string:word_int_index and appends it to the above created list i.e. the content for the vocab.txt file
    
with open('vocab.txt','w') as f: #the file vocab.txt is opened in write mode
    for line in word_line_vocab: #the for-loop iterates through each line in the word_line_vocab that contains the content in the desired format.
        f.write(line+'\n') #writes the line to the vocab.txt to the file, which is followed by a newline


# The created tokens were joined so as to create one whole string and all the lists were stored under a new column called 'Processed Review Text' which contained the processed reviews. This 'Processed Review Text' along with the rest of the dataset was then stored under a new file named as 'processed.csv'.

# In[21]:


process_rev_text=[' '.join(tk) for tk in doc_filter_tk] #Joins the tokens back into the strings so as to store them into 'Processed Review Text' column in the processed.csv
rev_data['Processed Review Text']=process_rev_text #Adds the joined token to the dataframe under the column Processed Review Text
rev_data.to_csv('processed.csv',index=False) #Saves the dataframe under the name processed.csv as required.


# ## Summary
# This task involved a series of text pre-processing steps on 'Review Text' column of the provided dataset. These steps included tokenization using the provided regex pattern, converting tokens to lowercase, filtering tokens having length less than 2, removing stopwords, and the removal of both least frequent(term frequency equal to 1) and the top 20 most frequent words based on document frequency. The final processed reviews was saved to the processed.csv file and the resulting vocabulary was saved under the name vocab.txt.
