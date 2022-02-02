# IBM-Project-Debater-German

# Steps for running a german model
RUN main.py to process the german model
---------------------------------------
## 1. Extracted the sentences containing our topic plus patterns
There are 3 parameters:
1.dc_path: The path of dataframe of the debate concepts (DC)
2.corpus_path: The path of folder of our german corpus. There are about 25000 txt files in the folder. The corpus can be found at 
3.extracted_sentence_path: Where to save the dictionary of all the extracted sentences
## 2. Extracted expansion concepts (EC) from previous extraceted sentences
There are 2 parameters:
1. extracted_sentence_path: The dictionary saved from previous step
2. extracted_topic_pairs_path: A dataframe of where to save the dictionary of all the extracted topic pairs (DC and EC)
## 3. Filter
There are 4 parameters:
1.embedding: Which word embedding is being used, fasttext, spacy, or statified. 
2.extracted_topic_pairs_path: A dataframe of topic pairs (DC and EC)that are extracted in the previous step.
3.frequency_dict: A dictionary of words and their occurrences in the corpus. It's conventient and timesaving to save them as dictionary for later use.
4.corpus_path: The path of folder of our german corpus. There are about 25000 txt files in the folder.
## 4. Get Feature
There are 1 parameter:
1.sentiment_path: A dictionary of words and their sentiment. It's conventient and timesaving to save them as dictionary for later use.
## 4. Training
There are 4 parameter:
lang: The language of the training
label_de: expansion_label_de.csv
label_en: expansion_label_en.csv
model: Which classification model is being used, logistic regression or decision trees.
