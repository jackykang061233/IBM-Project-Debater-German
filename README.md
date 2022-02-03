# IBM-Project-Debater-German

# Steps for running a german model
RUN main.py to process the german model
---------------------------------------
## 1. Extracted the sentences containing our topic plus patterns
There are 3 parameters:
1. dc_path: The path of dataframe of the debate concepts (DC). See topic.csv in folder example files
2. corpus_path: The path of folder of our german corpus. There are about 25000 txt files in the folder. The link of the corpus can be found at https://www.dropbox.com/sh/tq44pwn7thnie30/AADuO2rfcAHg0No0ZwqpWoB9a?dl=0
3. extracted_sentence_path: Where to save the dictionary of all the extracted sentences
## 2. Extracted expansion concepts (EC) from previous extraceted sentences
There are 2 parameters:
1. extracted_sentence_path: The dictionary saved from previous step
2. extracted_topic_pairs_path: A dataframe of where to save the dictionary of all the extracted topic pairs (DC and EC)
## 3. Filter
There are 4 parameters:
1. embedding: Which word embedding is being used, fasttext, spacy, or statified. 
2. extracted_topic_pairs_path: A dataframe of topic pairs (DC and EC)that are extracted in the previous step.
3. frequency_dict: A dictionary of words and their occurrences in the corpus. It's conventient and timesaving to save them as dictionary for later use. See final_frequency.pkt in folder example files
4. corpus_path: The path of folder of our german corpus. There are about 25000 txt files in the folder. The link of the corpus can be found at https://www.dropbox.com/sh/tq44pwn7thnie30/AADuO2rfcAHg0No0ZwqpWoB9a?dl=0
And the fasttext embedding can be found at https://fasttext.cc/docs/en/crawl-vectors.html
## 4. Get Feature
There are 1 parameter:
1. sentiment_path: A dictionary of words and their sentiment. It's conventient and timesaving to save them as dictionary for later use. See sentiment_de.pkt in folder example files
## 5. Training
There are 4 parameters:
1. lang: The language of the training
2. label_de: See expansion_label_de.csv in folder example files
3. label_en: See expansion_label_en.csv in folder example files
4. model: Which classification model is being used, logistic regression or decision trees.


