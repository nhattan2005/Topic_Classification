# **Topic Classification using Machine Learning and Deep Learning**
===========================================================

## **Project Overview**
-------------------

This project focuses on classifying text into specific topics using Machine Learning and Deep Learning techniques. The dataset utilized is the [VNTC (Vietnamese News Topic Classification)](https://github.com/duyvuleo/VNTC).

## **Steps Undertaken**
----------------------

### 1. **Data Collection and Preprocessing**
- Utilized VNTC dataset for both training and testing.
- Text Preprocessing:
  - Converted all text to lowercase.
  - Removed special characters.
  - Tokenized text.
  - Removed Vietnamese stopwords.

### 2. **Modeling**
- **Word2Vec**:
  - Employed Gensim to create a Word2Vec model, converting text into vectors (vector_size=100).
- **Data Transformation**:
  - Transformed dataset into vectors using the Word2Vec model.

### 3. **Model Training**
- **Naive Bayes (NB)**:
  - Used Naive Bayes as a baseline algorithm.
- **Deep Neural Network (DNN) with Recurrent Neural Network (RNN)**:
  - Developed a DNN model utilizing an RNN layer for sequential data processing.
- **DNN with Long Short-Term Memory (LSTM)**:
  - Enhanced the DNN model with an LSTM layer for improved sequential data handling.
- **DNN with Gated Recurrent Unit (GRU)**:
  - Constructed a DNN model employing a GRU layer as an RNN variant.

### 4. **Performance Evaluation**
- Metrics Used: Accuracy, Precision, Recall, F1-score.
- Compared performance across models to identify the best topic classification approach.

## **Results**
------------

- The **LSTM** model achieved the highest accuracy of **0.8871**, demonstrating effective topic classification from text.

## **Technologies and Libraries**
------------------------------

- **Programming Language**: Python
- **Word Embeddings**: Gensim (Word2Vec)
- **Machine Learning**: Scikit-learn (Naive Bayes, Model Evaluation)
- **Deep Learning Framework**: TensorFlow/Keras (DNN, RNN, LSTM, GRU)

## **Notes**
----------

- This project can be expanded by experimenting with other models, advanced text preprocessing techniques, or applying to different NLP tasks.

## **Contributing**
------------

Contributions are welcome. Feel free to fork, modify, and submit pull requests.

## **Acknowledgments**
------------
Special thanks to the creators of the VNTC dataset.
