# NLP Disaster Tweet Classification

This project aims to classify tweets related to disasters using Natural Language Processing (NLP) techniques. The dataset contains tweets labeled as either disaster-related (1) or not (0). 

## 📂 Dataset

The dataset consists of:
- **Text:** Raw tweets from various sources.
- **Target:** Binary label indicating whether a tweet is disaster-related (1) or not (0).

## 🔧 Preprocessing

To prepare the data, the following steps were applied:
1. **Tokenization:** Splitting text into words.
2. **Lowercasing:** Converting text to lowercase.
3. **Removing Stopwords & Punctuation:** Eliminating unnecessary words and symbols.
4. **Lemmatization:** Converting words to their root forms.
5. **Vectorization:** Transforming text into numerical representations using TF-IDF or Word Embeddings.

## 🏗️ Model Training

Several machine learning and deep learning models were explored:
- Logistic Regression
- Support Vector Machines (SVM)
- Random Forest
- LSTM (Long Short-Term Memory)
- BERT (Bidirectional Encoder Representations from Transformers)

## 📊 Evaluation

Models were evaluated using:
- **Accuracy**
- **Precision, Recall, F1-score**
- **Confusion Matrix**

## 🚀 Future Improvements

- Experimenting with transformer-based models like **RoBERTa** and **DistilBERT**.
- Implementing **hyperparameter tuning** for better performance.
- Using **data augmentation** techniques to improve classification on minority samples.

## 📌 How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   
python preprocess.py

python train.py

python evaluate.py


Let me know if you need any modifications! 
