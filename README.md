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

1. **Install dependencies**  
   Run the following command to install the required Python libraries:
   ```bash
   pip install -r requirements.txt

### 2️⃣ Load and Prepare the Dataset
Upload the dataset to your Google Drive if using Colab. Otherwise, ensure the dataset is in your working directory.

```python
import pandas as pd

# Load the dataset
file_path = 'train.csv'  # Adjust the path if needed
data = pd.read_csv(file_path)

# Inspect the dataset
print(data.head())
print(data.info())
```

### 3️⃣ Preprocess the Data
Clean the text by removing links, special characters, and unnecessary spaces.

```python
import re
from transformers import BertTokenizer

# Function to clean the text data
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove links
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = text.lower().strip()  # Convert to lowercase
    return text

# Apply cleaning to the 'text' column
data['text'] = data['text'].apply(clean_text)
```

### 4️⃣ Tokenization using BERT
Use Hugging Face's tokenizer to convert text into tokenized inputs.

```python
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize(text):
    return tokenizer(text, padding="max_length", truncation=True, return_tensors="pt")

data['tokenized'] = data['text'].apply(tokenize)
```

### 5️⃣ Train the Model
Fine-tune a pre-trained BERT model.

```python
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch

# Split the data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    data['tokenized'].tolist(), data['target'].tolist(), test_size=0.2
)

# Convert to tensor dataset
train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_texts), torch.tensor(train_labels))
val_dataset = torch.utils.data.TensorDataset(torch.tensor(val_texts), torch.tensor(val_labels))

# Load BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
)

# Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
trainer.train()
```

### 6️⃣ Evaluate the Model
Check model performance using validation data.

```python
trainer.evaluate()
```

### 7️⃣ Save the Model
Save the trained model for later use.

```python
model.save_pretrained("saved_model")
tokenizer.save_pretrained("saved_model")
```

### 8️⃣ Load the Model for Testing
To test new tweets, load the saved model:

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="saved_model", tokenizer="saved_model")

# Example Prediction
example_tweet = "Earthquake hits California, many people displaced."
print(classifier(example_tweet))
```

### 9️⃣ Make Predictions on Test Data
Load test dataset and make predictions.

```python
test_data = pd.read_csv("test.csv")
test_data["cleaned_text"] = test_data["text"].apply(clean_text)

# Run predictions
predictions = [classifier(tweet)[0]['label'] for tweet in test_data["cleaned_text"]]

# Save predictions
test_data["predictions"] = predictions
test_data.to_csv("submission.csv", index=False)
```


