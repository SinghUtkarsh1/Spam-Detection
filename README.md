# Spam-Detection

Read file: spamdetector.ipynb

Read file: spamdetector.ipynb
Here is a detailed project description you can use for your README file:

---

## Project Description

This project is an end-to-end Email/SMS Spam Detection system that leverages Natural Language Processing (NLP) and machine learning to classify text messages as "Spam" or "Not Spam". The solution is designed to be both educational and practical, demonstrating the full workflow from data exploration to model deployment.

### Key Features

- **Data Exploration & Preprocessing:**  
  The project uses the popular "SMSSpamCollection" dataset, which contains thousands of labeled SMS messages. The data is thoroughly explored for class balance, missing values, and duplicates, and is then cleaned and preprocessed. Preprocessing steps include:
  - Lowercasing text
  - Tokenization
  - Removal of stopwords and punctuation
  - Lemmatization

- **Feature Engineering:**  
  Additional features such as message length, word count, and sentence count are extracted to enrich the dataset.

- **Vectorization:**  
  Text data is transformed into numerical features using TF-IDF vectorization, enabling machine learning algorithms to process the messages.

- **Model Training & Evaluation:**  
  Multiple machine learning models are trained and evaluated to identify the best-performing classifier for spam detection. The final model is selected based on accuracy and other relevant metrics.

- **Deployment with Streamlit:**  
  The trained model and vectorizer are saved and integrated into a user-friendly Streamlit web application (`spam.py`). Users can input any message and instantly receive a prediction on whether it is spam or not.

- **Reusable Artifacts:**  
  The project includes:
  - `spamdetector.ipynb`: A Jupyter notebook containing the full data science workflow, from data loading and EDA to model training and evaluation.
  - `spam.py`: The Streamlit app for real-time spam detection.
  - `model.pkl` and `vectorizer.pkl`: Serialized model and vectorizer for fast, consistent predictions.
  - `SMSSpamCollection.txt`: The dataset used for training and evaluation.

### How It Works

1. **Data is loaded and cleaned** to remove duplicates and handle imbalances.
2. **Text messages are preprocessed** using NLP techniques to normalize and prepare them for analysis.
3. **TF-IDF vectorization** converts text into a format suitable for machine learning.
4. **A classification model** (such as Naive Bayes, SVM, or Logistic Regression) is trained to distinguish between spam and non-spam messages.
5. **The model is deployed** in a Streamlit web app, allowing users to interactively test messages for spam detection.

### Use Cases

- Educational resource for learning about NLP and text classification.
- Practical tool for filtering spam in messaging systems.
- Template for deploying other text classification models.

