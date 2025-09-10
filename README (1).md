# PROJECT | Fake News Detection using Natural Language Processing (NLP)

## Project Overview
This project involves detecting fake news articles by classifying them into "real" or "fake" categories using machine learning. 
The dataset consists of news articles with their corresponding labels (real or fake), and the goal is to preprocess the data, build a classification model, and evaluate its performance. 
The project includes training various machine learning models, such as Logistic Regression, Naive Bayes, Random Forest, K-Nearest Neighbors, and XGBoost, and comparing their performance.

## Features
- Load and preprocess a news dataset containing article titles and content.
- Clean and preprocess text data (e.g., removing punctuation, converting text to lowercase, and removing extra spaces).
- Train multiple machine learning models for classification, including Logistic Regression, Naive Bayes, Random Forest, K-Nearest Neighbors (KNN), and XGBoost.
- Evaluate model performance using accuracy, precision, recall, F1-score, and confusion matrices.
- Visualize confusion matrices and classification reports.
- Deploy the best-performing model to classify news articles in a validation dataset and generate predictions.

## Installation
To run this project, ensure that Python is installed along with the required libraries. You can install the necessary dependencies using the following command:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost flask fastapi lime
```

## Usage
1. Open the Python script or Jupyter Notebook.
2. Run the cells to load and preprocess the news dataset (train-test split).
3. Preprocess the text data (cleaning and converting text to numerical features).
4. Train the classification models (Logistic Regression, Naive Bayes, Random Forest, K-Nearest Neighbors, XGBoost).
5. Evaluate the models based on accuracy, precision, recall, and F1-score.
6. Visualize the confusion matrices and classification reports for each model.
7. Deploy the best-performing model and predict labels for the validation dataset.
8. Save the predictions to a new CSV file.

## Dataset
The dataset for training the model is provided with columns:

- **`label`**: The true label (0 for fake news, 1 for real news).
- **`title`**: The headline of the news article.
- **`text`**: The full content of the article.

The validation dataset contains news articles without labels, and our task is to predict whether each article is real or fake.

## Model Training
- Use Logistic Regression, Naive Bayes, Random Forest, K-Nearest Neighbors, and XGBoost for classification.
- Implement training, validation, and testing processes using cross-validation.
- Evaluate the model performance using metrics such as accuracy, precision, recall, and F1-score.

## Model Deployment
- Deploy the trained model using Flask or FastAPI.
- Provide an API endpoint to classify new news articles as either real or fake.

## Results
- Display the accuracy and loss metrics for each model.
- Visualize confusion matrices and classification reports.
- Compare the performance of the models and choose the best one based on accuracy.
