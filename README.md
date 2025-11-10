# Hate Speech Detection using Machine Learning

This project detects **Hate Speech**, **Offensive Language**, and **Neutral / Normal** text in social media posts using **Natural Language Processing (NLP)** and **Machine Learning**.  

It uses a dataset of real tweets and classifies each message into one of three categories:
- 🟥 **0 — Hate Speech**
- 🟧 **1 — Offensive Language**
- 🟩 **2 — Neutral / Normal**


## ⚙️ Technologies Used
- **Python 3**
- **Pandas, NumPy** → Data manipulation  
- **NLTK** → Text preprocessing, lemmatization, stopwords removal  
- **Scikit-learn** → TF-IDF Vectorization, ML models  
- **Matplotlib, Seaborn** → Data visualization  


## Preprocessing Steps
- Convert text to lowercase  
- Remove URLs, mentions, hashtags, digits, and punctuation  
- Remove stopwords  
- Lemmatize words  
- Apply **TF-IDF Vectorizer** to extract features


## Machine Learning Models Used
| Model | Description | Accuracy 

 Logistic Regression | Linear classifier for text | ~95% 
 Multinomial Naive Bayes | Great for word frequency data | ~95% 
 Random Forest | Ensemble model for robustness | ~95% 
 Linear SVM | High-performing linear classifier | ~95% 


## Results
All models performed exceptionally well with accuracy around **94–96%**,  
with **Random Forest** and **Naive Bayes** giving slightly better results.

## Example Predictions
 Example Tweet | Prediction 

 "I hate those people!" | 🟥 Hate Speech 
 "You're such an idiot!" | 🟧 Offensive Language 
 "Have a great day!" | 🟩 Neutral / Normal 
 "All immigrants should leave!" | 🟥 Hate Speech 


## Project Flow
1. Load & preprocess dataset  
2. Clean text using regex + NLP  
3. Extract features using TF-IDF  
4. Train 4 ML models  
5. Evaluate performance  
6. Visualize results with confusion matrices  
7. Predict custom text inputs


## 🚀 How to Run the Project
```bash
pip install -r requirements.txt
jupyter notebook Hate_Speech_Detection.ipynb
