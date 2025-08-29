# NLP Tasks Project

This project implements 8 comprehensive NLP tasks covering various aspects of Natural Language Processing, from basic sentiment analysis to advanced transformer-based models.

## Setup Instructions

1. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

2. **Download spaCy Models:**
```bash
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg
```

3. **Download NLTK Data:**
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
```

## Tasks Overview

### Task 1: Sentiment Analysis on Product Reviews
- **File:** `task1_sentiment_analysis.ipynb`
- **Description:** Binary sentiment classification using TF-IDF and logistic regression
- **Dataset:** IMDb Reviews or Amazon Product Reviews
- **Features:** Text preprocessing, vectorization, model comparison, visualization

### Task 2: News Category Classification
- **File:** `task2_news_classification.ipynb`
- **Description:** Multiclass classification of news articles into categories
- **Dataset:** AG News Dataset
- **Features:** Multiple classifiers, category-wise word analysis, neural networks

### Task 3: Fake News Detection
- **File:** `task3_fake_news_detection.ipynb`
- **Description:** Binary classification to detect fake vs real news
- **Dataset:** Fake and Real News Dataset
- **Features:** Advanced preprocessing, SVM, word cloud visualization

### Task 4: Named Entity Recognition
- **File:** `task4_ner.ipynb`
- **Description:** Extract named entities from news articles
- **Dataset:** CoNLL-2003 or custom news articles
- **Features:** spaCy NER, entity visualization, model comparison

### Task 5: Topic Modeling
- **File:** `task5_topic_modeling.ipynb`
- **Description:** Discover hidden topics using LDA and NMF
- **Dataset:** BBC News Dataset
- **Features:** Interactive visualizations, topic comparison, word distributions

### Task 6: Question Answering with Transformers
- **File:** `task6_qa_transformers.ipynb`
- **Description:** Build QA system using BERT and DistilBERT
- **Dataset:** SQuAD v1.1
- **Features:** Model fine-tuning, evaluation metrics, interactive interface

### Task 7: Text Summarization
- **File:** `task7_text_summarization.ipynb`
- **Description:** Generate summaries using T5, BART, and Pegasus
- **Dataset:** CNN-DailyMail
- **Features:** Abstractive and extractive summarization, ROUGE evaluation

### Task 8: Resume Screening
- **File:** `task8_resume_screening.ipynb`
- **Description:** Match resumes to job descriptions using embeddings
- **Dataset:** Resume and Job Dataset
- **Features:** Semantic similarity, skill extraction, ranking system

## Project Structure
```
NLP_Tasks_Project/
├── requirements.txt
├── README.md
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── visualization.py
│   └── evaluation.py
├── data/
│   └── (datasets will be downloaded here)
└── notebooks/
    ├── task1_sentiment_analysis.ipynb
    ├── task2_news_classification.ipynb
    ├── task3_fake_news_detection.ipynb
    ├── task4_ner.ipynb
    ├── task5_topic_modeling.ipynb
    ├── task6_qa_transformers.ipynb
    ├── task7_text_summarization.ipynb
    └── task8_resume_screening.ipynb
```

## Getting Started

1. Start with Task 1 (Sentiment Analysis) as it covers fundamental concepts
2. Progress through tasks in order as complexity increases
3. Each notebook is self-contained with detailed explanations
4. Utility functions are provided in the `utils/` directory for reuse across tasks

## Key Learning Outcomes

- Text preprocessing and cleaning techniques
- Feature extraction methods (TF-IDF, Word2Vec, BERT embeddings)
- Classical ML algorithms for NLP (Logistic Regression, SVM, Random Forest)
- Deep learning approaches (LSTM, Transformers)
- Evaluation metrics for different NLP tasks
- Visualization techniques for text data
- Real-world application deployment considerations
