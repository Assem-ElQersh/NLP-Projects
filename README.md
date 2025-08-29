# 🧠 NLP Tasks Project - Elevvo Internship

## 📋 Project Overview

This repository contains the complete implementation of 8 Natural Language Processing tasks completed during a 1-month internship at Elevvo. Each task demonstrates different aspects of NLP, from basic text preprocessing to advanced transformer-based models.

## 🎯 Tasks Implemented

### Task 1: Sentiment Analysis on Product Reviews
- **Dataset**: IMDB Reviews Dataset
- **Goal**: Classify movie reviews as positive/negative
- **Techniques**: TF-IDF, Count Vectorization, Logistic Regression, Naive Bayes, SVM
- **Results**: Achieved high accuracy with multiple classification algorithms
- **File**: `NLP_Tasks_Project/task1_sentiment_analysis.ipynb`

### Task 2: News Category Classification (Multiclass)
- **Dataset**: AG News Dataset
- **Goal**: Classify news articles into categories (sports, business, politics, tech)
- **Techniques**: TF-IDF, N-grams, Multiclass classification (OvR strategy)
- **Results**: Effective categorization with visualization of word distributions
- **File**: `NLP_Tasks_Project/task2_news_classification.ipynb`

### Task 3: Fake News Detection
- **Dataset**: Fake and Real News Dataset
- **Goal**: Distinguish between real and fake news articles
- **Techniques**: Text preprocessing, TF-IDF, Multiple classifiers
- **Results**: High accuracy in detecting fake news with feature importance analysis
- **File**: `NLP_Tasks_Project/task3_fake_news_detection.ipynb`

### Task 4: Named Entity Recognition (NER)
- **Dataset**: CoNLL-2003 Dataset
- **Goal**: Identify named entities (people, locations, organizations)
- **Techniques**: spaCy NER, IOB tagging, CRF models
- **Results**: Effective entity extraction with visualization using displaCy
- **File**: `NLP_Tasks_Project/task4_ner.ipynb`

### Task 5: Topic Modeling
- **Dataset**: BBC News Dataset
- **Goal**: Discover hidden topics in news articles
- **Techniques**: LDA, NMF, pyLDAvis visualization
- **Results**: Identified coherent topics with word-topic distributions
- **File**: `NLP_Tasks_Project/task5_topic_modeling.ipynb`

### Task 6: Question Answering with Transformers
- **Dataset**: SQuAD v1.1 Dataset
- **Goal**: Answer questions from context using transformer models
- **Techniques**: BERT, DistilBERT, RoBERTa, Span extraction
- **Results**: High accuracy with interactive QA system
- **File**: `NLP_Tasks_Project/task6_qa_transformers.ipynb`

### Task 7: Text Summarization
- **Dataset**: CNN-DailyMail Dataset
- **Goal**: Generate abstractive summaries using pre-trained models
- **Techniques**: T5, BART, Pegasus, ROUGE evaluation
- **Results**: Effective summarization with ROUGE score evaluation
- **File**: `NLP_Tasks_Project/task7_text_summarization.ipynb`

### Task 8: Resume Screening Using NLP
- **Dataset**: Resume vs. Job Description Matching Dataset
- **Goal**: Match resumes to job descriptions using semantic similarity
- **Techniques**: TF-IDF, Ridge Regression, Skill extraction, Streamlit web app
- **Results**: R² = 0.624, MAE = 0.565 with interactive web interface
- **Files**: 
  - `NLP_Tasks_Project/task8_resume_screening/` (Python package)
  - `NLP_Tasks_Project/run_task8_app.py` (Web app launcher)

## 🏗️ Project Structure

```
NLP_Tasks_Project/
├── task1_sentiment_analysis.ipynb
├── task2_news_classification.ipynb
├── task3_fake_news_detection.ipynb
├── task4_ner.ipynb
├── task5_topic_modeling.ipynb
├── task6_qa_transformers.ipynb
├── task7_text_summarization.ipynb
├── task8_resume_screening/
│   ├── __init__.py
│   ├── data.py
│   ├── preprocess.py
│   ├── models.py
│   ├── evaluate.py
│   ├── cli.py
│   └── app.py
├── run_task8_app.py
├── task6_qa_summary.txt
├── task7_summarization_summary.txt
└── task8_resume_screening_summary.txt
```

## 🚀 Quick Start

### Prerequisites

1. **Python Environment**: Python 3.8+
2. **Conda Environment**: 
   ```bash
   conda create -n NLP python=3.8
   conda activate NLP
   ```

3. **Required Libraries**:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn nltk spacy
   pip install transformers torch datasets rouge-score gensim pyLDAvis
   pip install streamlit joblib
   ```

### Running the Tasks

#### Tasks 1-7 (Jupyter Notebooks)
Each task is implemented as a Jupyter notebook. Open and run the cells sequentially:

```bash
jupyter notebook NLP_Tasks_Project/
```

#### Task 8 (Resume Screening - Python Package)

**Training the Model**:
```bash
python -m NLP_Tasks_Project.task8_resume_screening.cli train \
  --csv "Resume vs. Job Description Matching Dataset/resume_job_matching_dataset.csv" \
  --out_dir "NLP_Tasks_Project/task8_model"
```

**Scoring Individual Pairs**:
```bash
python -m NLP_Tasks_Project.task8_resume_screening.cli score \
  --model_dir "NLP_Tasks_Project/task8_model" \
  --job "Job description text..." \
  --resume "Resume text..."
```

**Launching Web Interface**:
```bash
python NLP_Tasks_Project/run_task8_app.py
```
Then open http://localhost:8501 in your browser.

## 📊 Key Results Summary

| Task | Dataset | Model | Performance | Key Features |
|------|---------|-------|-------------|--------------|
| 1 | IMDB | Logistic Regression | High Accuracy | Sentiment classification |
| 2 | AG News | TF-IDF + SVM | Good multiclass performance | News categorization |
| 3 | Fake News | Multiple classifiers | High accuracy | Fake news detection |
| 4 | CoNLL-2003 | spaCy NER | Effective extraction | Named entity recognition |
| 5 | BBC News | LDA/NMF | Coherent topics | Topic modeling |
| 6 | SQuAD v1.1 | BERT/DistilBERT | High EM/F1 | Question answering |
| 7 | CNN-DailyMail | T5/BART/Pegasus | Good ROUGE scores | Text summarization |
| 8 | Resume-Job | TF-IDF + Ridge | R² = 0.624 | Resume screening |

## 🔧 Technical Implementation Details

### Common NLP Pipeline
1. **Data Loading**: CSV/JSON dataset loading with validation
2. **Text Preprocessing**: Cleaning, tokenization, lemmatization
3. **Feature Extraction**: TF-IDF, embeddings, custom features
4. **Model Training**: Various ML/DL approaches
5. **Evaluation**: Appropriate metrics for each task
6. **Visualization**: Results analysis and insights

### Advanced Features
- **Transformer Models**: BERT, DistilBERT, RoBERTa, T5, BART, Pegasus
- **Interactive Systems**: QA system, web interface
- **Evaluation Metrics**: Accuracy, F1, ROUGE, EM, MAE, RMSE
- **Visualization**: Word clouds, confusion matrices, topic distributions

## 🎨 Web Interface (Task 8)

The Resume Screening system includes a Streamlit web application with:
- **Dual Input Panels**: Job description and resume text areas
- **Real-time Analysis**: Instant match score prediction
- **Skill Extraction**: Automatic skill identification and overlap analysis
- **Visual Results**: Color-coded match levels and detailed breakdowns
- **User-friendly Interface**: Clean, responsive design

## 📈 Learning Outcomes

### Technical Skills Developed
- **Text Preprocessing**: Cleaning, normalization, feature engineering
- **Machine Learning**: Classification, regression, clustering
- **Deep Learning**: Transformer models, fine-tuning
- **Evaluation**: Metrics selection and interpretation
- **Visualization**: Data presentation and analysis
- **Web Development**: Streamlit app creation

### NLP Concepts Mastered
- **Text Classification**: Binary and multiclass problems
- **Named Entity Recognition**: Entity extraction and tagging
- **Topic Modeling**: Unsupervised topic discovery
- **Question Answering**: Span extraction and comprehension
- **Text Summarization**: Abstractive summarization
- **Semantic Similarity**: Document matching and ranking

## 🔍 Key Insights

### Model Performance
- **Traditional ML**: Effective for structured NLP tasks
- **Transformer Models**: Superior for complex language understanding
- **Feature Engineering**: Critical for model performance
- **Evaluation Metrics**: Task-specific metrics essential for assessment

### Practical Applications
- **Sentiment Analysis**: Customer feedback analysis
- **News Classification**: Content organization
- **Fake News Detection**: Information verification
- **NER**: Information extraction from documents
- **Topic Modeling**: Content discovery and organization
- **QA Systems**: Customer support automation
- **Summarization**: Content condensation
- **Resume Screening**: HR automation

## 🚀 Future Enhancements

### Potential Improvements
- **Multilingual Support**: Extend to other languages
- **Real-time Processing**: Optimize for production deployment
- **Advanced Models**: Experiment with newer transformer architectures
- **Custom Datasets**: Fine-tune on domain-specific data
- **API Development**: RESTful API for model serving
- **Scalability**: Handle larger datasets and concurrent users

### Advanced Features
- **Active Learning**: Interactive model improvement
- **Explainable AI**: Model interpretability
- **Multi-modal**: Combine text with other data types
- **Real-time Learning**: Continuous model updates

## 📚 Resources and References

### Datasets Used
- IMDB Reviews Dataset
- AG News Dataset
- Fake and Real News Dataset
- CoNLL-2003 Dataset
- BBC News Dataset
- SQuAD v1.1 Dataset
- CNN-DailyMail Dataset
- Resume vs. Job Description Matching Dataset

### Key Libraries
- **Core**: pandas, numpy, matplotlib, seaborn
- **ML**: scikit-learn, XGBoost
- **NLP**: NLTK, spaCy, Gensim
- **DL**: PyTorch, Transformers
- **Visualization**: pyLDAvis, displaCy
- **Web**: Streamlit
- **Evaluation**: rouge-score

## 👨‍💻 Author

**Assem ElQersh**  
Elevvo NLP Internship Participant  
Duration: 1 Month  
Year: 2025

## 📄 License

This project was completed as part of the Elevvo 1-month NLP internship program. All implementations are for educational and demonstration purposes.

---

## 🎉 Conclusion

This comprehensive NLP project demonstrates mastery of fundamental to advanced natural language processing techniques. From basic text classification to sophisticated transformer-based models, the implementation covers the full spectrum of modern NLP applications. The project showcases both theoretical understanding and practical implementation skills, making it a valuable portfolio piece for NLP and machine learning roles.

**Total Tasks Completed: 8/8**
**Project Status: Complete**
