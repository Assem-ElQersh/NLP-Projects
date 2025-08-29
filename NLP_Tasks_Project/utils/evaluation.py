"""
Evaluation utilities for NLP tasks
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.model_selection import learning_curve
import warnings
warnings.filterwarnings('ignore')

def evaluate_classification(y_true, y_pred, y_pred_proba=None, labels=None, plot=True):
    """
    Comprehensive evaluation of classification models
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
        labels: Label names (optional)
        plot (bool): Whether to plot confusion matrix
        
    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    
    # Calculate basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Handle binary vs multiclass
    if len(np.unique(y_true)) == 2:
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        # ROC AUC if probabilities are provided
        roc_auc = None
        if y_pred_proba is not None:
            if y_pred_proba.ndim > 1:
                roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
            else:
                roc_auc = roc_auc_score(y_true, y_pred_proba)
    else:
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # ROC AUC for multiclass
        roc_auc = None
        if y_pred_proba is not None:
            roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Classification report
    class_report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    
    # Plot confusion matrix
    if plot:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'classification_report': class_report
    }
    
    return metrics


def plot_roc_curve(y_true, y_pred_proba, title='ROC Curve'):
    """
    Plot ROC curve for binary classification
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        title (str): Plot title
    """
    # Handle different probability formats
    if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] > 1:
        y_scores = y_pred_proba[:, 1]
    else:
        y_scores = y_pred_proba
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


def plot_learning_curves(estimator, X, y, cv=5, n_jobs=-1, 
                        train_sizes=np.linspace(0.1, 1.0, 10)):
    """
    Plot learning curves for a model
    
    Args:
        estimator: Sklearn estimator
        X: Features
        y: Labels
        cv (int): Cross-validation folds
        n_jobs (int): Number of jobs for parallel processing
        train_sizes: Training set sizes to use
    """
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.1, color='blue')
    
    plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation score')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                     alpha=0.1, color='red')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.title('Learning Curves')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


def compare_models(models_results, metric='f1_score'):
    """
    Compare multiple models performance
    
    Args:
        models_results (dict): Dictionary with model names as keys and results as values
        metric (str): Metric to compare
        
    Returns:
        pd.DataFrame: Comparison dataframe
    """
    comparison_data = []
    
    for model_name, results in models_results.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': results.get('accuracy', 0),
            'Precision': results.get('precision', 0),
            'Recall': results.get('recall', 0),
            'F1-Score': results.get('f1_score', 0),
            'ROC-AUC': results.get('roc_auc', 0) if results.get('roc_auc') is not None else 0
        })
    
    df = pd.DataFrame(comparison_data)
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    
    # Select metrics to plot (exclude ROC-AUC if all zeros)
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    if df['ROC-AUC'].sum() > 0:
        metrics_to_plot.append('ROC-AUC')
    
    x = np.arange(len(df))
    width = 0.15
    
    for i, metric in enumerate(metrics_to_plot):
        plt.bar(x + i*width, df[metric], width, label=metric)
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x + width * 2, df['Model'], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return df


def print_classification_summary(y_true, y_pred, model_name="Model"):
    """
    Print a formatted classification summary
    
    Args:
        y_true: True labels
        y_pred: Predicted labels  
        model_name (str): Name of the model
    """
    accuracy = accuracy_score(y_true, y_pred)
    
    if len(np.unique(y_true)) == 2:
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
    else:
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"\n{model_name} Performance:")
    print("=" * 40)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("=" * 40)


def calculate_class_distribution(y, labels=None):
    """
    Calculate and visualize class distribution
    
    Args:
        y: Labels
        labels: Label names (optional)
        
    Returns:
        pd.DataFrame: Class distribution
    """
    unique, counts = np.unique(y, return_counts=True)
    percentages = counts / len(y) * 100
    
    if labels is None:
        labels = [f'Class {i}' for i in unique]
    
    df = pd.DataFrame({
        'Class': labels,
        'Count': counts,
        'Percentage': percentages
    })
    
    # Plot distribution
    plt.figure(figsize=(10, 6))
    
    # Bar plot
    plt.subplot(1, 2, 1)
    plt.bar(df['Class'], df['Count'])
    plt.title('Class Distribution (Counts)')
    plt.xticks(rotation=45)
    
    # Pie chart
    plt.subplot(1, 2, 2)
    plt.pie(df['Percentage'], labels=df['Class'], autopct='%1.1f%%')
    plt.title('Class Distribution (Percentages)')
    
    plt.tight_layout()
    plt.show()
    
    return df
