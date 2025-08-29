"""
Streamlit web app for Resume-Job Matching
"""

import streamlit as st
import joblib
import os
import sys
import numpy as np
import pandas as pd

# Support running via Streamlit without a package context
try:
    from .preprocess import add_skill_overlap_features
    from .models import predict
except Exception:
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(pkg_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from task8_resume_screening.preprocess import add_skill_overlap_features
    from task8_resume_screening.models import predict


@st.cache_resource
def load_model(model_dir: str):
    """Load the trained model components"""
    try:
        transformer = joblib.load(os.path.join(model_dir, "transformer.joblib"))
        estimator = joblib.load(os.path.join(model_dir, "estimator.joblib"))
        return transformer, estimator
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None


def main():
    st.set_page_config(
        page_title="Resume-Job Matcher",
        page_icon="📋",
        layout="wide"
    )
    
    st.title("📋 Resume-Job Matching System")
    st.markdown("---")
    
    # Load model
    model_dir = "NLP_Tasks_Project/task8_model"
    transformer, estimator = load_model(model_dir)
    
    if transformer is None or estimator is None:
        st.error("❌ Model not found. Please run training first.")
        st.code("python -m NLP_Tasks_Project.task8_resume_screening.cli train --csv 'path/to/dataset.csv'")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Input sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Job Description")
        job_description = st.text_area(
            "Enter the job description:",
            height=300,
            placeholder="Paste the job description here..."
        )
    
    with col2:
        st.subheader("👤 Resume/CV")
        resume_text = st.text_area(
            "Enter the resume text:",
            height=300,
            placeholder="Paste the resume text here..."
        )
    
    # Match button
    if st.button("🔍 Analyze Match", type="primary", use_container_width=True):
        if not job_description.strip() or not resume_text.strip():
            st.warning("Please enter both job description and resume text.")
            return
        
        # Prepare data
        df = pd.DataFrame([{
            "job_description": job_description,
            "resume": resume_text,
            "match_score": 0
        }])
        
        # Add features
        df = add_skill_overlap_features(df)
        
        # Predict
        try:
            preds = predict((transformer, estimator), df)
            score = float(np.clip(preds[0], 1.0, 5.0))
            rounded_class = int(round(score))
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Matching Results")
            
            # Score display
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Match Score", f"{score:.2f}/5.0")
            
            with col2:
                st.metric("Match Class", f"{rounded_class}/5")
            
            with col3:
                # Color-coded match level
                if score >= 4.5:
                    st.success("🎯 Excellent Match")
                elif score >= 3.5:
                    st.info("✅ Good Match")
                elif score >= 2.5:
                    st.warning("⚠️ Moderate Match")
                else:
                    st.error("❌ Poor Match")
            
            # Skill overlap analysis
            st.subheader("🔧 Skill Analysis")
            jd_skills = df.iloc[0]["jd_skills"].split(";") if df.iloc[0]["jd_skills"] else []
            resume_skills = df.iloc[0]["resume_skills"].split(";") if df.iloc[0]["resume_skills"] else []
            overlap = df.iloc[0]["skill_overlap"]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Job Skills:**")
                if jd_skills:
                    for skill in jd_skills:
                        st.write(f"• {skill}")
                else:
                    st.write("No specific skills detected")
            
            with col2:
                st.write("**Resume Skills:**")
                if resume_skills:
                    for skill in resume_skills:
                        st.write(f"• {skill}")
                else:
                    st.write("No specific skills detected")
            
            with col3:
                st.write("**Skill Overlap:**")
                st.metric("Matching Skills", overlap)
                if overlap > 0:
                    st.write("**Common Skills:**")
                    common = set(jd_skills) & set(resume_skills)
                    for skill in common:
                        st.write(f"• {skill}")
                else:
                    st.write("No skill overlap detected")
                    
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **How to interpret scores:**
    - **5**: Excellent match - Strong skill alignment
    - **4**: Good match - Minor skill differences  
    - **3**: Moderate match - Some relevant skills
    - **2**: Weak match - Limited skill overlap
    - **1**: Poor match - Minimal relevance
    """)


if __name__ == "__main__":
    main()
