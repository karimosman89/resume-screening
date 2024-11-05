import pandas as pd

def create_synthetic_labels(resumes):
    """
    Generates synthetic labels for resumes.
    Label 1 for 'relevant' and 0 for 'not relevant'.
    """
    labels = [1 if "data science" in resume.lower() or "machine learning" in resume.lower() else 0 for resume in resumes]
    return labels

def load_job_description(job_desc_path):
    """
    Load a job description file and return preprocessed text.
    """
    with open(job_desc_path, "r", encoding="utf-8") as file:
        job_desc = file.read()
    return job_desc

