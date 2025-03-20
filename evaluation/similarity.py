from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def calculate_similarity(job_embedding, resume_embedding):
    # Ensure inputs are NumPy arrays
    if isinstance(job_embedding, list):
        job_embedding = np.array(job_embedding)
    if isinstance(resume_embedding, list):
        resume_embedding = np.array(resume_embedding)

    # Reshape embeddings to 2D if necessary
    if job_embedding.ndim == 1:
        job_embedding = job_embedding.reshape(1, -1)
    if resume_embedding.ndim == 1:
        resume_embedding = resume_embedding.reshape(1, -1)

    # Calculate cosine similarity
    similarity = cosine_similarity(job_embedding, resume_embedding)
    return similarity[0][0]


def filter_by_threshold(resumes, threshold=0.7000):

    # Validate that threshold is a number
    if not isinstance(threshold, (int, float)):
        raise ValueError("The 'threshold' parameter must be an integer or float.")

    return sorted([
        resume for i, resume in enumerate(resumes) if resume.similarity >= threshold
    ], key=lambda resume: resume.similarity)


