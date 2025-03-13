from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def calculate_similarity(job_embedding, resume_embeddings):
    """
    Calculate cosine similarity between job embedding and resume embeddings.

    Args:
        job_embedding (np.array): Embedding of the job description.
        resume_embeddings (list or np.array): List of embeddings for resumes.

    Returns:
        list: Similarity scores for each resume.
    """
    # Ensure inputs are NumPy arrays
    if isinstance(job_embedding, list):
        job_embedding = np.array(job_embedding)
    if isinstance(resume_embeddings, list):
        resume_embeddings = np.array(resume_embeddings)

    # Reshape embeddings to 2D if necessary
    if job_embedding.ndim == 1:
        job_embedding = job_embedding.reshape(1, -1)
    if resume_embeddings.ndim == 1:
        resume_embeddings = resume_embeddings.reshape(1, -1)

    # Calculate cosine similarity
    similarities = cosine_similarity(job_embedding, resume_embeddings)

    # Flatten and return as a list
    return similarities.flatten().tolist()


# now we want to filter the resumes using a specific threshold
def filter_by_threshold(similarities, threshold=0.7):
    return [
        i for i, score in enumerate(similarities) if score >= threshold
    ]  # the output here is a list of indices that have equal or greater threshold
