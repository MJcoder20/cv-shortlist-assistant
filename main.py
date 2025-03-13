import streamlit as sl

from embeddings.similarity import calculate_similarity, filter_by_threshold
from processing.fileProcessing import (
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_csv,
)
from services.ollama import (
    RESUME_PROMPT,
    RESUME_PROMPT2,
    RESUME_PROMPT3,
    make_request,
    extract_info,
    shortlist,
    final_analysis,
)
from embeddings.embeddingGen import EmbeddingGenerator


def main():
    sl.title(":memo: Resume Matcher")

    # Files uploader
    job_text = sl.text_input("Job Description")
    resume_files = sl.file_uploader(
        "Upload Resumes", type=["pdf", "docx", "txt", "csv"], accept_multiple_files=True
    )

    # User-defined criteria
    sl.write("### Evaluation Criteria")
    required_skills = sl.text_input(
        "Required skills (comma-separated)", "Python, Machine Learning"
    )
    min_experience = sl.number_input(
        "Minimum years of experience required", min_value=0, value=3
    )
    education_level = sl.selectbox("Required Education Level", ["Bachelor's Degree", "Master's Degree",
                                                                "PHD Degree", "None"])

    # Submit button
    if sl.button("Submit"):
        if job_text and resume_files:
            try:
                # Validate job text
                if not job_text.strip():
                    sl.error("The job description file is empty.")
                    return

                # Extract text from resumes
                sl.write("Extracting text from resumes...")
                resume_texts = []
                for f in resume_files:
                    if f.name.endswith(".pdf"):
                        resume_texts.append(extract_text_from_pdf(f))
                    elif f.name.endswith(".docx"):
                        resume_texts.append(extract_text_from_docx(f))
                    elif f.name.endswith(".csv"):
                        resume_texts.append(extract_text_from_csv(f))
                    elif f.name.endswith(".txt"):
                        resume_texts.append(f.read().decode("utf-8"))
                    else:
                        sl.error(f"Unsupported resume file format: {f.name}")
                        return

                # Validate resume texts
                if not any(resume_texts):
                    sl.error("No valid resume texts found.")
                    return

                embedder = EmbeddingGenerator()
                job_embedding = embedder.generate(job_text)
                resume_embeddings = [embedder.generate(text) for text in resume_texts]
                similarities = calculate_similarity(job_embedding, resume_embeddings)
                indices = filter_by_threshold(similarities, 0.50)

                resume_features = [
                    make_request(extract_info(
                        resume_texts[index],
                        text,
                        job_text,
                        RESUME_PROMPT,
                        min_experience,
                        required_skills,
                        education_level,

                    ))
                    for index, text in zip(indices, resume_texts)
                ]

                sl.write("### Extracted Resume Features:")
                resumes = []
                i = 1
                for features, similarity in zip(resume_features, similarities):
                    sl.write(f"Resume {i}:")
                    sl.write(features)
                    sl.write(f"Cosine Similarity: {similarity}")
                    resume = f"Resume {i}: {features}\nCosine Similarity: {similarity}"
                    resumes.append(resume)
                    i = i + 1

                # # Create an array from resumes list
                # array = np.array(resumes)
                # # Sort using mergesort
                # sorted_resumes = np.sort(array, kind='mergesort')

                short = make_request(shortlist(RESUME_PROMPT2, resumes))
                analysis = make_request(final_analysis(RESUME_PROMPT3, job_text, short))
                sl.write(short)
                sl.write(analysis)

            except Exception as e:
                sl.error(f"An error occurred: {e}")
        else:
            sl.warning("Please upload a job description and at least one resume.")


if __name__ == "__main__":
    main()
