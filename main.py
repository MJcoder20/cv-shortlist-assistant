import streamlit as sl

from Resume import Resume
from evaluation.similarity import calculate_similarity, filter_by_threshold
from processing.fileProcessing import (
    extract_text_from_pdf,
    extract_text_from_docx,
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
from testing.test_File import test_pdfFile_parsing, test_docxFile_parsing, test_embedding, test_similarity, \
    test_threshold, \
    test_request, test_extraction, test_shortlisting, test_analysis


def main():
    sl.title(":memo: Resume Matcher")
    # Files uploader
    job_text = sl.text_input("Job Description")
    resume_files = sl.file_uploader(
        "Upload Resumes", type=["pdf", "docx"], accept_multiple_files=True
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
                resumes = []
                for f in resume_files:
                    if f.name.endswith(".pdf"):
                        text = extract_text_from_pdf(f)
                    elif f.name.endswith(".docx"):
                        text = extract_text_from_docx(f)
                    else:
                        sl.error(f"Unsupported resume file format: {f.name}")
                        return
                    if not text:
                        sl.error("No valid resume texts found.")
                        return
                    else:
                        resumes.append(Resume(text))

                # Generating Embedding for the job description and resumes
                embedder = EmbeddingGenerator()
                job_embedding = embedder.generate(job_text)
                for i, resume in enumerate(resumes):
                    emb = embedder.generate(resume.text)
                    resume.embedding = emb
                    # Generating cosine similarity of each resume using the generated embedding
                    # for the job description and the resume
                    resume.similarity = calculate_similarity(job_embedding, emb)

                # returns sorted shortlisted resumes based on a threshold value compared
                # with the cosine similarity of each resume
                resumes = filter_by_threshold(resumes, 0.5450)

                # Extracting key features from the resumes
                resume_features = [
                    make_request(extract_info(
                        resumes,
                        resume.text,
                        job_text,
                        RESUME_PROMPT,
                        min_experience,
                        required_skills,
                        education_level,

                    ))
                    for resume in resumes
                ]

                # output the extracted features and cosine similarity for each resume
                sl.write("### Extracted Resume Features:")
                resumes_f = []
                i = 1
                for features, res in zip(resume_features, resumes):
                    sl.write(f"Resume {i}:")
                    sl.write(features)
                    sl.write(f"Cosine Similarity: {res.similarity}")
                    resume = f"Resume {i}: {features}\nCosine Similarity: {res.similarity}"
                    resumes_f.append(resume)
                    i = i + 1

                # shortlist the resumes into top 5
                short = make_request(shortlist(RESUME_PROMPT2, resumes_f))
                # Analyze top candidates' strengths and weaknesses, then conclude the best resume
                analysis = make_request(final_analysis(RESUME_PROMPT3, job_text, short))

                sl.write(short)  # output shortlist
                sl.write(analysis)  # output analysis

            except Exception as e:
                sl.error(f"An error occurred: {e}")
        else:
            sl.warning("Please upload a job description and at least one resume.")


if __name__ == "__main__":
    main()
    # Testing functions
    # test_pdfFile_parsing()
    # test_docxFile_parsing()
    # test_embedding()
    # test_similarity()
    # test_threshold()
    # test_request()
    # test_extraction()
    # test_shortlisting()
    # test_analysis()

