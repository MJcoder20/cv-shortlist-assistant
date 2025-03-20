import os
import numpy

from Resume import Resume
from embeddings.embeddingGen import EmbeddingGenerator
from evaluation.similarity import calculate_similarity, filter_by_threshold
from processing.fileProcessing import extract_text_from_docx, extract_text_from_pdf
import pytest
import numpy as np

from services.ollama import make_request, RESUME_PROMPT, extract_info, RESUME_PROMPT2, shortlist, final_analysis, \
    RESUME_PROMPT3

RESUMES = []
SHORT = ""


def test_pdfFile_parsing():
    print("test_pdf_parsing")
    path = "testing/8.pdf"
    if not os.path.exists(path):
        pytest.fail(f"Missing test file: {path}")
    text = extract_text_from_pdf(path)
    print(f"Parsed text: {text}")
    assert "HTML" in text, f"Expected 'Python' in parsed text. Got: {text}"


def test_docxFile_parsing():
    path = "testing/11.docx"
    if not os.path.exists(path):
        pytest.fail(f"Missing test file: {path}")
    text = extract_text_from_docx(path)
    assert "Mechanical" in text, f"Expected 'HTML' in parsed text. Got: {text}"


def test_embedding():
    generator = EmbeddingGenerator()
    text = "Python developer with NLP experience"
    embedding = generator.generate(text)
    assert isinstance(embedding, np.ndarray), "Embedding should be a numpy array"
    assert embedding.shape == (768,), f"Expected shape (768,), got {embedding.shape}"


def test_similarity():
    path = "testing/11.docx"
    resume_text = extract_text_from_docx(path)
    generator = EmbeddingGenerator()
    text = """
        1. Develop and execute the company’s business strategies in order to attain the goals of 
        the board and shareholders 
        2. Provide strategic advice to the board and Chairperson so that they will have accurate 
        view of the market and the company’s future 
        3. Prepare and implement comprehensive business plans to facilitate achievement by 
        planning cost-effective operations and market development activities 
        4. Ensure company policies and legal guidelines are communicated all the way from the 
        top down in the company and that they are followed at all times 
        5. Communicate and maintain trust relationships with shareholders, business partners and 
        authorities 
        6. Oversee the company’s financial performance, investments and other business ventures 
        7. Delegate responsibilities and supervise the work of executives providing guidance and 
        motivation to drive maximum performance 
        8. Read all submitted reports by lower rank managers to reward performance, prevent 
        issues and resolve problems """
    job_embedding = generator.generate(resume_text)
    resume_embedding = generator.generate(text)
    similarity = calculate_similarity(job_embedding, resume_embedding)
    # Assert that the variable is a float
    assert isinstance(similarity, numpy.float32), "The variable is not a float!"
    # If the assertion passes, the program continues
    print(f"The similarity variable has a float value of: {similarity}")


def test_threshold():
    resume_text1 = extract_text_from_docx("testing/11.docx")
    resume_text2 = extract_text_from_pdf("testing/8.pdf")
    resume_text3 = extract_text_from_docx("testing/5.docx")
    generator = EmbeddingGenerator()
    text = """
           1. Develop and execute the company’s business strategies in order to attain the goals of 
           the board and shareholders 
           2. Provide strategic advice to the board and Chairperson so that they will have accurate 
           view of the market and the company’s future 
           3. Prepare and implement comprehensive business plans to facilitate achievement by 
           planning cost-effective operations and market development activities 
           4. Ensure company policies and legal guidelines are communicated all the way from the 
           top down in the company and that they are followed at all times 
           5. Communicate and maintain trust relationships with shareholders, business partners and 
           authorities 
           6. Oversee the company’s financial performance, investments and other business ventures 
           7. Delegate responsibilities and supervise the work of executives providing guidance and 
           motivation to drive maximum performance 
           8. Read all submitted reports by lower rank managers to reward performance, prevent 
           issues and resolve problems """
    job_embedding = generator.generate(text)
    resume1 = Resume(resume_text1)
    resume1.embedding = generator.generate(resume1.text)
    resume1.similarity = calculate_similarity(job_embedding, resume1.embedding)

    resume2 = Resume(resume_text2)
    resume2.embedding = generator.generate(resume2.text)
    resume2.similarity = calculate_similarity(job_embedding, resume2.embedding)

    resume3 = Resume(resume_text3)
    resume3.embedding = generator.generate(resume3.text)
    resume3.similarity = calculate_similarity(job_embedding, resume3.embedding)

    resumes = [resume1, resume2, resume3]
    resumes = filter_by_threshold(resumes, 0.544)
    for resume in resumes:
        # should print two similarity scores as the third is below the threshold.
        print(f" {resume.similarity}")
        assert resume.similarity >= 0.544, "The similarity is below the threshold value!"

    print("Success!")


def test_request():
    response = make_request("What can you tell me about smurfs?")
    assert response, "Your request failed!"
    print(f"Answer is: \n{response}")


def test_extraction():
    generator = EmbeddingGenerator()
    job_text = """
               1. Develop and execute the company’s business strategies in order to attain the goals of 
               the board and shareholders 
               2. Provide strategic advice to the board and Chairperson so that they will have accurate 
               view of the market and the company’s future 
               3. Prepare and implement comprehensive business plans to facilitate achievement by 
               planning cost-effective operations and market development activities 
               4. Ensure company policies and legal guidelines are communicated all the way from the 
               top down in the company and that they are followed at all times 
               5. Communicate and maintain trust relationships with shareholders, business partners and 
               authorities 
               6. Oversee the company’s financial performance, investments and other business ventures 
               7. Delegate responsibilities and supervise the work of executives providing guidance and 
               motivation to drive maximum performance 
               8. Read all submitted reports by lower rank managers to reward performance, prevent 
               issues and resolve problems """
    job_embedding = generator.generate(job_text)
    resume1 = Resume(extract_text_from_docx("testing/10.docx"))
    resume1.embedding = generator.generate(resume1.text)
    resume1.similarity = calculate_similarity(job_embedding, resume1.embedding)

    resume2 = Resume(extract_text_from_pdf("testing/8.pdf"))
    resume2.embedding = generator.generate(resume2.text)
    resume2.similarity = calculate_similarity(job_embedding, resume2.embedding)

    resume3 = Resume(extract_text_from_docx("testing/5.docx"))
    resume3.embedding = generator.generate(resume3.text)
    resume3.similarity = calculate_similarity(job_embedding, resume3.embedding)

    resume4 = Resume(extract_text_from_docx("testing/11.docx"))
    resume4.embedding = generator.generate(resume4.text)
    resume4.similarity = calculate_similarity(job_embedding, resume4.embedding)

    resume5 = Resume(extract_text_from_docx("testing/12.docx"))
    resume5.embedding = generator.generate(resume5.text)
    resume5.similarity = calculate_similarity(job_embedding, resume5.embedding)

    resume6 = Resume(extract_text_from_docx("testing/13.docx"))
    resume6.embedding = generator.generate(resume6.text)
    resume6.similarity = calculate_similarity(job_embedding, resume6.embedding)

    resume7 = Resume(extract_text_from_pdf("testing/1.pdf"))
    resume7.embedding = generator.generate(resume7.text)
    resume7.similarity = calculate_similarity(job_embedding, resume7.embedding)

    resume8 = Resume(extract_text_from_pdf("testing/2.pdf"))
    resume8.embedding = generator.generate(resume8.text)
    resume8.similarity = calculate_similarity(job_embedding, resume8.embedding)

    resume9 = Resume(extract_text_from_pdf("testing/3.pdf"))
    resume9.embedding = generator.generate(resume9.text)
    resume9.similarity = calculate_similarity(job_embedding, resume9.embedding)

    resume10 = Resume(extract_text_from_pdf("testing/4.pdf"))
    resume10.embedding = generator.generate(resume10.text)
    resume10.similarity = calculate_similarity(job_embedding, resume10.embedding)

    resumes = [resume1, resume2, resume3, resume4, resume5, resume6, resume7, resume8, resume9, resume10]
    resumes = filter_by_threshold(resumes, 0.544)
    features = []
    for resume in resumes:
        features.append(make_request(extract_info(resumes, resume.text, job_text, RESUME_PROMPT)))

    resumes_f = []
    i = 1
    for feature, res in zip(features, resumes):
        print(f"Resume {i}:")
        print(feature)
        print(f"Cosine Similarity: {res.similarity}")
        resume = f"Resume {i}: {feature}\nCosine Similarity: {res.similarity}"
        resumes_f.append(resume)
        i = i + 1
    assert resumes_f, "Extraction failed!"
    print("Resume features have been extracted successfully.")
    RESUMES.extend(resumes_f)


def test_shortlisting():
    short = make_request(shortlist(RESUME_PROMPT2, RESUMES))
    assert "Top 5" in short, f"The resume shortlisting has failed!\nWe got: {short}"
    print("Resumes with extracted features have been shortlisted successfully.")
    SHORT.join(short)


def test_analysis():
    job_text = """
                   1. Develop and execute the company’s business strategies in order to attain the goals of 
                   the board and shareholders 
                   2. Provide strategic advice to the board and Chairperson so that they will have accurate 
                   view of the market and the company’s future 
                   3. Prepare and implement comprehensive business plans to facilitate achievement by 
                   planning cost-effective operations and market development activities 
                   4. Ensure company policies and legal guidelines are communicated all the way from the 
                   top down in the company and that they are followed at all times 
                   5. Communicate and maintain trust relationships with shareholders, business partners and 
                   authorities 
                   6. Oversee the company’s financial performance, investments and other business ventures 
                   7. Delegate responsibilities and supervise the work of executives providing guidance and 
                   motivation to drive maximum performance 
                   8. Read all submitted reports by lower rank managers to reward performance, prevent 
                   issues and resolve problems """
    response = make_request(final_analysis(RESUME_PROMPT3, job_text, SHORT))
    assert "Best Resume" in response, f"Final analysis phase failed!\nWe got: {response}"
    print("Our application's functionality is integrated successfully :)")
