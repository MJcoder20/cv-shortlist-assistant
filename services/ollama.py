import time

from dotenv import load_dotenv
import os
import requests


RESUME_PROMPT = """
    Extract the skills, experience, and qualifications from the following Resume text with the output 
    is precise and abbreviated.
            Format the output as:
            - Email Address: [email address of the applicant]
            - Technical Skills: [list of technical skills] 
            - Soft Skills: [list of soft skills]
            - Experience: [list of experience requirements]
            - Qualifications: [list of qualifications]
            - Score: [integer value of 0 to 100 for the resume compared to resume objects listed below
             based on its level of match to the job description and criteria listed below]
            - Justification of given score: [bullet point list of reasons why the resume got 
            the previous score value]
            and Evaluate this resume against the job description: {job_text} and the following criteria:
            - Required Skills: {required_skills}
            - Minimum Experience: {min_experience} years
            - Required Education: {education_level}

                
            Resume:
            {{text}}
            All Resumes:
            {{resume_objects}}
            """

RESUME_PROMPT2 = """                   
    Execute the following based on the text provided below: 
    - Shortlist the resumes into the top 5 resumes based on the score.
    
    Text:
    {prev_data}
"""

RESUME_PROMPT3 = """
    Execute the following based on the text of the resume shortlist provided below: 
    - Analyze each resumes general strengths and weaknesses regardless of the job description.
    - Analyze each resumes strengths and weaknesses considering the job description{job_text}.
    
    Format the output as shown for each applicant's resume:
    - General Strengths: [list of strengths regardless of the job description]
    - General Weaknesses: [list of weaknesses regardless of the job description] 
    - Strengths: [list of strengths]
    - Weaknesses: [list of weaknesses]
    
    Then Conclude the best resume based on the previous data 
    and output it in the format: 
    - Best Resume: [Name of the applicant]
    - [why this was concluded as best resume]
    - Email Address: [email address of the applicant]
        
    Text:
    {prev_data}
"""

load_dotenv()
API_URL = os.getenv("API_URL")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")


def make_request(prompt):
    try:
        # OpenRouter API configuration
        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": OLLAMA_MODEL,  # model name as shown in openrouter website
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        }

        # Make the API request
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise an error for bad status codes

        # Debug: Print the response and status code
        print("API Response Status Code:", response.status_code)
        print("API Response Content:", response.text)

        # Extract the response content
        result = response.json()
        return (
            result["choices"][0]["message"]["content"] if "choices" in result else None
        )

    except Exception as e:
        print(f"Error: {e}")
        return f"{e}"


def extract_info(resume_texts, text, job_text, prompt_template, min_experience=None, required_skills=None, education_level=None):
    # Inject user criteria into the prompt
    prompt = prompt_template.format(
        min_experience=min_experience,
        required_skills=required_skills,
        education_level=education_level,
        job_text=job_text,
        resume_texts=resume_texts).replace("{text}", text)
    return prompt


def shortlist(prompt_template, prev_data):
    # Inject resumes and their extracted features into the prompt
    prompt = prompt_template.format(prev_data=prev_data)
    return prompt


def final_analysis(prompt_template, job_text, prev_data):
    # Inject shortlisted resumes and job description into the prompt
    prompt = prompt_template.format(job_text=job_text, prev_data=prev_data)
    return prompt
