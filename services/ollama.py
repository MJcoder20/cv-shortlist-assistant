import requests

# OpenRouter API configuration
API_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY = "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"
headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

# "model": "meta-llama/llama-3.1-8b-instruct:free",  # no
# "model": "meta-llama/llama-3.2-1b-instruct:free",   # yes
# "model": "meta-llama/llama-3.2-3b-instruct",  # no
# "model": "microsoft/phi-3-mini-128k-instruct:free",  # yes
# "model": "google/gemma-2-9b-it:free",  # no
# "model": "mistralai/mistral-7b-instruct:free",  # yes

model = "mistralai/mistral-7b-instruct:free"

RESUME_PROMPT = """
    Extract the skills, experience, and qualifications from the following Resume text with the output is precise and abbreviated.
                Format the output as:
                - Technical Skills: [list of technical skills] 
                - Soft Skills: [list of soft skills]
                - Experience: [list of experience requirements]
                - Qualifications: [list of qualifications]
                - Score: [score value out of 10 for the resume compared to all resumes based on its level of match to
                 the job description and criteria listed below]
                - Justification of given score: [bullet point list of reasons why the resume got the previous score value]
                and Evaluate this resume against the job description: {job_text} and the following criteria:
                - Required Skills: {required_skills}
                - Minimum Experience: {min_experience} years
                - Required Education: {education_level}

                
            Resume:
            {{text}}
            All Resumes:
            {{resume_texts}}
            """

RESUME_PROMPT2 = """                   
            Execute the following based on the text provided below: 
            - Shortlist the resumes into the top 5 resumes based on the score.
            
            Text:
            {prev_data}
"""

RESUME_PROMPT3 = """
            Execute the following based on the text of the resume shortlist provided below: 
            - Analyze each resumes general strengths and weaknesses.
            - Analyze each resumes strengths and weaknesses considering the job description{job_text}.
            - Conclude the best resume based on the previous data 
            and output it in the format: 
                - Best Resume: [Name of resume]
                - [why this was concluded as best resume]
            
            Format the output as shown for each resume:
            - Strengths: [list of strengths]
            - Weaknesses: [list of weaknesses]
                
            Text:
            {prev_data}
"""


def make_request(prompt):
    try:

        payload = {
            "model": model,  # model name as shown in openrouter website
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


def extract_info(resume_texts, text, job_text, prompt_template, min_experience, required_skills, education_level):
    # Inject user criteria into the prompt
    prompt = prompt_template.format(
        min_experience=min_experience,
        required_skills=required_skills,
        education_level=education_level,
        job_text=job_text,
        resume_texts=resume_texts).replace("{text}", text)
    return prompt


def shortlist(prompt_template, prev_data):
    # Inject user criteria into the prompt
    prompt = prompt_template.format(prev_data=prev_data)
    return prompt


def final_analysis(prompt_template, job_text, prev_data):
    # Inject user criteria into the prompt
    prompt = prompt_template.format(job_text=job_text, prev_data=prev_data)
    return prompt
