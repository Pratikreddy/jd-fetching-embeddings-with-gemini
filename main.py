#this is working but the vectors have stopped getting passed

import json
import faiss
import pandas as pd
import numpy as np
import ollama
import google.generativeai as genai
import time

API_KEY = ""
genai.configure(api_key=API_KEY)

# Load FAISS index and DataFrame
try:
    faiss_index = faiss.read_index('resume_index_50.faiss')
    resume_df = pd.read_pickle('resumes_50.pkl')
    print("Successfully loaded FAISS index and resume DataFrame")
except FileNotFoundError:
    print("Error: FAISS index or resume DataFrame not found. Please ensure the files exist.")
    exit(1)

def get_embeddings(text, model='snowflake-arctic-embed:335m'):
    try:
        print(f"Getting embeddings for text: {text[:50]}...")
        response = ollama.embeddings(model=model, prompt=text)
        print("Successfully obtained embeddings")
        return response['embedding']
    except Exception as e:
        print(f"Error getting embeddings: {str(e)}")
        return None

def find_similar_resumes(job_description, top_n=20):
    print("Finding similar resumes...")
    job_embedding = np.array(get_embeddings(json.dumps(job_description))).astype('float32').reshape(1, -1)
    if job_embedding is not None:
        distances, indices = faiss_index.search(job_embedding, top_n)
        print(f"Found {top_n} similar resumes")
        return resume_df.iloc[indices[0]]
    print("Failed to find similar resumes due to embedding error")
    return pd.DataFrame()

def gemini_json(system_prompt, user_prompt, model_name='gemini-1.5-pro-latest'):
    model = genai.GenerativeModel(model_name)
    combined_prompt = f"""
    System: {system_prompt}
    User: {user_prompt}
    Instructions: Always respond in the specified JSON format.
    """
    
    print("Full data being sent to LLM:")
    print(combined_prompt)
    
    try:
        print("Sending request to Gemini API...")
        start_time = time.time()
        response = model.generate_content(
            combined_prompt,
            generation_config={
                "temperature": 0.7,
                "top_p": 1,
                "top_k": 1,
                "max_output_tokens": 2048,
                "response_mime_type": "application/json"
            },
            stream=False,
        )
        end_time = time.time()
        print(f"Received response from Gemini API. Time taken: {end_time - start_time:.2f} seconds")
        parsed_response = json.loads(response.text)
        print(f"Full LLM Response: {json.dumps(parsed_response, indent=2)}")
        return parsed_response
    except Exception as e:
        print(f"An error occurred. Full error message:")
        print(str(e))
        return None

def process_job_description(job_info):
    print("Processing job description...")
    system_prompt = """
    You are an AI assistant processing job descriptions. Analyze the given job information and structure it into the following JSON format:
    {
        "phase": "job_description",
        "status": "complete" or "incomplete",
        "job_description": {
            "title": "Job Title",
            "responsibilities": ["Responsibility 1", "Responsibility 2", ...],
            "skills": [{"skillName": "Skill 1"}, {"skillName": "Skill 2"}, ...],
            "experience": "Required experience",
            "education": "Required education"
        },
        "missing_info": "Description of what information is still needed" (only if status is "incomplete")
    }
    Ensure all fields are filled based on the provided information. If any crucial information is missing, set the status to "incomplete" and describe what's missing.
    """

    return gemini_json(system_prompt, job_info)

def rank_candidates(job_description, top_resumes):
    print("Ranking candidates...")
    system_prompt = """
    You are an AI assistant ranking job candidates based on their resumes and a given job description.
    Analyze the job description and candidates' information, then rank the candidates.
    Respond in the following JSON format:
    {
        "ranked_candidates": [
            {
                "id": "Candidate ID",
                "rank": 1,
                "reason": "Brief explanation for this ranking"
            },
            ...
        ]
    }
    """
    
    job_desc_str = json.dumps(job_description)
    
    # Prepare resume data without vector information
    relevant_fields = ['ID', 'Category', 'aiextract']
    prepared_resumes = top_resumes[relevant_fields].to_dict(orient='records')
    resumes_str = json.dumps(prepared_resumes)
    
    user_prompt = f"Job Description: {job_desc_str}\n\nCandidate Resumes: {resumes_str}"
    
    response = gemini_json(system_prompt, user_prompt)
    if response:
        print("Successfully ranked candidates")
    else:
        print("Failed to rank candidates")
    return response.get('ranked_candidates', []) if response else None

def execute_pipeline(job_description):
    print("Executing pipeline...")
    top_resumes = find_similar_resumes(job_description)
    print("Top matching resumes found")
    ranked_candidates = rank_candidates(job_description, top_resumes)

    if ranked_candidates:
        print("Top Ranked Candidates:")
        print(json.dumps(ranked_candidates, indent=2))
    else:
        print("Failed to rank candidates")

def main():
    print("Welcome to the Job Description Processor and Resume Matcher!")
    
    job_info = input("Enter the full job description: ")
    print("Processing job description...")
    result = process_job_description(job_info)
    
    if result:
        if result.get('status') == 'complete':
            print("Job description is complete. Proceeding to resume matching.")
            print("Processed Job Description:")
            print(json.dumps(result['job_description'], indent=2))
            execute_pipeline(result['job_description'])
        else:
            print(f"Job description is incomplete. {result.get('missing_info', 'Unknown error')}")
            print("Current Job Description:")
            print(json.dumps(result.get('job_description', {}), indent=2))
    else:
        print("Failed to process job description. Please try again.")

if __name__ == "__main__":
    main()
