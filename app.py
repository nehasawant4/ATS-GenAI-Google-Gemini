import streamlit as st
import re
import io
import os
import base64
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import pdf2image
import pytesseract
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Configure Google Gemini with a text model
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_gemini_response(prompt, job_description, resume_text):
    input_text = f"""
    {prompt}

    Job Description:
    {job_description}

    Resume:
    {resume_text}
    """
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content([input_text])
    return response.text


def input_pdf_setup(uploaded_file):
    if uploaded_file is not None:
        images = pdf2image.convert_from_bytes(uploaded_file.read())
        text = ""
        for image in images:
            text += extract_text_from_image(image) + " "
        return text.strip()
    else:
        raise FileNotFoundError("No file uploaded")


def extract_text_from_image(image):
    return pytesseract.image_to_string(image)


nltk.download('punkt')
nltk.download('stopwords')


def extract_keywords(text):
    words = word_tokenize(text.lower())
    custom_stopwords = set(stopwords.words('english')) - {"must", "required"}
    keywords = [word for word in words if word.isalpha() and word not in custom_stopwords]
    return set(keywords)


def calculate_match(job_description, resume):
    jd_keywords = extract_keywords(job_description)
    resume_keywords = extract_keywords(resume)
    common_keywords = jd_keywords.intersection(resume_keywords)
    missing_keywords = jd_keywords.difference(resume_keywords)
    match_percentage = (len(common_keywords) / len(jd_keywords)) * 100 if jd_keywords else 0
    return match_percentage, missing_keywords


# Streamlit App
st.set_page_config(page_title="ATS Resume Expert")
st.header("ATS Tracking System")
input_text = st.text_area("Job Description: ", key="input")
uploaded_file = st.file_uploader("Upload Your Resume (PDF): ", type=["pdf"])

if uploaded_file is not None:
    st.write("PDF Uploaded Successfully")

submit1 = st.button("Tell Me About My Resume")
submit2 = st.button("Percentage Match")

input_prompt1 = """
You are an experienced Technical Human Resource Manager. 
Your task is to review the provided resume against the specified job description and share your professional evaluation on whether the candidate's profile aligns with the role. 
Highlight the strengths and weaknesses of the applicant in relation to the job requirements.

For each resume, provide the following:
Technical Skills: Evaluate the relevance and proficiency of technical skills listed.
Experience: Assess the candidate’s previous experience and how it aligns with the job responsibilities.
Education: Review the educational qualifications and their relevance to the job profile.
Additional Skills: Note any additional skills or experiences that are beneficial for the role.
Overall Fit: Provide an overall assessment of how well the candidate fits the job profile, highlighting strengths and potential areas of concern.
"""

if submit1:
    if uploaded_file is not None:
        resume_text = input_pdf_setup(uploaded_file)
        response = get_gemini_response(input_prompt1, input_text, resume_text)
        st.subheader("Response:")
        st.write(response)
    else:
        st.write("Please Upload Your Resume")

if submit2:
    if uploaded_file is not None:
        resume_text = input_pdf_setup(uploaded_file)
        match_percentage, missing_keywords = calculate_match(input_text, resume_text)
        st.subheader("Response:")
        st.write(f"Percentage Match: {match_percentage:.2f}%")
        st.write(f"Missing Keywords: {', '.join(missing_keywords)}")
    else:
        st.write("Please Upload Your Resume")
