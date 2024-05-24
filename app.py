from dotenv import load_dotenv

load_dotenv()
import base64
import streamlit as st
import os
import io
import pdf2image
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(input,pdf_content,prompt):
    model=genai.GenerativeModel('gemini-pro-vision')
    response=model.generate_content([input,pdf_content[0],prompt])
    return response.text

def input_pdf_setup(uploaded_file):
    if uploaded_file is not None:
        ## Convert the PDF to image
        images=pdf2image.convert_from_bytes(uploaded_file.read())

        first_page=images[0]

        # Convert to bytes
        img_byte_arr = io.BytesIO()
        first_page.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        pdf_parts = [
            {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(img_byte_arr).decode()  # encode to base64
            }
        ]
        return pdf_parts
    else:
        raise FileNotFoundError("No file uploaded")

## Streamlit App

st.set_page_config(page_title="ATS Resume Expert")
st.header("ATS Tracking System")
input_text=st.text_area("Job Description: ",key="input")
uploaded_file=st.file_uploader("Upload Your Resume (PDF): ", type=["pdf"])


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

input_prompt2 = """
You are a skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality. Your task is to evaluate the provided resume against the given job description. Follow these steps:

Calculate the percentage match based on keyword comparison between the job description and the resume.
List the keywords from the job description that are missing in the resume.
Provide final thoughts on the candidate's suitability for the role.
"""

if submit1:
    if uploaded_file is not None:
        pdf_content=input_pdf_setup(uploaded_file)
        response=get_gemini_response(input_prompt1,pdf_content,input_text)
        st.subheader("Response:")
        st.write(response)
    else:
        st.write("Please Upload Your Resume")

elif submit2:
    if uploaded_file is not None:
        pdf_content=input_pdf_setup(uploaded_file)
        response=get_gemini_response(input_prompt2,pdf_content,input_text)
        st.subheader("Response:")
        st.write(response)
    else:
        st.write("Please Upload Your Resume")
