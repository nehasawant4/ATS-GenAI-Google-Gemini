from dotenv import load_dotenv

load_dotenv()
import base64
import streamlit as st
import io
import pdf2image
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

def input_pdf_setup(uploaded_file):
    if uploaded_file is not None:
        images = pdf2image.convert_from_bytes(uploaded_file.read())

        first_page = images[0]

        img_byte_arr = io.BytesIO()
        first_page.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        pdf_parts = [
            {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(img_byte_arr).decode()
            }
        ]
        return pdf_parts
    else:
        raise FileNotFoundError("No file uploaded")

def evaluate_with_fine_tuned_model(job_description, resume):
    model = TFAutoModelForSequenceClassification.from_pretrained('./fine_tuned_model')
    tokenizer = AutoTokenizer.from_pretrained('./fine_tuned_model')

    inputs = tokenizer(job_description, resume, return_tensors="tf", truncation=True, padding="max_length", max_length=512)
    outputs = model(inputs)
    scores = outputs.logits.numpy()
    return scores


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

input_prompt2 = """
You are a skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality. Your task is to evaluate the provided resume against the given job description. Follow these steps:

Calculate the percentage match based on keyword comparison between the job description and the resume.
List the keywords from the job description that are missing in the resume.
Provide final thoughts on the candidate's suitability for the role.
"""

if submit1:
    if uploaded_file is not None:
        pdf_content = input_pdf_setup(uploaded_file)
        resume_text = pdf_content[0]['data'] 
        response = evaluate_with_fine_tuned_model(input_text, resume_text)
        st.subheader("Response:")
        st.write(response)
    else:
        st.write("Please Upload Your Resume")

elif submit2:
    if uploaded_file is not None:
        pdf_content = input_pdf_setup(uploaded_file)
        resume_text = pdf_content[0]['data']
        response = evaluate_with_fine_tuned_model(input_text, resume_text)
        st.subheader("Response:")
        st.write(response)
    else:
        st.write("Please Upload Your Resume")
