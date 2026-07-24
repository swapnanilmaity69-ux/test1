import os
import io
import docx
import streamlit as st
from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import List

# Page Setup
st.set_page_config(page_title="AI Resume Evaluator", page_icon="📄", layout="centered")

st.title("📄 AI Resume & CV Evaluator")
st.write("Upload your resume in **PDF, DOCX, PNG, or JPG** format to get an automated ATS evaluation.")

# API Key Sidebar Input
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Enter your Gemini API Key:", type="password")
    st.markdown("[Get a free Gemini API Key](https://aistudio.google.com/)")

# Pydantic Schema for Evaluation Output
class ResumeAssessment(BaseModel):
    candidate_name: str
    target_role: str
    skills: List[str]
    years_of_experience: float
    ats_score: int
    strengths: List[str]
    improvements: List[str]
    suggested_keywords: List[str]

# Processing Function
def analyze_resume_bytes(file_bytes: bytes, filename: str, client: genai.Client) -> ResumeAssessment:
    ext = filename.split(".")[-1].lower()
    prompt = "Analyze this resume. Evaluate its ATS optimization, extract candidate details, and output actionable feedback."

    if ext == "pdf":
        content_input = [
            prompt,
            types.Part.from_bytes(data=file_bytes, mime_type="application/pdf")
        ]
    elif ext in ["png", "jpg", "jpeg"]:
        mime = "image/png" if ext == "png" else "image/jpeg"
        content_input = [
            prompt,
            types.Part.from_bytes(data=file_bytes, mime_type=mime)
        ]
    elif ext == "docx":
        # Read byte buffer for DOCX
        doc = docx.Document(io.BytesIO(file_bytes))
        extracted_text = "\n".join([p.text for p.text in doc.paragraphs if p.text.strip()])
        content_input = f"{prompt}\n\nResume Content:\n{extracted_text}"
    else:
        raise ValueError("Unsupported format")

    response = client.models.generate_content(
        model="gemini-2,0-flash",
        contents=content_input,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ResumeAssessment,
            temperature=0.2,
        ),
    )
    return response.parsed

# File Upload Widget
uploaded_file = st.file_uploader("Upload your CV / Resume", type=["pdf", "docx", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    if not api_key:
        st.error("Please enter your Gemini API Key in the sidebar to proceed.")
    else:
        if st.button("Evaluate Resume", type="primary"):
            try:
                # Initialize Client with User Key
                client = genai.Client(api_key=api_key)
                
                with st.spinner("Analyzing resume structure and content..."):
                    file_bytes = uploaded_file.getvalue()
                    result = analyze_resume_bytes(file_bytes, uploaded_file.name, client)

                st.success("Analysis Complete!")
                
                # Display Results in Visual Dashboard
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Candidate Name", value=result.candidate_name)
                    st.metric(label="Target Role", value=result.target_role)
                with col2:
                    st.metric(label="ATS Score", value=f"{result.ats_score} / 100")
                    st.metric(label="Experience", value=f"{result.years_of_experience} Years")

                st.divider()

                # Skills Section
                st.subheader("🛠 Extracted Skills")
                st.write(", ".join([f"`{skill}`" for skill in result.skills]))

                # Feedback Sections
                col_a, col_b = st.columns(2)
                with col_a:
                    st.subheader("✅ Strengths")
                    for s in result.strengths:
                        st.markdown(f"- {s}")
                with col_b:
                    st.subheader("💡 Key Improvements")
                    for imp in result.improvements:
                        st.markdown(f"- {imp}")

                st.subheader("🔍 Recommended ATS Keywords")
                st.write(", ".join([f"**{kw}**" for kw in result.suggested_keywords]))

            except Exception as e:
                st.error(f"Error processing document: {e}")
