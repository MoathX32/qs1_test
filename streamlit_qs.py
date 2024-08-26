import os
import streamlit as st
import google.generativeai as genai
import json
import io
import random
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from langdetect import detect
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Define the path to the input folder
DATA_FOLDER_PATH = "./Data"

# Initialize logging
logging.basicConfig(level=logging.DEBUG)

# Function to extract and chunk PDF text
def get_single_pdf_chunks(pdf, text_splitter):
    pdf_reader = PdfReader(pdf)
    pdf_chunks = []
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        page_chunks = text_splitter.split_text(page_text)
        pdf_chunks.extend(page_chunks)
    return pdf_chunks

# Function to get chunks from multiple PDFs
def get_all_pdfs_chunks(pdf_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=500
    )

    all_chunks = []
    for pdf in pdf_docs:
        pdf_chunks = get_single_pdf_chunks(pdf, text_splitter)
        all_chunks.extend(pdf_chunks)
    
    random.shuffle(all_chunks)
    
    return all_chunks

# Function to clean and parse JSON responses from the model
def clean_json_response(response_text):
    try:
        response_json = json.loads(response_text)
        return response_json
    except json.JSONDecodeError:
        try:
            cleaned_text = re.sub(r'```json', '', response_text).strip()
            cleaned_text = re.sub(r'```', '', cleaned_text).strip()
            match = re.search(r'(\{.*\}|\[.*\])', cleaned_text, re.DOTALL)
            if match:
                cleaned_text = match.group(0)
                response_json = json.loads(cleaned_text)
                return response_json
            else:
                logging.error("No JSON object or array found in response")
                raise ValueError("No JSON object or array found in response")
        except (ValueError, json.JSONDecodeError) as e:
            logging.error(f"Response is not a valid JSON: {str(e)}")
            raise ValueError(f"Response is not a valid JSON: {str(e)}")

# Function to generate a common prompt template
def get_prompt_template(context, num_questions, question_type, is_english):
    if question_type == "MCQ":
        prompt_type = "multiple-choice questions (MCQs)"
        options_format = "Create a set of MCQs with 4 answer options each and provide the correct answer."

    elif question_type == "TF":
        prompt_type = "true/false questions"
        options_format = "Create a set of True/False questions. No options are needed, but the correct answer is needed."
        
    elif question_type == "WRITTEN":
        prompt_type = "open-ended written questions"
        options_format = "Create open-ended written questions that require a descriptive answer. No options are needed, but the correct answer is needed."

    prompt_template = f"""
            You are an AI assistant tasked with generating {num_questions} {prompt_type} related to the presented study material's grammar and comprehension from the given context. Don't get out of the context.
            Ensure the following guidelines while generating the questions:
            1. Vary the types of questions between open and closed, and between direct and reflective questions to ensure a comprehensive assessment.
            2. Focus on deep understanding by asking questions that measure the students' grasp of key concepts, requiring explanations or examples where appropriate.
            3. Relate questions to real-life scenarios to help students see the broader relevance of the material.
            4. Encourage critical thinking by including questions that ask 'why' or explore potential consequences.
            5. Include questions of varying difficulty levels to accommodate students with different abilities, ensuring some questions are answerable by all.
            6. Ensure clarity in the wording of questions to avoid ambiguity and confusion.
            7. The language of the question must be the same as the language of the content presented in the lessons, whether in a MCQ, true/false, or written question.
            Ensure the output is in JSON format with fields 'question', 'options', and 'correct_answer'.
            {options_format}
            Context: {context}\n
            """
    
    return prompt_template

# Function to generate questions using the model
def generate_questions(context, num_questions, question_type, is_english):
    prompt = get_prompt_template(context, num_questions, question_type, is_english)
    try:
        # Assuming the generate_text method only requires prompt and temperature
        response = genai.generate_text(
            prompt=prompt,
            temperature=0.7
        )
        response_text = response['text'].strip()  # Assuming the response is in a dictionary with 'text'
        logging.debug(f"Raw response from model: {response_text}")

        if response_text:
            return clean_json_response(response_text)
        else:
            logging.error("Empty response from the model")
            raise ValueError("Empty response from the model")
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise ValueError(str(e))

# Streamlit UI Components
st.title("Automated Question Generation System")

# Select lesson files
files = [f for f in os.listdir(DATA_FOLDER_PATH) if f.endswith('.pdf')]
selected_files = st.multiselect("Select lesson files", files)

# Select question type
question_type = st.selectbox("Select question type", ["MCQ", "TF", "WRITTEN"])

# Specify the number of questions
num_questions = st.number_input("Number of questions", min_value=1, max_value=20, value=5)

# Generate questions button
if st.button("Generate Questions"):
    if selected_files:
        results = {}
        for pdf_filename in selected_files:
            lesson_name = os.path.splitext(pdf_filename)[0]
            pdf_path = os.path.join(DATA_FOLDER_PATH, pdf_filename)
            
            with open(pdf_path, "rb") as pdf_file:
                pdf_content = io.BytesIO(pdf_file.read())
                text_chunks = get_all_pdfs_chunks([pdf_content])
                context = " ".join(random.sample(text_chunks, min(num_questions, len(text_chunks))))
                
                is_english = detect(context[:500]) == 'en'
                
                generated_questions = generate_questions(context, num_questions, question_type, is_english)
                results[pdf_filename] = generated_questions

        st.session_state.generated_questions = results
        st.success("Questions generated successfully!")
    else:
        st.warning("Please select at least one lesson file.")

# Display generated questions if available
if "generated_questions" in st.session_state:
    st.subheader("Generated Questions")
    for lesson, questions in st.session_state.generated_questions.items():
        st.write(f"Lesson: {lesson}")
        for question in questions:
            st.write(question)
