import os
import google.generativeai as genai
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
import streamlit as st
import io
import re
import logging

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Streamlit UI
st.title("AI-Generated Questions from PDFs")
st.markdown("Upload PDF lessons, select question types, and specify the number of questions to generate.")

# Load PDFs from the 'Data' folder in GitHub
data_folder = "Data"
pdf_files = [f for f in os.listdir(data_folder) if f.endswith('.pdf')]

# Function to read PDFs and split into chunks
def get_single_pdf_chunks(pdf, text_splitter):
    pdf_reader = PdfReader(pdf)
    pdf_chunks = []
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        page_chunks = text_splitter.split_text(page_text)
        pdf_chunks.extend(page_chunks)
    return pdf_chunks

def get_all_pdfs_chunks(pdf_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=500
    )

    all_chunks = []
    for pdf in pdf_docs:
        pdf_chunks = get_single_pdf_chunks(pdf, text_splitter)
        all_chunks.extend(pdf_chunks)
    return all_chunks

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
                st.error("No JSON object or array found in response")
                return None
        except (ValueError, json.JSONDecodeError) as e:
            logging.error(f"Response is not a valid JSON: {str(e)}")
            st.error(f"Response is not a valid JSON: {str(e)}")
            return None

def get_prompt_template(context, num_questions, question_type, is_english):
    if question_type == "MCQ":
        prompt_type = "multiple-choice questions (MCQs)"
        options_format = "Create a set of MCQs with 4 answer options each and provide the correct answer."

    elif question_type == "TF":
        prompt_type = "true/false questions"
        options_format = "Create a set of True/False questions. No options are needed, but the correct answer is required."

    elif question_type == "WRITTEN":
        prompt_type = "open-ended written questions"
        options_format = "Create open-ended written questions that require a descriptive answer. No options are needed, but the correct answer is required."

    prompt_template = f"""
        You are an AI assistant tasked with generating {num_questions} {prompt_type} related to presented study material grammar and comprehension from the given context. Do not stray from the context.
        Ensure the following guidelines while generating the questions:
        1. Vary the types of questions between open and closed, and between direct and reflective questions to ensure a comprehensive assessment.
        2. Focus on deep understanding by asking questions that measure the students' grasp of key concepts, requiring explanations or examples where appropriate.
        3. Relate questions to real-life scenarios to help students see the broader relevance of the material.
        4. Encourage critical thinking by including questions that ask 'why' or explore potential consequences.
        5. Include questions of varying difficulty levels to accommodate students with different abilities, ensuring some questions are answerable by all.
        6. Ensure clarity in the wording of questions to avoid ambiguity and confusion.
        7. The language of the question must be the same as the language of the content presented in the lessons, whether in MCQs, true/false, or written question format.
        Ensure the output is in JSON format with fields 'question', 'options', and 'correct_answer'.
        {options_format}
        Context: {context}\n
    """

    return prompt_template

def generate_questions(context, num_questions, question_type, is_english, model):
    prompt_template = get_prompt_template(context, num_questions, question_type, is_english)

    try:
        response = model.start_chat(history=[]).send_message(prompt_template)
        response_text = response.text.strip()
        logging.debug(f"Raw response from model: {response_text}")

        if response_text:
            return clean_json_response(response_text)
        else:
            logging.error("Empty response from the model")
            st.error("Empty response from the model")
            return None
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        st.error(f"Error: {str(e)}")
        return None

# Streamlit UI elements
selected_lessons = st.multiselect("Select lessons to process:", pdf_files)
selected_types = st.multiselect(
    "Select question types:",
    ["MCQ", "TF", "WRITTEN"],
    default=["MCQ", "TF", "WRITTEN"]
)
num_questions = st.number_input("Number of questions per lesson:", min_value=1, value=5)

# Allow users to set different percentages for each type
type_percentages = {}
for q_type in selected_types:
    percentage = st.slider(f"Percentage for {q_type}:", min_value=0, max_value=100, value=100//len(selected_types))
    type_percentages[q_type] = percentage

if st.button("Start Processing"):
    if not selected_lessons:
        st.error("Please select at least one lesson.")
    else:
        st.info("Processing started...")

        pdf_docs = [io.BytesIO(open(os.path.join(data_folder, lesson), "rb").read()) for lesson in selected_lessons]

        text_chunks = get_all_pdfs_chunks(pdf_docs)
        context = " ".join(text_chunks)

        generation_config = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "max_output_tokens": 8000,
        }
        system_instruction = "You are a helpful document answering assistant. You care about the user and user experience. You always make sure to fulfill user requests."
        model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest", generation_config=generation_config, system_instruction=system_instruction)

        for q_type, percentage in type_percentages.items():
            if percentage > 0:
                adjusted_num_questions = (num_questions * percentage) // 100
                st.info(f"Generating {adjusted_num_questions} questions of type {q_type}...")

                questions_json = generate_questions(context, adjusted_num_questions, q_type, True, model)

                if questions_json:
                    st.json(questions_json)
                else:
                    st.error(f"Failed to generate {q_type} questions.")

        st.success("Processing completed.")
