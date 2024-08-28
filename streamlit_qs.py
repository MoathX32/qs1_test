import os
import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
import io
import re
import random
import logging
import sqlite3
from langdetect import detect

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize logging
logging.basicConfig(level=logging.DEBUG)

# Initialize the model globally
generation_config = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "max_output_tokens": 8000,
}
system_instruction = "You are a helpful document answering assistant."
model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest", generation_config=generation_config, system_instruction=system_instruction)

# Initialize databases
def initialize_database():
    conn = sqlite3.connect('questions.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS questions (
            id INTEGER PRIMARY KEY,
            lesson_name TEXT,
            question TEXT UNIQUE,
            question_type TEXT,
            options TEXT,
            correct_answer TEXT
        )
    ''')
    conn.commit()
    return conn, cursor

def initialize_reviewed_database():
    conn = sqlite3.connect('reviewed_questions.db')
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reviewed_questions (
            id INTEGER PRIMARY KEY,
            lesson_name TEXT,
            question TEXT UNIQUE,
            question_type TEXT,
            options TEXT,
            correct_answer TEXT,
            modification_reason TEXT,
            approved INTEGER DEFAULT 0,
            is_deletion BOOLEAN DEFAULT 0
        )
    ''')
    conn.commit()
    return conn, cursor

# Initialize main and reviewed databases
conn, cursor = initialize_database()
reviewed_conn, reviewed_cursor = initialize_reviewed_database()

# Function to extract and chunk PDF text
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
    
    random.shuffle(all_chunks)
    
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
        except (ValueError, json.JSONDecodeError) as e:
            logging.error(f"Response is not a valid JSON: {str(e)}")
            st.error(f"Response is not a valid JSON: {str(e)}")

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
            You are an AI assistant tasked with generating {num_questions} {prompt_type} related to presented study material grammar and comprehension from the given context don't get out of the context.  
            Ensure the following guidelines while generating the questions:
            1. Vary the types of questions between open and closed, and between direct and reflective questions to ensure a comprehensive assessment.
            2. Focus on deep understanding by asking questions that measure the students' grasp of key concepts, requiring explanations or examples where appropriate.
            3. Relate questions to real-life scenarios to help students see the broader relevance of the material.
            4. Encourage critical thinking by including questions that ask 'why' or explore potential consequences.
            5. Include questions of varying difficulty levels to accommodate students with different abilities, ensuring some questions are answerable by all.
            6. Ensure clarity in the wording of questions to avoid ambiguity and confusion.
            7. The language of the question must be the same as the language of the content presented in the lessons, whether in MCQs or true-false or written questions.
            Ensure the output is in JSON format with fields 'question', 'options', and 'correct_answer'.
            {options_format}
            Context: {context}\n
            """
    
    return prompt_template

def generate_questions(context, num_questions, question_type, model, is_english):
    prompt = get_prompt_template(context, num_questions, question_type, is_english)
    try:
        response = model.start_chat(history=[]).send_message(prompt)
        response_text = response.text.strip()
        logging.debug(f"Raw response from model: {response_text}")

        if response_text:
            return clean_json_response(response_text)
        else:
            logging.error("Empty response from the model")
            st.error("Empty response from the model")
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        st.error(f"Error: {str(e)}")

def check_existing_questions(new_questions, question_type):
    existing_questions = []
    for question in new_questions:
        cursor.execute("SELECT question FROM questions WHERE question = ? AND question_type = ?", (question['question'], question_type))
        if cursor.fetchone():
            existing_questions.append(question['question'])
    return existing_questions

def save_new_questions(lesson_name, new_questions, question_type):
    for question in new_questions:
        options = json.dumps(question.get('options', []), ensure_ascii=False)
        correct_answer = question.get('correct_answer', None)
        try:
            cursor.execute(
                "INSERT INTO questions (lesson_name, question, question_type, options, correct_answer) VALUES (?, ?, ?, ?, ?)", 
                (lesson_name, question['question'], question_type, options, correct_answer)
            )
        except sqlite3.IntegrityError:
            continue
    conn.commit()

# Streamlit UI
st.title("PDF Question Generator")
st.write("Automatically generate questions from PDF files.")

# Input Folder and PDF Files
pdf_directory = "Data"
if not os.path.exists(pdf_directory):
    st.error(f"Input folder '{pdf_directory}' does not exist. Please ensure it is in the correct location.")
    st.stop()

pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
if not pdf_files:
    st.warning("No PDF files found in the 'Data' folder.")
    st.stop()

# Question Types and Percentages
st.sidebar.header("Question Types")
question_types = {"MCQ": st.sidebar.slider("MCQ Percentage", 0, 100, 50),
                  "TF": st.sidebar.slider("True/False Percentage", 0, 100, 30),
                  "WRITTEN": st.sidebar.slider("Written Question Percentage", 0, 100, 20)}

# Lessons and Percentages
st.sidebar.header("Select Lessons")
selected_lessons = {}
for pdf in pdf_files:
    lesson_name = os.path.splitext(pdf)[0]
    percentage = st.sidebar.slider(f"{lesson_name} Percentage", 0, 100, 100)
    selected_lessons[lesson_name] = percentage

# Number of Questions
num_questions = st.sidebar.number_input("Number of Questions per Lesson", min_value=1, value=5)

# Process Button
if st.button("Generate Questions"):
    st.write("Processing PDFs...")
    
    results = {}
    for pdf_filename, percentage in selected_lessons.items():
        lesson_name = os.path.splitext(pdf_filename)[0]
        count = num_questions * percentage // 100

        question_type = max(question_types, key=question_types.get)
        pdf_path = os.path.join(pdf_directory, pdf_filename)
        
        with open(pdf_path, "rb") as pdf_file):
            pdf_content = io.BytesIO(pdf_file.read())
            text_chunks = get_all_pdfs_chunks([pdf_content])
            context = " ".join(random.sample(text_chunks, min(count, len(text_chunks))))
            
            is_english = detect(context[:500]) == 'en'
            
            generated_questions = generate_questions(context, count, question_type, model, is_english)
            
            existing_questions = check_existing_questions(generated_questions, question_type)
            while existing_questions:
                generated_questions = generate_questions(context, count, question_type, model, is_english)
                existing_questions = check_existing_questions(generated_questions, question_type)

            save_new_questions(lesson_name, generated_questions, question_type)
            
            results[pdf_filename] = generated_questions

    st.write("Question generation completed. Here are the results:")

    for pdf_filename, questions in results.items():
        st.subheader(f"Lesson: {pdf_filename}")
        for i, question in enumerate(questions):
            st.write(f"**Question {i + 1}:** {question['question']}")
            if question['options']:
                for j, option in enumerate(question['options']):
                    st.write(f"- Option {j + 1}: {option}")
            st.write(f"**Correct Answer:** {question['correct_answer']}")
            st.write("---")
