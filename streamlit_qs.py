import sqlite3
import streamlit as st
import os
import json
import random
import io
import logging
import re
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
import pandas as pd

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2500,
    chunk_overlap=500
)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load environment variables and configure the model
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Set up the model configuration
generation_config = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "max_output_tokens": 8000,
}
system_instruction = "You are a helpful document answering assistant. You care about user and user experience. You always make sure to fulfill user requests."
model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest", generation_config=generation_config, system_instruction=system_instruction)

# Initialize the databases
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
            correct_answer TEXT,
            rating TEXT DEFAULT 'Unrated'
        )
    ''')
    conn.commit()
    return conn, cursor

conn, cursor = initialize_database()

# Function to insert data into the database
def save_new_question(lesson_name, questions, question_type, context, correct_answer):
    for question in questions:
        options = json.dumps(question.get('options', []), ensure_ascii=False)
        question_text = question.get('question', '')
        correct_answer = question.get('correct_answer', None)
        try:
            cursor.execute(
                "INSERT INTO questions (lesson_name, question, question_type, options, correct_answer) VALUES (?, ?, ?, ?, ?)",
                (lesson_name, question_text, question_type, options, correct_answer)
            )
        except sqlite3.IntegrityError:
            continue
    conn.commit()

# Function to query data from the database
def get_questions():
    cursor.execute("SELECT * FROM questions")
    rows = cursor.fetchall()
    return rows

# Function to rate a question as "Good" or "Bad"
def rate_question(question_id, rating):
    cursor.execute(
        "UPDATE questions SET rating = ? WHERE id = ?",
        (rating, question_id)
    )
    conn.commit()

# Function to download the database
def download_database():
    with open('questions.db', 'rb') as f:
        st.download_button(
            label="Download Database",
            data=f,
            file_name="questions.db",
            mime="application/octet-stream"
        )

# Function to extract and chunk PDF text
def get_all_pdfs_chunks(pdf_docs):
    all_chunks = []
    for pdf in pdf_docs:
        pdf_chunks = get_single_pdf_chunks(pdf, text_splitter)
        all_chunks.extend(pdf_chunks)
    
    random.shuffle(all_chunks)
    
    return all_chunks

def get_single_pdf_chunks(pdf, text_splitter):
    pdf_reader = PdfReader(pdf)
    pdf_chunks = []
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        page_chunks = text_splitter.split_text(page_text)
        pdf_chunks.extend(page_chunks)
    return pdf_chunks

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
                return None
        except (ValueError, json.JSONDecodeError) as e:
            logging.error(f"Response is not a valid JSON: {str(e)}")
            return None

# Function to generate a common prompt template
def get_prompt_template(context, num_questions, question_type):
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
            You are an AI assistant tasked with generating {num_questions} {prompt_type} related to presented study material grammar and comprehension from the given context. Do not get out of the context.
            Ensure the following guidelines while generating the questions:
            1. Vary the types of questions between open and closed, and between direct and reflective questions to ensure a comprehensive assessment.
            2. Focus on deep understanding by asking questions that measure the students' grasp of key concepts, requiring explanations or examples where appropriate.
            3. Relate questions to real-life scenarios to help students see the broader relevance of the material.
            4. Encourage critical thinking by including questions that ask 'why' or explore potential consequences.
            5. Include questions of varying difficulty levels to accommodate students with different abilities, ensuring some questions are answerable by all.
            6. Ensure clarity in the wording of questions to avoid ambiguity and confusion.
            7. The language of the question must be the same as the language of the content presented in the lessons, whether in MCQs, true/false, or written questions.
            Ensure the output is in JSON format with fields 'question', 'options', and 'correct_answer'.
            {options_format}
            Context: {context}\n
            """
    
    return prompt_template

# Function to generate questions using the model
def generate_questions(context, num_questions, question_type):
    prompt = get_prompt_template(context, num_questions, question_type)
    try:
        response = model.start_chat(history=[]).send_message(prompt)
        response_text = response.text.strip()
        logging.debug(f"Raw response from model: {response_text}")

        if response_text:
            return clean_json_response(response_text)
        else:
            logging.error("Empty response from the model")
            return None
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return None

# Path to the folder containing PDF files
DATA_FOLDER_PATH = "./Data"

# Step 1: File Selection
st.title("Step 1: Select PDF Files to Process")
files = [f for f in os.listdir(DATA_FOLDER_PATH) if f.endswith('.pdf')]
selected_files = st.multiselect("Select files to process", files)

if selected_files:
    # Step 2: Set Options for Selected Files
    st.title("Step 2: Set Options for Selected Files")
    lesson_percentage_distribution = {}

    for file in selected_files:
        st.subheader(f"Set percentages for {file}")
        lesson_name = os.path.splitext(file)[0]
        lesson_percentage_distribution[lesson_name] = {}

        for question_type in ["MCQ", "TF", "WRITTEN"]:
            lesson_percentage_distribution[lesson_name][question_type] = st.slider(
                f"Percentage for {question_type} in {file}",
                0, 100, 33
            )

    num_questions = st.number_input("Enter the total number of questions per lesson", min_value=1, max_value=100, value=10)

    # Step 3: Generate Questions
    if st.button("Generate Questions"):
        if not any(lesson_percentage_distribution.values()):
            st.error("Please set at least one percentage distribution for each lesson.")
        else:
            results = {}
            for pdf_filename in selected_files:
                lesson_name = os.path.splitext(pdf_filename)[0]
                pdf_path = os.path.join(DATA_FOLDER_PATH, pdf_filename)

                try:
                    with open(pdf_path, "rb") as pdf_file:
                        pdf_content = io.BytesIO(pdf_file.read())
                        text_chunks = get_all_pdfs_chunks([pdf_content])
                        context = " ".join(random.sample(text_chunks, min(num_questions, len(text_chunks))))

                        for question_type, percentage in lesson_percentage_distribution[lesson_name].items():
                            num_type_questions = int((percentage / 100) * num_questions)
                            if num_type_questions > 0:
                                generated_questions = generate_questions(context, num_type_questions, question_type)

                                if generated_questions:
                                    save_new_question(lesson_name, generated_questions, question_type, context, "")
                                    results.setdefault(pdf_filename, []).extend(generated_questions)
                                else:
                                    st.error(f"Failed to generate {question_type} questions for {pdf_filename}.")
                except FileNotFoundError:
                    st.error(f"File {pdf_filename} not found in the folder '{DATA_FOLDER_PATH}'.")

            st.success("Questions generated successfully.")
            st.write(results)

# Display the questions in the database
st.subheader("Current Questions in the Database")
questions_df = pd.DataFrame(get_questions(), columns=['ID', 'Lesson Name', 'Question', 'Question Type', 'Options', 'Correct Answer', 'Rating'])
st.dataframe(questions_df)

# Rate the questions
if not questions_df.empty:
    st.subheader("Rate Questions")
    question_id = st.selectbox("Select Question ID to Rate", questions_df['ID'])
    rating = st.radio("Rating", ["Good", "Bad"])

    if st.button("Submit Rating"):
        rate_question(question_id, rating)
        st.success("Rating submitted successfully!")
        questions_df = pd.DataFrame(get_questions(), columns=['ID', 'Lesson Name', 'Question', 'Question Type', 'Options', 'Correct Answer', 'Rating'])  # Refresh the data
        st.dataframe(questions_df)

# Download the database
download_database()

# Close the database connection at the end
conn.close()
