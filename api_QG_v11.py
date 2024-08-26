import os
import google.generativeai as genai
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import JSONResponse
import io
import re
import random
import logging
import sqlite3
from langdetect import detect
from pydantic import BaseModel
from typing import List

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = FastAPI()

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


# Initialize main database for questions
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

# Initialize reviewed questions database
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

# Main and reviewed databases
conn, cursor = initialize_database()
reviewed_conn, reviewed_cursor = initialize_reviewed_database()

# Clear data in both databases (for testing purposes)
def clear_database(cursor , reviewed_cursor):
    cursor.execute('DELETE FROM questions')
    reviewed_cursor.execute("DELETE FROM reviewed_questions")
    conn.commit()
    reviewed_conn.commit()

# Clear data in both databases (for testing purposes)
clear_database(cursor , reviewed_cursor)

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
                raise HTTPException(status_code=500, detail="No JSON object or array found in response")
        except (ValueError, json.JSONDecodeError) as e:
            logging.error(f"Response is not a valid JSON: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Response is not a valid JSON: {str(e)}")

# Function to generate a common prompt template
def get_prompt_template(context, num_questions, question_type, is_english):
    if question_type == "MCQ":
        prompt_type = "multiple-choice questions (MCQs)"
        options_format = "Create a set of MCQs with 4 answer options each and provide the correct answer."

    elif question_type == "TF":
        prompt_type = "true/false questions"
        options_format = "Create a set of True/False questions. No options is needed. but the correct answer is needed"

        
    elif question_type == "WRITTEN":
        prompt_type = "open-ended written questions"
        options_format = "Create open-ended written questions that require a descriptive answer. No options is needed. but the correct answer is needed"

    prompt_template = f"""
            You are an AI assistant tasked with generating {num_questions} {prompt_type} related to presented study material grammar and comprehension from the given context don't get out of the context.  
            Ensure the following guidelines while generating the questions:
            1. Vary the types of questions between open and closed, and between direct and reflective questions to ensure a comprehensive assessment.
            2. Focus on deep understanding by asking questions that measure the students' grasp of key concepts, requiring explanations or examples where appropriate.
            3. Relate questions to real-life scenarios to help students see the broader relevance of the material.
            4. Encourage critical thinking by including questions that ask 'why' or explore potential consequences.
            5. Include questions of varying difficulty levels to accommodate students with different abilities, ensuring some questions are answerable by all.
            6. Ensure clarity in the wording of questions to avoid ambiguity and confusion.
            7.The language of the question must be the same as the language of the content presented in the lessons, whether in a MCQs or true-false or written question.
            Ensure the output is in JSON format with fields 'question', 'options', and 'correct_answer'.
            {options_format}
            Context: {context}\n
            """
    
    return prompt_template

# Function to generate questions using the model
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
            raise HTTPException(status_code=500, detail="Empty response from the model")
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Function to check for duplicate questions
def check_existing_questions(new_questions, question_type):
    existing_questions = []
    for question in new_questions:
        cursor.execute("SELECT question FROM questions WHERE question = ? AND question_type = ?", (question['question'], question_type))
        if cursor.fetchone():
            existing_questions.append(question['question'])
    return existing_questions

# Function to save new questions to the main database
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

# Function to send feedback to the model for learning
def send_feedback(prompt, score, reasons):
    
    try:
        response = model.feedback(prompt=prompt, score=score, reasons=reasons)
        return response
    except Exception as e:
        logging.error(f"Error in feedback submission: {str(e)}")
        raise

# Function to update the model with reviewed questions and reasons
def update_model_with_reviewed_questions():
    reviewed_cursor.execute("SELECT lesson_name, question, question_type, options, correct_answer, modification_reason FROM reviewed_questions WHERE approved = 1")
    reviewed_questions = reviewed_cursor.fetchall()

    for lesson_name, question, question_type, options, correct_answer, reason in reviewed_questions:
        context = f"Lesson: {lesson_name}\nQuestion: {question}\nOptions: {options}\nCorrect Answer: {correct_answer}\n"
        is_english = detect(context[:500]) == 'en'
        prompt = get_prompt_template(context, 1, question_type, is_english) + f"\nModification Reason: {reason}\n"

        # Send feedback to the model to reinforce learning from the correction
        try:
            feedback_response = send_feedback(prompt, score=1, reasons=["Correction"])
            logging.debug(f"Feedback response from model: {feedback_response}")
        except Exception as e:
            logging.error(f"Error in sending feedback: {str(e)}")

# Example endpoint to trigger the update model process
@app.post("/apply-feedback/")
async def apply_feedback():
    try:
        update_model_with_reviewed_questions()
        return JSONResponse(content={"status": "Model updated with reviewed questions successfully."})
    except Exception as e:
        logging.error(f"Error in applying feedback: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update the model with reviewed questions.")

# API endpoint to generate questions from PDFs
@app.post("/generate-questions/")
async def generate_questions_endpoint(pdf_directory: str = Form(...), question_counts: str = Form(...), question_types: str = Form(None)):
    try:
        question_counts = json.loads(question_counts)
        if question_types:
            question_types = json.loads(question_types)
        else:
            question_types = {}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format for question_counts or question_types")
    
    files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "max_output_tokens": 8000,
    }
    system_instruction = "You are a helpful document answering assistant."
    model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest", generation_config=generation_config, system_instruction=system_instruction)

    results = {}

    for pdf_filename in files:
        lesson_name = os.path.splitext(pdf_filename)[0]
        if lesson_name not in question_counts:
            continue

        count = question_counts[lesson_name]
        question_type = question_types.get(lesson_name, "MCQ")
        pdf_path = os.path.join(pdf_directory, pdf_filename)
        
        with open(pdf_path, "rb") as pdf_file:
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

    return JSONResponse(content=results)

# Pydantic models for supervisor decision
class Decision(BaseModel):
    question_id: int
    approval_status: bool

class Decisions(BaseModel):
    decisions: List[Decision]

# API endpoint to modify, approve, or delete questions
@app.post("/modify-question/")
async def modify_question_endpoint(
    question_id: int = Form(...),
    modified_question: str = Form(None),
    modified_options: str = Form(None),
    modified_answer: str = Form(None),
    modification_reason: str = Form(...),
    delete: bool = Form(False)
):
    try:
        # Fetch the original question from the main database
        cursor.execute("SELECT * FROM questions WHERE id = ?", (question_id,))
        original_question = cursor.fetchone()

        if not original_question:
            raise HTTPException(status_code=404, detail="Original question not found")

        # Check if the question already exists in the reviewed_questions database
        reviewed_cursor.execute(
            "SELECT id FROM reviewed_questions WHERE question = ? AND question_type = ?", 
            (original_question[2], original_question[3])
        )
        existing_reviewed_question = reviewed_cursor.fetchone()

        # Handle delete operation
        if delete:
            if existing_reviewed_question:
                # Update the existing reviewed question with the new deletion reason
                reviewed_cursor.execute(
                    "UPDATE reviewed_questions SET modification_reason = ?, approved = 0, is_deletion = 1 WHERE id = ?",
                    (modification_reason, existing_reviewed_question[0])
                )
            else:
                # Log the deletion request in the reviewed_questions table
                reviewed_cursor.execute(
                    "INSERT INTO reviewed_questions (lesson_name, question, question_type, options, correct_answer, modification_reason, approved, is_deletion) VALUES (?, ?, ?, ?, ?, ?, 0, 1)",
                    (
                        original_question[1],  # lesson_name
                        original_question[2],  # question text
                        original_question[3],  # question_type
                        original_question[4],  # options
                        original_question[5],  # correct_answer
                        modification_reason    # reason for deletion
                    )
                )
            reviewed_conn.commit()

            return {"status": "Deletion request logged and pending supervisor approval."}

        # Handle modify operation
        updated_question = modified_question if modified_question else original_question[2]
        updated_options = modified_options if modified_options else original_question[4]
        updated_answer = modified_answer if modified_answer else original_question[5]

        if existing_reviewed_question:
            # Update the existing reviewed question with the new modification details
            reviewed_cursor.execute(
                "UPDATE reviewed_questions SET lesson_name = ?, question = ?, question_type = ?, options = ?, correct_answer = ?, modification_reason = ?, approved = 0, is_deletion = 0 WHERE id = ?",
                (
                    original_question[1],  # lesson_name
                    updated_question,
                    original_question[3],  # question_type
                    updated_options,
                    updated_answer,
                    modification_reason,
                    existing_reviewed_question[0]  # ID of the existing reviewed question
                )
            )
        else:
            # Save the reviewed question to the reviewed_questions database
            reviewed_cursor.execute(
                "INSERT INTO reviewed_questions (lesson_name, question, question_type, options, correct_answer, modification_reason, approved, is_deletion) VALUES (?, ?, ?, ?, ?, ?, 0, 0)",
                (
                    original_question[1],  # lesson_name
                    updated_question,
                    original_question[3],  # question_type
                    updated_options,
                    updated_answer,
                    modification_reason
                )
            )
        reviewed_conn.commit()

        return {"status": "Modification request logged and pending supervisor approval."}

    except sqlite3.IntegrityError as e:
        logging.error(f"Integrity error occurred: {str(e)}")
        raise HTTPException(status_code=400, detail="Integrity error occurred. Possible duplicate.")
    except Exception as e:
        logging.error(f"Unexpected error in modification: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# API endpoint for supervisor to approve or reject modifications
@app.post("/supervise-questions/")
async def supervise_questions(decisions: Decisions):
    try:
        # Iterate over each decision in the list
        for decision in decisions.decisions:
            question_id = decision.question_id
            approval_status = decision.approval_status

            # Fetch the reviewed question based on the question ID
            reviewed_cursor.execute("SELECT * FROM reviewed_questions WHERE id = ?", (question_id,))
            reviewed_question = reviewed_cursor.fetchone()

            if not reviewed_question:
                raise HTTPException(status_code=404, detail=f"Reviewed question with ID {question_id} not found")

            # Update the approval status in the reviewed_questions table
            reviewed_cursor.execute(
                "UPDATE reviewed_questions SET approved = ? WHERE id = ?",
                (1 if approval_status else 0, question_id)
            )
            reviewed_conn.commit()

            # Check if the request is for deletion and if it was approved
            if reviewed_question[-1]:  # Assuming the `is_deletion` column is the last in reviewed_questions
                if approval_status:
                    cursor.execute("DELETE FROM questions WHERE id = ?", (question_id,))
                    conn.commit()
                    return {"status": f"Question ID {question_id} deleted successfully."}
            else:
                # If the modification is approved, update the corresponding question in the main questions table
                if approval_status:
                    cursor.execute(
                        "UPDATE questions SET question = ?, options = ?, correct_answer = ? WHERE id = ?",
                        (
                            reviewed_question[2],  # updated question text
                            reviewed_question[4],  # updated options
                            reviewed_question[5],  # updated correct answer
                            question_id             # original question ID
                        )
                    )
                    conn.commit()

        return {"status": "Batch processing of questions completed successfully."}

    except sqlite3.IntegrityError as e:
        logging.error(f"Integrity error occurred: {str(e)}")
        raise HTTPException(status_code=400, detail="Integrity error occurred. Possible duplicate.")
    except Exception as e:
        logging.error(f"Unexpected error in modification: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
