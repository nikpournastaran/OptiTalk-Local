import json
import re
import ollama

# --- Configuration ---
# Professor's input text files (must be in the same directory)
INPUT_FILES = [
    {"filename": "input_testuali_tipologia1.txt", "task": "grammar_check"},
    {"filename": "input_testuali_tipologia2.txt", "task": "role_analysis"} # or subject_analysis depending on content
]
OUTPUT_FILE = "professor_dataset.jsonl"
MODEL_NAME = "qwen2:0.5b"  # The local model used to generate answers

def extract_questions(filename):
    """
    Reads the professor's text file and extracts questions based on the specific format
    (dashed lines and 'Domanda X').
    """
    questions = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Pattern to find questions between dashed lines
        # Assumes the question appears after "Domanda X" and the separator lines
        sections = re.split(r'-+Domanda \d+-+', content)
        
        for section in sections:
            if not section.strip(): continue
            
            # Clean text: remove empty lines and extra whitespace
            lines = [line.strip() for line in section.strip().split('\n') if line.strip()]
            
            if lines:
                # Logic to extract the question text
                # For Type 1 (Multiple Choice): Capture all text until the professor's answer key
                question_text = ""
                for line in lines:
                    # Stop reading when reaching the answer section (indicated by quotes or specific phrases)
                    if line.startswith('"') or line.startswith('La frase') or line.startswith('Il soggetto'): 
                        break 
                    question_text += line + " "
                
                if question_text:
                    questions.append(question_text.strip())
                    
    except FileNotFoundError:
        print(f"❌ File {filename} not found!")
    
    return questions

def get_model_response(question, task):
    """
    Sends the question to the local LLM and retrieves the generated answer.
    """
    system_prompt = """
Sei un esperto professore di linguistica italiana.
1. Se la domanda chiede di trovare il soggetto, rispondi solo con il soggetto.
2. Se la domanda chiede il ruolo di 'che', spiega la funzione grammaticale.
3. Se la domanda chiede quale frase è corretta, indica la frase corretta e spiega brevemente perché.
Rispondi in italiano.
"""
    try:
        response = ollama.chat(model=MODEL_NAME, messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': question}
        ])
        return response['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    print("🚀 Starting Dataset Generation...")
    final_data = []
    
    for file_info in INPUT_FILES:
        fname = file_info["filename"]
        task = file_info["task"]
        print(f"\n📂 Processing {fname}...")
        
        questions = extract_questions(fname)
        print(f"   Found {len(questions)} questions.")
        
        for i, q in enumerate(questions):
            print(f"   🤖 Asking Model Q{i+1}...")
            
            # Get the response from our local model (Minerva/OptiTalk)
            model_answer = get_model_response(q, task)
            
            entry = {
                "id": f"{fname.split('.')[0]}_Q{i+1}",
                "task": task,
                "question": q,
                "answer_minerva": model_answer # The actual model answer is recorded here
            }
            final_data.append(entry)

    # Save to JSONL file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in final_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print(f"\n✅ Done! Dataset saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()