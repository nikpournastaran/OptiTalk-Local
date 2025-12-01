import json
import csv
import ollama
import os

# --- Configuration ---
INPUT_JSONL = "professor_dataset.jsonl"       # <--- فایل جدید که الان ساختی
OUT_CSV = "evaluations_professor.csv"         # <--- اسم خروجی نهایی

# مدل اصلی (Main Model)
JUDGE_MODELS = [
    "llama3.2" 
]

EVAL_PROMPT_IT = """Sei un valutatore esperto di risposte in italiano.
Valuta la pertinenza complessiva della risposta di Minerva rispetto alla domanda dell'utente usando questa scala Likert (usa ESATTAMENTE un valore intero 1-5):

1 - Per niente pertinente
2 - Poco pertinente
3 - Abbastanza poco pertinente
4 - Abbastanza pertinente
5 - Completamente pertinente

IMPORTANTE: Rispondi ESCLUSIVAMENTE con JSON valido:
{{
  "score_pertinenza": <intero 1-5>,
  "commento": "<breve motivazione>"
}}

Domanda (utente):
{question}

Risposta (Minerva):
{answer}
"""

def read_jsonl(path):
    rows = []
    if not os.path.exists(path):
        print(f"❌ Error: File {path} not found!")
        return []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def clean_json_response(text):
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end != -1:
            return json.loads(text[start:end])
        return json.loads(text)
    except:
        return {"score_pertinenza": 1, "commento": "JSON Parse Error"}

def main():
    print("🚀 Starting AI Judge Evaluation on CPU...")
    
    data = read_jsonl(INPUT_JSONL)
    if not data: return

    results = []
    
    for row in data:
        qid = row.get("id")
        question = row.get("question")
        answer = row.get("answer_minerva")
        
        print(f"\n📝 Judging QID: {qid}")
        row_result = {"id": qid, "question": question, "answer": answer}

        for model_name in JUDGE_MODELS:
            print(f"   ⚖️  Judge ({model_name}) is thinking...")
            prompt = EVAL_PROMPT_IT.format(question=question, answer=answer)
            
            try:
                # --- تغییرات طلایی اینجاست (خط‌های پایین) ---
                response = ollama.chat(
                    model=model_name, 
                    messages=[{'role': 'user', 'content': prompt}],
                    # تنظیمات دقیق برای کاهش رم و افزایش دقت
                    options={
                        'num_ctx': 512,     # کاهش حافظه برای جلوگیری از پر شدن رم
                        'temperature': 0.1, # کاهش خلاقیت برای رعایت دقیق فرمت JSON
                        'num_thread': 4     # استفاده از 4 هسته پردازنده
                    }
                )
                
                score_data = clean_json_response(response['message']['content'])
                
                row_result[f"{model_name}_score"] = score_data.get('score_pertinenza')
                row_result[f"{model_name}_comment"] = score_data.get('commento')
                print(f"      ✅ Score: {score_data.get('score_pertinenza')}/5")
            except Exception as e:
                print(f"      ❌ Error: {e}")

        results.append(row_result)

    keys = results[0].keys() if results else []
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        dict_writer = csv.DictWriter(f, keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)

    print(f"\n🎉 Evaluation Complete! File saved: {OUT_CSV}")

if __name__ == "__main__":
    main()