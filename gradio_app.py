import gradio as gr
import speech_recognition as sr
import ollama
import edge_tts
import asyncio
import os
import shutil
from datetime import datetime

# --- Configuration ---
# Using the lightweight quantized model for CPU optimization
LLM_MODEL = "qwen2:0.5b"   # Alternative: "llama3.2:1b"
VOICE = "it-IT-ElsaNeural" # High-quality Italian Neural Voice (Edge-TTS)
LANG_CODE = "it-IT"        # Google Speech Recognition Language Code

# --- 1. Text-to-Speech Engine (Edge-TTS) ---
async def generate_audio(text):
    """
    Generates audio from text using Microsoft Edge TTS (Async).
    """
    output_file = f"response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
    communicate = edge_tts.Communicate(text, VOICE)
    await communicate.save(output_file)
    return output_file

def text_to_speech_sync(text):
    """
    Wrapper to run the async TTS function within synchronous Gradio.
    """
    return asyncio.run(generate_audio(text))

# --- 2. LLM Engine (Ollama) ---
def query_ollama(prompt):
    """
    Sends the prompt to the local Ollama instance with a Linguistics Professor persona.
    """
    try:
        response = ollama.chat(model=LLM_MODEL, messages=[
            # دستورالعمل جدید: تو یک استاد زبان‌شناسی هستی
            {'role': 'system', 'content': """
Sei un esperto professore di linguistica italiana. Il tuo compito è duplice:
1. Analisi Grammaticale: Identifica il ruolo grammaticale delle parole specifiche (es. 'che' può essere congiunzione, pronome, aggettivo).
2. Correzione Grammaticale: Valuta se una frase è corretta o sbagliata. Se è sbagliata, spiega perché.

Rispondi in modo preciso, tecnico ma breve.
"""},
            {'role': 'user', 'content': prompt},
        ])
        return response['message']['content']
    except Exception as e:
        return f"Ollama Error: {str(e)}"

# --- 3. Main Processing Pipeline ---
def transcribe_and_respond(audio_path):
    """
    Handles the full pipeline: Audio Input -> STT -> LLM -> TTS -> Audio Output.
    """
    if audio_path is None:
        return "No audio detected.", None

    # A) Save audio file locally (Requirement for analysis)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = "recorded_audio"
    os.makedirs(save_dir, exist_ok=True)
    saved_path = os.path.join(save_dir, f"input_{timestamp}.wav")
    shutil.copy(audio_path, saved_path)

    # B) Speech-to-Text (Google SR - Optimized for CPU)
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            # Transcribe audio in Italian
            text = recognizer.recognize_google(audio_data, language=LANG_CODE)
    except sr.UnknownValueError:
        return "Could not understand audio.", None
    except Exception as e:
        return f"STT Error: {str(e)}", None

    # C) Generate response via LLM
    ai_response = query_ollama(text)

    # D) Convert text response to audio (TTS)
    audio_output_path = text_to_speech_sync(ai_response)

    # Final Output: Chat log + Audio response
    log_text = f"🗣️ User: {text}\n🤖 Assistant: {ai_response}"
    return log_text, audio_output_path

# --- 4. Web Interface (Gradio) ---
demo = gr.Interface(
    fn=transcribe_and_respond,
    inputs=gr.Audio(type="filepath", label="Record Voice (Click to Start)"),
    outputs=[
        gr.Textbox(label="Conversation Log"),
        gr.Audio(label="Voice Response") 
    ],
    title="🚀 OptiTalk: High-Performance Local Assistant",
    description="Optimized pipeline using Ollama + EdgeTTS + Gradio (Running on Intel UHD)."
)

if __name__ == "__main__":
    print("🚀 Gradio Server Starting...")
    demo.launch(server_name="0.0.0.0", server_port=7860)