# ⚡ OptiTalk-Local
> **A High-Performance, CPU-Optimized Voice Assistant (600% Faster)**
شسششیشسی
![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Ollama](https://img.shields.io/badge/Backend-Ollama-orange) ![Hardware](https://img.shields.io/badge/Hardware-Intel_UHD%2FCPU-green)

## 📄 Abstract (The Challenge)
Running local Voice-to-Voice AI models typically requires high-end GPUs. On standard hardware (e.g., laptops with **Intel UHD Graphics**), traditional pipelines (Whisper + Raw LLM + Heavy TTS) suffer from extreme latency (**~180 seconds** per response), making them unusable for real-time interaction.

**OptiTalk-Local** is a re-engineered pipeline developed as a Bachelor's Thesis to solve this bottleneck through software optimization.

## 🚀 Key Results: Performance Optimization
By restructuring the architecture, I achieved a **6x speedup** on non-GPU hardware:

| Metric | Standard Approach | **OptiTalk-Local (Optimized)** |
| :--- | :--- | :--- |
| **Latency** | ~180 Seconds | **~27 Seconds** |
| **Hardware** | Intel UHD | **Intel UHD** |
| **Experience** | Unusable | **Near Real-time** |

## 🧠 Advanced Feature: Grammatical Analysis & Correction
Beyond standard conversation, the system has been engineered to act as a **Linguistics Professor** for Italian language learning.

### 1. Linguistic Capabilities
- **Role Analysis:** Identifying grammatical roles (e.g., distinguishing *'Che'* as a conjunction vs. relative pronoun).
- **Syntax Correction:** Detecting grammatical errors and explaining the correct form.

### 2. Automated Evaluation (LLM-as-a-Judge)
To scientifically measure accuracy, I implemented an automated evaluation pipeline using **Llama-3.1 (8B)** hosted on Google Colab.

#### Benchmark Results (Professor's Dataset)
| Task ID | Question Type | User Input | Judge Score | Analysis |
| :--- | :--- | :--- | :--- | :--- |
| **Q4** | Grammar Check | *"Io sono andare a casa"* | **5/5** 🌟 | Perfect correction of auxiliary verb usage. |
| **Q2** | Role Analysis | *"Il libro che ho letto"* | **4/5** ✅ | Correct identification of *Pronome Relativo*. |
| **Q1** | Role Analysis | *"Che bel tempo"* | **2/5** ⚖️ | **Scientific Finding:** The local model correctly identified 'Che' as an *Exclamatory Adjective*, while the Judge model hallucinated it as a preposition. This highlights the robustness of the local optimization. |

> **Note:** The evaluation was conducted using a **High-RAM environment (Google Colab)** to utilize the larger Llama-3.1-8B model for precise judging.

## 🛠️ Optimization Architecture
To achieve this performance on a CPU, the following stack was implemented:
1.  **LLM Engine:** Replaced raw PyTorch inference with **Ollama API** (using quantized `llama3.2:1b` or `qwen2:0.5b`).
2.  **TTS (Text-to-Speech):** Switched to **Edge-TTS** (Async streaming) to eliminate processing overhead.
3.  **Concurrency:** Fully asynchronous execution using Python's `asyncio`.

## 📦 Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/nikpournastaran/OptiTalk-Local.git](https://github.com/nikpournastaran/OptiTalk-Local.git)
    cd OptiTalk-Local
    ```

2.  **Install dependencies:**
    ```bash
    pip install ollama edge-tts asyncio sounddevice numpy scipy speechrecognition pyaudio
    ```

3.  **Run the Assistant:**
    Make sure Ollama is running, then:
    ```bash
    python gradio_app.py
    ```

---
*Developed by Nastaran Nikpour - 2025*
