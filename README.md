# 🧠 AI Language Tutor with Voice

An intelligent, voice-enabled, multilingual AI language assistant built with **LangChain**, **Ollama**, **gTTS**, **speech recognition**, and **Streamlit**.

This app allows users to:
- 🎙️ Speak or type in one language
- 🔄 Translate to another language
- 🧠 Understand grammar & cultural nuance using LLM explanations
- 🔊 Hear the translated sentence spoken aloud
- 💬 See a complete chat-like memory of past interactions with audio playback

---

## 🚀 Features

- ✅ **Text or voice input** (STT via Google)
- ✅ **Multi-language translation** using Hugging Face MarianMT
- ✅ **LLM-powered grammar and nuance explanation** via Ollama (LLaMA 3)
- ✅ **Text-to-speech** via gTTS
- ✅ **Session memory** with chat-style UI and audio playback
- ✅ **Two-way translation**: English ↔ Hindi / French / Spanish

---

## 🛠 Tech Stack

| Layer        | Tools Used |  
|--------------|------------|  
| UI           | Streamlit |  
| Translation  | Hugging Face MarianMT (via Transformers + LangChain) |  
| TTS (speech) | gTTS (Google Text-to-Speech) |  
| STT (input)  | `speech_recognition` (Google Speech API) |  
| LLM          | Ollama + LangChain (LLaMA 3 8B) |  
| Logic Engine | LangChain Runnables + PromptTemplate |  
| Audio Playback | `pygame` |  

---

## 🔧 How to Run Locally

### 1. Clone the repo  

git clone https://github.com/shreyaa-mohan/AI-Language-Tutor.git
cd ai-language-tutor  
### 2. Install dependencies  

pip install -r requirements.txt

### 3. Start Ollama (for the explanation LLM)  
Make sure Ollama is installed and running:  
ollama run llama3
### 4. Run the Streamlit app

streamlit run app.py   

### 🌍 Supported Language Pairs

English	to Hindi, French, Spanish  
Hindi to English  
French to English  
Spanish to English  

### ✨ Example Use Cases
- Speak in Hindi → get English translation → hear it aloud → understand the grammar changes
- Type in English → get French translation → learn grammar + tone → hear pronunciation
- Act as a language tutor in multilingual contexts
