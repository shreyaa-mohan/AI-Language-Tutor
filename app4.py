import streamlit as st
import tempfile
import os
import time
import io # <-- ADD for in-memory files

# For Audio
from gtts import gTTS
import pygame

# For Speech-to-Text
import speech_recognition as sr

# For Translation & Explanation
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_huggingface import HuggingFacePipeline 
from langchain_ollama import ChatOllama

# ----------------------------
# Model Loading (Cached)
# ----------------------------
@st.cache_resource
def load_translation_model(model_name):
    # (Unchanged)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        pipe = pipeline("translation", model=model, tokenizer=tokenizer)
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        st.error(f"Failed to load translation model '{model_name}'. Error: {e}")
        st.stop()

@st.cache_resource
def get_explanation_llm():
    # (Unchanged)
    try:
        return ChatOllama(model="llama3:8b", temperature=0.3)
    except Exception as e:
        st.error(f"Failed to connect to Ollama. Is it running? Error: {e}")
        st.stop()

# ----------------------------
# Core Logic Functions
# ----------------------------
def get_translation_explanation(original_text, translated_text, source_lang, target_lang, llm):
    # (Unchanged)
    explanation_prompt = ChatPromptTemplate.from_template(
        """
        You are a language tutor. A user translated a sentence. Explain the grammatical and cultural nuances.
        Original Text ({source_lang}): "{original_text}"
        Translated Text ({target_lang}): "{translated_text}"
        Provide a brief, bullet-pointed explanation of:
        1. Key Grammar Changes (e.g., word order, verb tense).
        2. Cultural Nuance & Formality.
        Keep it concise and easy to understand.
        """
    )
    explanation_chain = explanation_prompt | llm | StrOutputParser()
    return explanation_chain.invoke({
        "source_lang": source_lang, "target_lang": target_lang,
        "original_text": original_text, "translated_text": translated_text,
    })

# --- MODIFIED: Text-to-Speech (TTS) ---
def text_to_speech(text, lang_code):
    """
    Generates audio from text using gTTS and returns the audio data as bytes.
    It also plays the audio directly.
    """
    try:
        # Create an in-memory file
        audio_fp = io.BytesIO()
        tts = gTTS(text=text, lang=lang_code, slow=False)
        tts.write_to_fp(audio_fp)
        
        # Rewind the file pointer to the beginning to read from it
        audio_fp.seek(0)
        audio_bytes = audio_fp.read()

        # Play the audio using pygame
        pygame.mixer.init()
        # Load the audio from the in-memory bytes object
        pygame.mixer.music.load(io.BytesIO(audio_bytes))
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        
        pygame.mixer.quit()
        
        # Return the audio data so it can be stored
        return audio_bytes

    except Exception as e:
        st.error(f"Text-to-Speech failed: {e}")
        return None

def speech_to_text(lang_code):
    # (Unchanged)
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening...")
        r.pause_threshold = 1.5
        r.adjust_for_ambient_noise(source, duration=1)
        try:
            audio = r.listen(source, timeout=10, phrase_time_limit=15)
        except sr.WaitTimeoutError:
            st.warning("No speech detected.")
            return ""
    st.info("Processing...")
    try:
        return r.recognize_google(audio, language=lang_code)
    except sr.UnknownValueError:
        st.warning("Could not understand the audio.")
        return ""
    except sr.RequestError as e:
        st.error(f"Speech service error; {e}")
        return ""

# ----------------------------
# UI Configuration
# ----------------------------
st.set_page_config(page_title="AI Language Tutor", layout="wide")
st.title("🧠 AI Language Tutor with Audio ")

# --- Language Mappings ---
translation_models = {
    ("English", "Hindi"): "Helsinki-NLP/opus-mt-en-hi", ("English", "French"): "Helsinki-NLP/opus-mt-en-fr",
    ("English", "Spanish"): "Helsinki-NLP/opus-mt-es-en", ("Hindi", "English"): "Helsinki-NLP/opus-mt-hi-en",
    ("French", "English"): "Helsinki-NLP/opus-mt-fr-en", ("Spanish", "English"): "Helsinki-NLP/opus-mt-es-en",
}
speech_lang_codes = { "Hindi": ("hi", "hi-IN"), "French": ("fr", "fr-FR"), "Spanish": "es" }

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- MODIFIED: Display Chat History with Audio ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "original" in message:
            st.markdown(f"**Original ({message.get('lang', '')}):**\n> {message['original']}")
        if "translation" in message:
            st.markdown(f"**Translation ({message.get('lang', '')}):**\n> {message['translation']}")
        
        # If the message has audio data, display the st.audio widget
        if "audio" in message and message["audio"] is not None:
            st.audio(message["audio"], format="audio/mp3")

        if "explanation" in message:
            with st.expander("Show Grammar & Nuance Explanation"):
                st.markdown(message["explanation"])

# --- Main App Layout ---
st.markdown("---")
explanation_llm = get_explanation_llm()
col1, col2 = st.columns(2)

with col1:
    st.header("English ➡️ Other Language")
    target_lang_1 = st.selectbox("Translate English to:", ["Hindi", "French", "Spanish"], key="lang_select_1")
    english_text_input = st.text_area("Enter text in English:", value="The weather is wonderful today.", key="en_text")

    if st.button("Translate, Explain & Speak", key="translate_button", type="primary"):
        if english_text_input.strip():
            st.session_state.messages.append({"role": "user", "original": english_text_input, "lang": "English"})

            model_name = translation_models[("English", target_lang_1)]
            with st.spinner(f"Translating to {target_lang_1}..."):
                translator = load_translation_model(model_name)
                translated_text = translator.invoke(english_text_input)
            
            with st.spinner("Generating explanation..."):
                explanation = get_translation_explanation(english_text_input, translated_text, "English", target_lang_1, explanation_llm)

            # --- MODIFIED: Get audio data back ---
            gtts_code = speech_lang_codes[target_lang_1][0] if isinstance(speech_lang_codes[target_lang_1], tuple) else speech_lang_codes[target_lang_1]
            audio_bytes = text_to_speech(translated_text, gtts_code)

            # --- MODIFIED: Store audio in session state ---
            st.session_state.messages.append({
                "role": "assistant",
                "translation": translated_text,
                "explanation": explanation,
                "audio": audio_bytes, # <-- Store the audio data
                "lang": target_lang_1
            })
            st.rerun()
        else:
            st.warning("Please enter text to translate.")

with col2:
    st.header("Other Language ➡️ English")
    source_lang_2 = st.selectbox("Translate from:", ["Hindi", "French", "Spanish"], key="lang_select_2")

    if st.button(f"Record & Analyze (Speak in {source_lang_2})", key="record_button", type="primary"):
        sr_code = speech_lang_codes[source_lang_2][1] if isinstance(speech_lang_codes[source_lang_2], tuple) else speech_lang_codes[source_lang_2]
        transcribed_text = speech_to_text(sr_code)

        if transcribed_text:
            st.session_state.messages.append({"role": "user", "original": transcribed_text, "lang": source_lang_2})
            
            model_name = translation_models[(source_lang_2, "English")]
            with st.spinner("Translating back to English..."):
                translator_to_english = load_translation_model(model_name)
                english_translation = translator_to_english.invoke(transcribed_text)
            
            with st.spinner("Generating explanation..."):
                explanation = get_translation_explanation(transcribed_text, english_translation, source_lang_2, "English", explanation_llm)
            
            # --- MODIFIED: Get audio data back ---
            audio_bytes = text_to_speech(english_translation, "en")

            # --- MODIFIED: Store audio in session state ---
            st.session_state.messages.append({
                "role": "assistant",
                "translation": english_translation,
                "explanation": explanation,
                "audio": audio_bytes, # <-- Store the audio data
                "lang": "English"
            })
            st.rerun()