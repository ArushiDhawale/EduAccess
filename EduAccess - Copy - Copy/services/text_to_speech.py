from gtts import gTTS
import io

def text_to_speech_bytes(text, lang='en'):
    """
    Converts English text into speech audio bytes using gTTS.
    Returns bytes that can be directly played in Streamlit using st.audio().
    
    Args:
        text: text to be spoken
        lang: language code (default 'en')
    Returns:
        tuple: (audio_bytes, error_message)
    """
    if not text or not text.strip():
        return None, "Empty text provided."
        
    try:
        # Generate speech in memory
        tts = gTTS(text=text, lang=lang)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        audio_bytes = fp.read()
        return audio_bytes, None
    except Exception as e:
        return None, f"Text-to-speech generation failed: {e}"
