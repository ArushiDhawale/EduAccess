import speech_recognition as sr
import os
import io

def is_microphone_available():
    """
    Checks if PyAudio is installed and if a microphone is accessible.
    """
    try:
        # Check if pyaudio can be imported
        import pyaudio
        # Try to list microphones
        mics = sr.Microphone.list_microphone_names()
        return len(mics) > 0
    except Exception:
        return False

def speech_to_text_from_mic():
    """
    Records audio from the local system microphone and transcribes it.
    This works when running Streamlit locally on a machine with a microphone.
    
    Returns:
        tuple: (transcribed_text, error_message)
    """
    if not is_microphone_available():
        return None, "Microphone device or PyAudio library is not available on this server."
        
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            # Adjust for ambient noise before listening
            recognizer.adjust_for_ambient_noise(source, duration=1.0)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=8)
            
        # Transcribe using Google's free Web Speech API
        text = recognizer.recognize_google(audio)
        return text, None
    except sr.WaitTimeoutError:
        return None, "Listening timed out. No speech detected."
    except sr.UnknownValueError:
        return None, "Could not understand the audio."
    except sr.RequestError as e:
        return None, f"Speech recognition service error: {e}"
    except Exception as e:
        return None, f"An unexpected error occurred: {e}"


def speech_to_text_from_audio(audio_bytes):
    """Transcribe browser-recorded WAV audio bytes."""
    if not audio_bytes:
        return None, "No speech recording was provided."

    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio = recognizer.record(source)
        return recognizer.recognize_google(audio), None
    except sr.UnknownValueError:
        return None, "Could not understand the audio. Please record again."
    except sr.RequestError as e:
        return None, f"Speech recognition service error: {e}"
    except Exception as e:
        return None, f"Could not read the recording: {e}"
