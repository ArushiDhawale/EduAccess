import streamlit as st
import cv2
import numpy as np
import os
import time
import json
from collections import deque

# Import custom services
from services.hand_landmarks import HandLandmarkExtractor, normalize_landmarks
from services.sign_predictor import ISLSignPredictor, PredictionSmoother
from services.isl_nlp import load_sign_vocabulary, text_to_gloss, get_sign_video
from services.speech_to_text import speech_to_text_from_audio
from services.text_to_speech import text_to_speech_bytes
from ml.generate_mock_data import generate_synthetic_dataset
from ml.train_from_videos import extract_video_dataset
from ml.train import train_model

# Page configuration
st.set_page_config(
    page_title="Two-Way Indian Sign Language Translator",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
    <style>
        /* General Theme override */
        .stApp {
            background: linear-gradient(135deg, #0d1b2a 0%, #1b263b 100%);
            color: #e0e1dd;
            font-family: 'Outfit', 'Inter', sans-serif;
        }
        
        /* Glassmorphic Container */
        .glass-card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 24px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            margin-bottom: 20px;
        }
        
        /* Heading styling */
        h1, h2, h3 {
            color: #52b788 !important;
            font-weight: 700;
        }
        
        .main-title {
            text-align: center;
            font-size: 3rem;
            background: linear-gradient(90deg, #52b788, #74c69d, #95d5b2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 5px;
            font-weight: 800;
        }
        
        .subtitle {
            text-align: center;
            color: #a3b18a;
            font-size: 1.2rem;
            margin-bottom: 30px;
        }
        
        /* Custom status indicators */
        .status-badge-ok {
            background-color: rgba(82, 183, 136, 0.15);
            color: #52b788;
            padding: 6px 12px;
            border-radius: 8px;
            border: 1px solid rgba(82, 183, 136, 0.3);
            font-size: 0.9rem;
            display: inline-block;
            margin-right: 5px;
        }
        
        .status-badge-warn {
            background-color: rgba(244, 162, 97, 0.15);
            color: #f4a261;
            padding: 6px 12px;
            border-radius: 8px;
            border: 1px solid rgba(244, 162, 97, 0.3);
            font-size: 0.9rem;
            display: inline-block;
            margin-right: 5px;
        }

        .status-badge-error {
            background-color: rgba(230, 57, 70, 0.15);
            color: #e63946;
            padding: 6px 12px;
            border-radius: 8px;
            border: 1px solid rgba(230, 57, 70, 0.3);
            font-size: 0.9rem;
            display: inline-block;
            margin-right: 5px;
        }
        
        /* Interactive element transitions */
        .stButton>button {
            border-radius: 8px;
            transition: all 0.3s ease;
            font-weight: 600;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(82, 183, 136, 0.3);
        }
        
        /* Text representation card */
        .translation-card {
            background: rgba(82, 183, 136, 0.08);
            border-radius: 12px;
            border-left: 5px solid #52b788;
            padding: 15px;
            margin-top: 15px;
            font-size: 1.3rem;
            font-weight: 500;
        }
        
        .unsupported-card {
            background: rgba(230, 57, 70, 0.08);
            border-radius: 12px;
            border-left: 5px solid #e63946;
            padding: 15px;
            margin-top: 15px;
            font-size: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize Session State
if "translation_sequence" not in st.session_state:
    st.session_state.translation_sequence = []
if "last_stable_sign" not in st.session_state:
    st.session_state.last_stable_sign = None
if "current_prediction" not in st.session_state:
    st.session_state.current_prediction = None
if "confidence" not in st.session_state:
    st.session_state.confidence = 0.0
if "text_input" not in st.session_state:
    st.session_state.text_input = ""
if "voice_transcript" not in st.session_state:
    st.session_state.voice_transcript = None
if "voice_error" not in st.session_state:
    st.session_state.voice_error = None
if "audio_to_play" not in st.session_state:
    st.session_state.audio_to_play = None
if "model_trained_trigger" not in st.session_state:
    st.session_state.model_trained_trigger = False

# Configuration Paths
VOCAB_PATH = "config/sign_vocabulary.json"
MODEL_PATH = "models/isl_static_classifier.pkl"
LABEL_MAPPING_PATH = "models/label_mapping.json"

# Load models and services safely
@st.cache_resource
def load_predictor(trained_trigger=False):
    return ISLSignPredictor(model_path=MODEL_PATH, label_mapping_path=LABEL_MAPPING_PATH)

@st.cache_resource
def load_extractor():
    try:
        return HandLandmarkExtractor(min_detection_confidence=0.6, min_tracking_confidence=0.6)
    except Exception as e:
        st.sidebar.error(f"Failed to initialize MediaPipe Hands: {e}")
        return None

# Sidebar Content
st.sidebar.markdown("### 🤟 Two-Way ISL System")

# 1. Model Status Indicators
st.sidebar.markdown("#### System Status")
vocab = load_sign_vocabulary(VOCAB_PATH)

# Vocabulary Status
if vocab:
    st.sidebar.markdown(f'<div class="status-badge-ok">✓ Vocabulary loaded ({len(vocab)} signs)</div>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('<div class="status-badge-error">✗ Vocabulary configuration not found</div>', unsafe_allow_html=True)

# MediaPipe Status
extractor = load_extractor()
if extractor:
    st.sidebar.markdown('<div class="status-badge-ok">✓ MediaPipe Hands loaded</div>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('<div class="status-badge-error">✗ MediaPipe Hands unavailable</div>', unsafe_allow_html=True)

# ML Classifier Status
predictor = load_predictor(st.session_state.model_trained_trigger)
model_loaded = predictor.is_model_loaded()

if model_loaded:
    st.sidebar.markdown('<div class="status-badge-ok">✓ ML Classifier loaded</div>', unsafe_allow_html=True)
    trained_labels = set(predictor.reverse_mapping.values())
    missing_training = sorted(set(vocab) - trained_labels)
    if missing_training:
        st.sidebar.warning(
            "Add videos for webcam training: " + ", ".join(missing_training)
        )
else:
    st.sidebar.markdown('<div class="status-badge-warn">⚠ ML Classifier not found</div>', unsafe_allow_html=True)
    st.sidebar.markdown("""
        <div style="font-size:0.85rem; color:#f4a261; margin-top:8px;">
        Prediction is disabled because the trained model was not found at <code>models/isl_static_classifier.pkl</code>.
        </div>
    """, unsafe_allow_html=True)

# 2. Demo training button
if not model_loaded:
    st.sidebar.markdown("#### Demo Mode Setup")
    st.sidebar.info("You can generate a synthetic dataset and train a Random Forest model instantly to try out the Camera Translation.")
    if st.sidebar.button("🚀 Generate Mock Data & Train Model"):
        with st.sidebar.status("Setting up demo model...", expanded=True) as status:
            status.write("Generating mock hand landmarks...")
            generate_synthetic_dataset(output_path="data/isl_dataset.csv", vocabulary_path=VOCAB_PATH, samples_per_sign=30)
            status.write("Training Random Forest classifier...")
            success = train_model(csv_path="data/isl_dataset.csv", model_dir="models")
            if success:
                st.session_state.model_trained_trigger = True
                status.update(label="Training complete! Reloading model...", state="complete", expanded=False)
                st.rerun()
            else:
                status.update(label="Training failed.", state="error", expanded=True)

st.sidebar.markdown("#### Train From Sign Videos")
st.sidebar.caption("Uses MP4 filenames as labels and extracts real hand landmarks.")
if st.sidebar.button("🎬 Train from Sign Videos", use_container_width=True):
    with st.sidebar.status("Training from sign videos...", expanded=True) as status:
        extracted = extract_video_dataset(
            video_dir="assets/signs",
            output_path="data/isl_dataset.csv",
        )
        if extracted:
            success = train_model(csv_path="data/isl_dataset.csv", model_dir="models")
        else:
            success = False
        if success:
            st.session_state.model_trained_trigger = not st.session_state.model_trained_trigger
            status.update(label="Video model trained. Reloading...", state="complete", expanded=False)
            st.rerun()
        else:
            status.update(label="Video training failed.", state="error", expanded=True)

# Configurable Hyperparameters
st.sidebar.markdown("#### Classifier Settings")
CONFIDENCE_THRESHOLD = st.sidebar.slider("Confidence Threshold", min_value=0.5, max_value=1.0, value=0.75, step=0.05)
SMOOTHING_WINDOW = st.sidebar.slider("Smoothing Window Size", min_value=3, max_value=15, value=8)
MIN_STABLE_FRAMES = st.sidebar.slider("Min Stable Frames", min_value=2, max_value=10, value=4)

# Instantiate prediction smoother
smoother = PredictionSmoother(
    smoothing_window=SMOOTHING_WINDOW,
    min_stable_frames=MIN_STABLE_FRAMES,
    confidence_threshold=CONFIDENCE_THRESHOLD
)

# Header
st.markdown('<div class="main-title">Two-Way Indian Sign Language (ISL) Translator</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Empowering communication using real-time machine learning and video synthesis</div>', unsafe_allow_html=True)

# Create Main Tabs
tab1, tab2 = st.tabs(["🎥 Sign → Text", "✍️ Text/Speech → Sign"])

# ==================== TAB 1: Sign → Text ====================
with tab1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### Sign to Text & Speech")
    st.write("Position your hand in front of the camera. The system will detect hand landmarks and output the translated sign when stable.")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Camera mode selector
        camera_mode = st.radio(
            "Select Camera Mode",
            ["Disabled", "Live Webcam Feed (Local/Dev)"],
            index=0,
            horizontal=True
        )
        
        frame_placeholder = st.empty()
        
        # Check if extractor or predictor is missing
        if camera_mode != "Disabled" and not extractor:
            st.error("MediaPipe hands model is not loaded.")
        elif camera_mode != "Disabled" and not model_loaded:
            st.warning("Prediction is disabled: trained model is missing. Please set up the demo model in the sidebar.")
        
        # 1. LIVE WEBCAM FEED (LOCAL)
        elif camera_mode == "Live Webcam Feed (Local/Dev)":
            run_cam = st.checkbox("Start Live Stream", value=True)
            if run_cam:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("Camera unavailable. Check browser/camera permissions and close other apps using the camera.")
                else:
                    try:
                        while run_cam:
                            ret, frame = cap.read()
                            if not ret:
                                st.error("Failed to read frames from camera.")
                                break
                            
                            # Flip frame horizontally for natural mirror view
                            frame = cv2.flip(frame, 1)
                            
                            # Process frame
                            landmarks, annotated_frame = extractor.extract_landmarks(frame)
                            
                            # Run prediction
                            if landmarks is not None and model_loaded:
                                normalized = normalize_landmarks(landmarks)
                                prediction = predictor.predict(normalized)
                                
                                if prediction:
                                    st.session_state.current_prediction = prediction["label"]
                                    st.session_state.confidence = prediction["confidence"]
                                    
                                    # Smoothing
                                    stable_label = smoother.add_prediction(prediction)
                                    if stable_label:
                                        st.session_state.last_stable_sign = stable_label
                                        # Auto-add sign to sequence if enabled
                                        # (Optionally check if it's not a duplicate of the last inserted)
                                        if not st.session_state.translation_sequence or st.session_state.translation_sequence[-1] != stable_label:
                                            st.session_state.translation_sequence.append(stable_label)
                                else:
                                    st.session_state.current_prediction = None
                                    st.session_state.confidence = 0.0
                            else:
                                st.session_state.current_prediction = None
                                st.session_state.confidence = 0.0
                                smoother.add_prediction(None)
                                
                            # Draw prediction overlay in the frame
                            if st.session_state.current_prediction:
                                cv2.putText(
                                    annotated_frame,
                                    f"{st.session_state.current_prediction} ({st.session_state.confidence*100:.1f}%)",
                                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (82, 183, 136), 2
                                )
                                
                            # Display annotated frame
                            frame_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)
                            
                            # Yield execution for a brief moment
                            time.sleep(0.03)
                    finally:
                        cap.release()
                        frame_placeholder.empty()
            
    with col2:
        st.markdown("#### Real-time Detection Status")
        
        # Display current frame prediction
        pred_label = st.session_state.current_prediction
        pred_conf = st.session_state.confidence
        
        if pred_label:
            st.metric(label="Current Sign Detected", value=pred_label)
            st.metric(label="Confidence", value=f"{pred_conf*100:.1f}%")
        else:
            st.info("No confident sign detected (pose your hand in camera view)")
            
        st.markdown("---")
        st.markdown("#### Translation Sentence Builder")
        
        # Manual confirmed sign add
        sub_col1, sub_col2 = st.columns(2)
        with sub_col1:
            if st.button("➕ Confirm/Add Sign", use_container_width=True):
                if pred_label:
                    if not st.session_state.translation_sequence or st.session_state.translation_sequence[-1] != pred_label:
                        st.session_state.translation_sequence.append(pred_label)
                    else:
                        st.warning("Prevented duplicate sign insertion.")
                else:
                    st.error("No detected sign to add.")
        with sub_col2:
            if st.button("↩️ Undo last sign", use_container_width=True):
                if st.session_state.translation_sequence:
                    st.session_state.translation_sequence.pop()
                    st.rerun()
                    
        if st.button("🗑️ Clear Translation", use_container_width=True):
            st.session_state.translation_sequence = []
            st.session_state.audio_to_play = None
            st.rerun()
            
        # Display translation sequence
        st.markdown("##### Current Translated Sentence:")
        if st.session_state.translation_sequence:
            sentence = " ".join(st.session_state.translation_sequence).replace("_", " ").title()
            st.markdown(f'<div class="translation-card">🔊 {sentence}</div>', unsafe_allow_html=True)
            
            # Speak translation
            if st.button("🔊 Speak Translation", use_container_width=True):
                with st.spinner("Synthesizing speech..."):
                    audio_bytes, err = text_to_speech_bytes(sentence)
                    if audio_bytes:
                        st.session_state.audio_to_play = audio_bytes
                    else:
                        st.error(err)
                        
            if st.session_state.audio_to_play:
                st.audio(st.session_state.audio_to_play, format="audio/mp3")
        else:
            st.write("(Sentence is currently empty. Add signs to build translation)")
            
    st.markdown('</div>', unsafe_allow_html=True)


# ==================== TAB 2: Text/Speech → Sign ====================
with tab2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### Text or Speech to Sign Sequence")
    st.write("Enter an English sentence or use voice input to convert it into a sequence of corresponding Indian Sign Language videos.")
    
    input_mode = st.radio("Input method", ["Text", "Speech"], horizontal=True, key="input_mode")

    if input_mode == "Text":
        input_text = st.text_input(
            "Enter English text:",
            key="text_input",
            placeholder="e.g. hello thank you doctor please"
        )
    else:
        input_text = ""
        st.info("Speech mode is active. Record your sentence below, then translate it.")
        audio_file = st.audio_input("Record speech")
        if audio_file is not None:
            with st.spinner("Transcribing recording..."):
                text, err = speech_to_text_from_audio(audio_file.getvalue())
            if text:
                st.session_state.voice_transcript = text
                st.session_state.voice_error = None
                st.success(f"Recorded speech: \"{text}\"")
            else:
                st.session_state.voice_error = err
                st.warning(err)
    
    # Translate Action
    if st.button("Translate to ISL Sign Sequence", type="primary", use_container_width=True):
        source_text = input_text.strip() if input_mode == "Text" else (st.session_state.voice_transcript or "").strip()
        if source_text:
            # Process conversion
            results = text_to_gloss(source_text, config_path=VOCAB_PATH)
            
            glosses = results["glosses"]
            unsupported = results["unsupported"]
            
            st.markdown("#### Conversion Result")
            
            # Show gloss sequence
            st.markdown("##### ISL Gloss Sequence:")
            if glosses:
                st.info(" ➔ ".join(glosses))
            else:
                st.warning("No matching ISL signs could be extracted from input.")
                
            # Show unsupported words
            if unsupported:
                st.markdown('<div class="unsupported-card">', unsafe_allow_html=True)
                st.markdown(f"⚠️ **Unsupported words (skipped):** {', '.join(unsupported)}")
                st.markdown("</div>", unsafe_allow_html=True)
                
            # Display Sign Videos
            st.markdown("##### Sign Sequence Output:")
            if glosses:
                # Create streamlit columns for each sign in sequence
                cols = st.columns(len(glosses))
                
                for idx, gloss in enumerate(glosses):
                    with cols[idx]:
                        st.markdown(f"<div style='text-align: center; font-weight: bold;'>{gloss}</div>", unsafe_allow_html=True)
                        
                        video_path = get_sign_video(gloss, config_path=VOCAB_PATH)
                        if video_path:
                            # Render video component
                            st.video(video_path)
                        else:
                            # Render warning / missing asset placeholder
                            st.warning(f"Sign video unavailable for: {gloss}")
                            st.markdown("""
                                <div style='border: 2px dashed rgba(244, 162, 97, 0.4); border-radius: 8px; height: 120px; display: flex; align-items: center; justify-content: center; color: #f4a261; text-align: center; padding: 10px; font-size: 0.85rem;'>
                                Video file missing at assets/signs/
                                </div>
                            """, unsafe_allow_html=True)
            else:
                st.write("Provide an input text containing valid ISL vocabulary words (see sidebar).")
        else:
            st.error("Please enter text or record speech first.")
            
    st.markdown('</div>', unsafe_allow_html=True)

# Footer info
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #a3b18a; font-size: 0.85rem; padding-bottom: 20px;">
    Two-Way ISL Translator Module • Created as a standalone component for easy project integration.
    </div>
""", unsafe_allow_html=True)
