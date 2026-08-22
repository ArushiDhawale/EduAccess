from pathlib import Path
import sys

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
FEATURE_ROOT = PROJECT_ROOT / "EduAccess - Copy - Copy"
VOCAB_PATH = FEATURE_ROOT / "config" / "sign_vocabulary.json"
MODEL_PATH = FEATURE_ROOT / "models" / "isl_static_classifier.pkl"
LABEL_MAPPING_PATH = FEATURE_ROOT / "models" / "label_mapping.json"


@st.cache_resource
def _load_isl_services():
    """Load the copied feature's reusable services without running its app."""
    feature_root = str(FEATURE_ROOT)
    if feature_root not in sys.path:
        sys.path.insert(0, feature_root)

    try:
        from services.hand_landmarks import HandLandmarkExtractor, normalize_landmarks
        from services.sign_predictor import ISLSignPredictor, PredictionSmoother
        from services.isl_nlp import load_sign_vocabulary, text_to_gloss
    except Exception as error:
        return None, error

    return {
        "HandLandmarkExtractor": HandLandmarkExtractor,
        "normalize_landmarks": normalize_landmarks,
        "ISLSignPredictor": ISLSignPredictor,
        "PredictionSmoother": PredictionSmoother,
        "load_sign_vocabulary": load_sign_vocabulary,
        "text_to_gloss": text_to_gloss,
    }, None


@st.cache_resource
def _load_detector():
    services, error = _load_isl_services()
    if services is None:
        return None, error
    try:
        return services["HandLandmarkExtractor"](
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        ), None
    except Exception as detector_error:
        return None, detector_error


@st.cache_resource
def _load_predictor():
    services, error = _load_isl_services()
    if services is None:
        return None, error
    try:
        return services["ISLSignPredictor"](
            model_path=str(MODEL_PATH),
            label_mapping_path=str(LABEL_MAPPING_PATH),
        ), None
    except Exception as predictor_error:
        return None, predictor_error


def _get_video_path(gloss, vocabulary):
    sign_info = vocabulary.get(gloss, {})
    configured_path = sign_info.get("video")
    if not configured_path:
        return None
    video_path = FEATURE_ROOT / configured_path
    return video_path if video_path.is_file() else None


def _init_isl_state():
    defaults = {
        "isl_current_sign": None,
        "isl_confidence": 0.0,
        "isl_translation": [],
        "isl_gloss_sequence": [],
        "isl_unsupported_words": [],
        "isl_input_text": "",
        "isl_voice_transcript": None,
        "isl_audio_to_play": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _render_sign_to_text(services, vocabulary, detector, predictor):
    st.subheader("Sign to Text and Speech")
    st.write("Capture a hand sign to extract landmarks and classify it with the existing ISL model.")

    camera_frame = st.camera_input("Capture sign")
    if camera_frame is not None:
        try:
            import cv2
            import numpy as np

            frame = cv2.imdecode(
                np.frombuffer(camera_frame.getvalue(), dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            landmarks, annotated_frame = detector.extract_landmarks(frame) if detector else (None, None)
            if annotated_frame is not None:
                st.image(annotated_frame, channels="BGR", caption="Hand landmark visualization")

            if landmarks is None:
                st.warning("No hand detected. Capture another frame with your hand visible.")
            else:
                normalized = services["normalize_landmarks"](landmarks)
                st.caption(f"Detected {len(landmarks)} landmarks and {len(normalized)} normalized features.")
                prediction = predictor.predict(normalized) if predictor and predictor.is_model_loaded() else None
                if prediction:
                    st.session_state.isl_current_sign = prediction["label"]
                    st.session_state.isl_confidence = prediction["confidence"]
                    st.success(f"Detected sign: {prediction['label']}")
                else:
                    st.warning("ISL model unavailable or unable to classify this frame.")
        except Exception as error:
            st.error(f"ISL camera processing failed: {error}")

    current_sign = st.session_state.isl_current_sign
    if current_sign:
        st.metric("Current sign", current_sign)
        st.metric("Confidence", f"{st.session_state.isl_confidence:.1%}")

    add_col, undo_col, clear_col = st.columns(3)
    with add_col:
        if st.button("Add sign", use_container_width=True):
            if current_sign and (not st.session_state.isl_translation or st.session_state.isl_translation[-1] != current_sign):
                st.session_state.isl_translation.append(current_sign)
            elif not current_sign:
                st.warning("Capture a confident sign first.")
    with undo_col:
        if st.button("Undo", use_container_width=True) and st.session_state.isl_translation:
            st.session_state.isl_translation.pop()
    with clear_col:
        if st.button("Clear", use_container_width=True):
            st.session_state.isl_translation = []
            st.session_state.isl_current_sign = None
            st.session_state.isl_confidence = 0.0
            st.session_state.isl_audio_to_play = None

    sentence = " ".join(st.session_state.isl_translation).replace("_", " ").title()
    st.markdown(f"**Translated sentence:** {sentence or 'No signs added yet.'}")
    if sentence and st.button("Speak translation"):
        try:
            from services.text_to_speech import text_to_speech_bytes
            audio_bytes, error = text_to_speech_bytes(sentence)
            if audio_bytes:
                st.session_state.isl_audio_to_play = audio_bytes
            else:
                st.warning(error or "Speech output unavailable.")
        except Exception as error:
            st.warning(f"Speech output unavailable: {error}")
    if st.session_state.isl_audio_to_play:
        st.audio(st.session_state.isl_audio_to_play, format="audio/mp3")


def _render_text_to_sign(services, vocabulary):
    st.subheader("Text or Speech to Sign")
    input_mode = st.radio("Input method", ["Text", "Speech"], horizontal=True, key="isl_input_mode")

    if input_mode == "Text":
        st.session_state.isl_input_text = st.text_input(
            "English text",
            value=st.session_state.isl_input_text,
            key="isl_text_widget",
            placeholder="hello, thank you, I need help",
        )
    else:
        audio_file = st.audio_input("Record speech")
        if audio_file is not None:
            try:
                from services.speech_to_text import speech_to_text_from_audio
                transcript, error = speech_to_text_from_audio(audio_file.getvalue())
                if transcript:
                    st.session_state.isl_voice_transcript = transcript
                    st.session_state.isl_input_text = transcript
                    st.success(f"Recorded speech: {transcript}")
                else:
                    st.warning(error or "Speech recognition unavailable.")
            except Exception as error:
                st.warning(f"Speech recognition unavailable: {error}")
        st.caption(st.session_state.isl_input_text or "No speech transcript yet.")

    if st.button("Translate to ISL signs", type="primary"):
        source_text = st.session_state.isl_input_text.strip()
        if not source_text:
            st.warning("Enter text or record speech first.")
        else:
            result = services["text_to_gloss"](source_text, config_path=str(VOCAB_PATH))
            st.session_state.isl_gloss_sequence = result["glosses"]
            st.session_state.isl_unsupported_words = result["unsupported"]

    glosses = st.session_state.isl_gloss_sequence
    unsupported = st.session_state.isl_unsupported_words
    if glosses:
        st.markdown(f"**ISL gloss sequence:** {' -> '.join(glosses)}")
        columns = st.columns(len(glosses))
        for column, gloss in zip(columns, glosses):
            with column:
                st.markdown(f"**{gloss}**")
                video_path = _get_video_path(gloss, vocabulary)
                if video_path:
                    st.video(str(video_path))
                else:
                    st.warning(f"Sign video unavailable for {gloss}.")
    elif st.session_state.isl_input_text:
        st.info("No supported ISL signs were found.")
    if unsupported:
        st.warning(f"Unsupported words skipped: {', '.join(unsupported)}")


def render_isl_translator():
    _init_isl_state()
    st.header("Two-Way Indian Sign Language Translator")

    services, service_error = _load_isl_services()
    if services is None:
        st.error(f"ISL services unavailable: {service_error}")
        return

    vocabulary = services["load_sign_vocabulary"](str(VOCAB_PATH))
    if not vocabulary:
        st.warning("ISL vocabulary unavailable.")

    detector, detector_error = _load_detector()
    predictor, predictor_error = _load_predictor()

    status_col1, status_col2 = st.columns(2)
    with status_col1:
        st.success("Hand detector ready") if detector else st.warning(f"ISL hand detector unavailable: {detector_error}")
    with status_col2:
        if predictor and predictor.is_model_loaded():
            st.success("ISL model ready")
        else:
            st.warning(f"ISL model unavailable: {predictor_error or 'model file or label mapping is missing'}")

    sign_tab, text_tab = st.tabs(["Sign -> Text", "Text/Speech -> Sign"])
    with sign_tab:
        _render_sign_to_text(services, vocabulary, detector, predictor)
    with text_tab:
        _render_text_to_sign(services, vocabulary)
