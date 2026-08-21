import os
import streamlit as st
import tempfile
from moviepy import VideoFileClip
from faster_whisper import WhisperModel
from utils.summarizer import generate_lecture_summary, simplify_content
import streamlit.components.v1 as components

st.set_page_config(
    page_title="EduAcess",
    page_icon="📚",
    layout="wide"
)

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

div.stButton > button {
    width: 100%;
    border-radius: 10px;
}

[data-testid="stSidebar"] {
    width: 280px;
}

</style>
""", unsafe_allow_html=True)

# Whisper Model Loader
@st.cache_resource
def load_whisper_model():
    return WhisperModel("base.en", device="cpu", compute_type="int8")

# Helper function to convert seconds to WebVTT timestamp format (HH:MM:SS.mmm)
def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

# Generate WebVTT content from Whisper segments
def generate_vtt(segments) -> str:
    vtt_content = "WEBVTT\n\n"
    for i, seg in enumerate(segments, start=1):
        start = format_timestamp(seg.start)
        end = format_timestamp(seg.end)
        text = seg.text.strip()
        vtt_content += f"{i}\n{start} --> {end}\n{text}\n\n"
    return vtt_content

# Folder where the per-letter ISL images live (A.png, B.png, ...)
ISL_DIR = os.path.join(os.path.dirname(__file__), "isl")

with st.sidebar:
    st.title("EduAcess")

    page = st.radio(
        "Features",
        [
            "Home",
            "Audio Transcription",
            "Summary Generator",
            "Simplify Content",
            "Translation",
            "Indian Sign Language"
        ]
    )
    
    st.write("---")
    st.header("🔑 API Settings")
    api_key = st.text_input("Gemini API Key", type="password", value=os.environ.get("GEMINI_API_KEY", ""), key="gemini_api_key")

if page == "Home":

    st.markdown("""
    ### Welcome to EduAcess

    Features:
    - 🎤 Audio to Text (File Upload & Live)
    - 📝 Smart Summaries (Powered by Gemini)
    - 📖 Simplified Learning
    - 🌐 Multi-language Support
    - 🤟 Indian Sign Language
    """)

elif page == "Audio Transcription":

    st.header("🎤 Audio Transcription")

    tab_upload, tab_live = st.tabs(["📁 Upload Video/Audio", "🎙️ Live Lecture"])

    with tab_upload:
        media_file = st.file_uploader(
            "Upload Video or Audio",
            type=["mp4", "mov", "mkv", "mp3", "wav", "m4a"]
        )

        if media_file:
            file_extension = os.path.splitext(media_file.name)[1].lower()
            is_video = file_extension in [".mp4", ".mov", ".mkv"]

            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_media:
                temp_media.write(media_file.read())
                media_path = temp_media.name

            audio_path = media_path
            if is_video:
                audio_path = media_path.replace(file_extension, ".wav")

            with st.spinner("Processing media and transcribing..."):
                if is_video:
                    video_clip = VideoFileClip(media_path)
                    video_clip.audio.write_audiofile(audio_path, logger=None)
                    video_clip.close()

                model = load_whisper_model()
                segments_gen, _ = model.transcribe(audio_path, beam_size=1)
                segments = list(segments_gen)
                plain_text = " ".join([s.text.strip() for s in segments])

                vtt_text = None
                if is_video:
                    vtt_text = generate_vtt(segments)
                    vtt_path = media_path.replace(file_extension, ".vtt")
                    with open(vtt_path, "w", encoding="utf-8") as f:
                        f.write(vtt_text)

            st.success("Transcription complete!")

            if is_video:
                st.video(media_path, subtitles=vtt_path)
            else:
                st.audio(media_path)
                st.text_area("Transcript Output", value=plain_text, height=200)

            col1, col2 = st.columns(2)
            with col1:
                if is_video and vtt_text:
                    st.download_button(
                        label="📥 Download VTT Subtitle File",
                        data=vtt_text,
                        file_name="lecture_captions.vtt",
                        mime="text/vtt"
                    )
            with col2:
                st.download_button(
                    label="📥 Download Full Text Transcript",
                    data=plain_text,
                    file_name="lecture_transcript.txt",
                    mime="text/plain"
                )

    with tab_live:
        st.subheader("🎙️ Live Speech-to-Text")
        html_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
        if os.path.exists(html_path):
            with open(html_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            components.html(html_content, height=650, scrolling=True)
        else:
            st.error("index.html not found in templates directory.")

elif page == "Summary Generator":

    st.header("📝 Summary Generator")

    if not api_key:
        st.warning("⚠️ Please enter a Gemini API Key in the sidebar to generate a summary.")

    transcript = st.text_area(
        "Transcript",
        height=250
    )

    if st.button("Generate Summary"):
        if not api_key:
            st.error("⚠️ Please enter a Gemini API Key in the sidebar.")
        elif not transcript.strip():
            st.warning("Please enter a transcript to summarize.")
        else:
            with st.spinner("Generating summary..."):
                try:
                    summary = generate_lecture_summary(transcript, api_key)
                    st.markdown("### Summary Output")
                    st.write(summary)
                    st.download_button(
                        label="📥 Download Summary & Notes",
                        data=summary,
                        file_name="lecture_summary.md",
                        mime="text/markdown"
                    )
                except Exception as e:
                    st.error(f"Error generating summary: {str(e)}")

elif page == "Simplify Content":

    st.header("📖 Simplify Concepts")

    if not api_key:
        st.warning("⚠️ Please enter a Gemini API Key in the sidebar to simplify content.")

    content = st.text_area(
        "Enter Content",
        height=250
    )

    if "simplified_result" not in st.session_state:
        st.session_state.simplified_result = ""

    if st.button("Simplify"):
        if not api_key:
            st.error("⚠️ Please enter a Gemini API Key in the sidebar.")
        elif not content.strip():
            st.warning("Please enter some content to simplify.")
        else:
            with st.spinner("Simplifying..."):
                try:
                    st.session_state.simplified_result = simplify_content(content, api_key)
                except Exception as e:
                    st.error(f"Error simplifying content: {str(e)}")

    st.text_area(
        "Simplified Content",
        value=st.session_state.simplified_result,
        height=250
    )

elif page == "Translation":

    st.header("🌐 Translation")

    language = st.selectbox(
        "Select Language",
        ["Hindi", "Marathi"]
    )

    text = st.text_area(
        "Enter Text"
    )

    st.button("Translate")

    st.text_area(
        "Translated Output"
    )

elif page == "Indian Sign Language":

    st.header("🤟 Indian Sign Language")

    word = st.text_input(
        "Enter Word"
    ).strip().upper()

    if st.button("Show ISL"):
        if not word:
            st.warning("Please enter a word first.")
        elif not os.path.isdir(ISL_DIR):
            st.error(
                f"Couldn't find the '{os.path.basename(ISL_DIR)}' folder next to app.py. "
                "Make sure it's uploaded alongside your script."
            )
        else:
            missing = [ch for ch in word if ch.isalpha() and
                       not os.path.isfile(os.path.join(ISL_DIR, f"{ch}.png"))]

            if missing:
                st.error(
                    "No sign image found for: " + ", ".join(sorted(set(missing))) +
                    ". Add the matching PNG files to the isl/ folder."
                )
            else:
                st.write(f"Showing signs for **{word}**:")
                letters = [ch for ch in word if ch.isalpha()]
                cols = st.columns(len(letters)) if letters else []
                for col, ch in zip(cols, letters):
                    with col:
                        st.image(
                            os.path.join(ISL_DIR, f"{ch}.png"),
                            caption=ch,
                            use_container_width=True
                        )
