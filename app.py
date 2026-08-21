import os
import streamlit as st

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

st.title("📚 EduAcess")
st.subheader("Inclusive Learning for Every Student")

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

if page == "Home":

    st.markdown("""
    ### Welcome to EduAcess

    Features:
    - 🎤 Audio to Text
    - 📝 Smart Summaries
    - 📖 Simplified Learning
    - 🌐 Multi-language Support
    - 🤟 Indian Sign Language
    """)

elif page == "Audio Transcription":

    st.header("🎤 Audio Transcription")

    audio_file = st.file_uploader(
        "Upload Audio",
        type=["mp3", "wav", "m4a"]
    )

    if audio_file:
        st.audio(audio_file)
        st.button("Generate Transcript")

elif page == "Summary Generator":

    st.header("📝 Summary Generator")

    transcript = st.text_area(
        "Transcript",
        height=250
    )

    st.button("Generate Summary")

    st.text_area(
        "Summary Output",
        height=250
    )

elif page == "Simplify Content":

    st.header("📖 Simplify Concepts")

    content = st.text_area(
        "Enter Content",
        height=250
    )

    st.button("Simplify")

    st.text_area(
        "Simplified Content",
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
