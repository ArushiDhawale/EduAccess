
# 📚 EduAccess

**EduAccess** is an AI-powered inclusive classroom learning assistant designed to make education accessible for every student. By combining real-time speech-to-text, video/audio captioning, AI summarization, concept simplification, translation, and Indian Sign Language visualization, EduAccess ensures that learning has no barriers.

---

## 🚀 Key Features

*   **🎤 Audio/Video Transcription**: Upload lecture videos (`.mp4`, `.mov`, `.mkv`) or audio (`.mp3`, `.wav`, `.m4a`) to automatically generate highly accurate Whisper-based transcriptions and WebVTT subtitle files. Play videos inside the app with synchronized subtitles.
*   **🎙️ Live Lecture Speech-to-Text**: Stream real-time captions directly in the browser using the Web Speech API—ideal for capturing live classroom lectures.
*   **📝 AI Summaries**: Convert full lecture transcripts into structured study notes (Executive Summary, Key Takeaways, Terminology, Revision Questions) using Gemini.
*   **📖 Simplify Concepts**: Paste complex technical content or dense paragraphs to instantly receive an easy-to-read, structured breakdown with simple language and analogies.
*   **🌐 Multi-Language Translation**: Translate lecture content into regional languages like Hindi and Marathi.
*   **🤟 Indian Sign Language (ISL)**: Type words to instantly visualize spelling and concepts using Indian Sign Language images.

---

## 📂 Project Structure

```
EduAccess/
├── app.py                # Main Streamlit application entry point
├── templates/
│   └── index.html        # Live Speech-to-Text HTML/JS page (Web Speech API)
├── utils/
│   ├── __init__.py       # Package marker
│   └── summarizer.py     # AI helper functions (summary, simplify content)
├── isl/                  # [Optional] Place ISL sign PNGs here (A.png, B.png, ...)
└── README.md             # Project documentation
```

---

## 🛠️ Setup & Installation

### 1. Prerequisites
Ensure you have Python 3.9+ installed. You may also need to install `ffmpeg` on your system for audio extraction features to function (via `moviepy`).

*   **macOS**: `brew install ffmpeg`
*   **Windows**: Download from the official website and add it to your System PATH.
*   **Linux**: `sudo apt install ffmpeg`

### 2. Clone the Repository
```bash
git clone https://github.com/ArushiDhawale/EduAccess.git
cd EduAccess
```

### 3. Install Dependencies
Install all required Python libraries:
```bash
pip install streamlit moviepy faster-whisper google-genai
```

### 4. Get a Gemini API Key
To use the AI Summarizer and Simplifier features:
1. Obtain an API Key from [Google AI Studio](https://aistudio.google.com/).
2. You can enter the API Key directly in the app's sidebar, or export it as an environment variable in your terminal:
    ```bash
    export GEMINI_API_KEY="your_api_key_here"
    ```

---

## 🏃 Running the Application

Launch the Streamlit dashboard from the project root:

```bash
streamlit run app.py
```

Open the local address (typically `http://localhost:8501`) in your web browser. Note that Google Chrome is highly recommended for the **Live Lecture Speech-to-Text** tab to function correctly, as it relies on Google's Web Speech API implementation.
