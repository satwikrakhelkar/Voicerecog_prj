import streamlit as st
import speech_recognition as sr
import pyaudio
import wave
import tempfile
import os

# Configure page
st.set_page_config(
    page_title="Speech Recognition App",
    page_icon="🎤",
    layout="wide"
)

# Initialize session state
if 'recognition_result' not in st.session_state:
    st.session_state.recognition_result = ""
if 'is_listening' not in st.session_state:
    st.session_state.is_listening = False

def record_audio(duration=5, sample_rate=44100, chunk=1024):
    """Record audio from microphone"""
    try:
        # Initialize PyAudio
        p = pyaudio.PyAudio()
        
        # Open stream
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=chunk
        )
        
        st.info(f"🎤 Recording for {duration} seconds...")
        frames = []
        
        # Record audio
        for i in range(0, int(sample_rate / chunk * duration)):
            data = stream.read(chunk)
            frames.append(data)
        
        # Stop and close stream
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            wf = wave.open(tmp_file.name, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(sample_rate)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            return tmp_file.name
            
    except Exception as e:
        st.error(f"Error recording audio: {str(e)}")
        return None

def recognize_speech_from_file(audio_file_path, language='en-US'):
    """Convert speech to text from audio file"""
    try:
        r = sr.Recognizer()
        
        with sr.AudioFile(audio_file_path) as source:
            # Adjust for ambient noise
            r.adjust_for_ambient_noise(source)
            # Record the audio
            audio = r.record(source)
        
        # Perform speech recognition
        text = r.recognize_google(audio, language=language)
        return text
        
    except sr.UnknownValueError:
        return "Could not understand the audio"
    except sr.RequestError as e:
        return f"Error with the speech recognition service: {e}"
    except Exception as e:
        return f"Error: {str(e)}"

def recognize_speech_from_microphone(language='en-US'):
    """Direct microphone to speech recognition"""
    try:
        r = sr.Recognizer()
        
        with sr.Microphone() as source:
            st.info("🎤 Adjusting for ambient noise... Please wait.")
            r.adjust_for_ambient_noise(source, duration=1)
            st.info("🎤 Listening... Speak now!")
            
            # Listen for audio
            audio = r.listen(source, timeout=10, phrase_time_limit=10)
            
        st.info("🔄 Processing audio...")
        
        # Perform speech recognition
        text = r.recognize_google(audio, language=language)
        return text
        
    except sr.WaitTimeoutError:
        return "Listening timeout - no speech detected"
    except sr.UnknownValueError:
        return "Could not understand the audio"
    except sr.RequestError as e:
        return f"Error with the speech recognition service: {e}"
    except Exception as e:
        return f"Error: {str(e)}"

# Main UI
st.title("🎤 Speech Recognition App")
st.markdown("Convert your speech to text using Google's Speech Recognition API")

# Sidebar for settings
st.sidebar.header("Settings")
language_options = {
    'English (US)': 'en-US',
    'English (UK)': 'en-GB',
    'Spanish': 'es-ES',
    'French': 'fr-FR',
    'German': 'de-DE',
    'Italian': 'it-IT',
    'Portuguese': 'pt-PT',
    'Russian': 'ru-RU',
    'Japanese': 'ja-JP',
    'Korean': 'ko-KR',
    'Chinese (Mandarin)': 'zh-CN'
}

selected_language = st.sidebar.selectbox(
    "Select Language",
    options=list(language_options.keys()),
    index=0
)

language_code = language_options[selected_language]

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎙️ Live Microphone Recognition")
    
    if st.button("Start Live Recording", type="primary"):
        with st.spinner("Listening..."):
            result = recognize_speech_from_microphone(language_code)
            st.session_state.recognition_result = result
    
    st.subheader("📁 Upload Audio File")
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'flac', 'm4a'],
        help="Upload an audio file to convert speech to text"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file.name.split(".")[-1]}') as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_file_path = tmp_file.name
        
        st.audio(uploaded_file, format=f'audio/{uploaded_file.name.split(".")[-1]}')
        
        if st.button("Convert Audio to Text"):
            with st.spinner("Processing audio file..."):
                # Convert to WAV if necessary
                if not uploaded_file.name.lower().endswith('.wav'):
                    try:
                        from pydub import AudioSegment
                        audio = AudioSegment.from_file(temp_file_path)
                        wav_path = temp_file_path.replace(temp_file_path.split('.')[-1], 'wav')
                        audio.export(wav_path, format="wav")
                        result = recognize_speech_from_file(wav_path, language_code)
                        os.unlink(wav_path)  # Clean up
                    except ImportError:
                        st.error("Please install pydub for non-WAV file support: pip install pydub")
                        result = "Error: pydub not installed"
                    except Exception as e:
                        result = f"Error converting audio: {str(e)}"
                else:
                    result = recognize_speech_from_file(temp_file_path, language_code)
                
                st.session_state.recognition_result = result
        
        # Clean up temporary file
        try:
            os.unlink(temp_file_path)
        except Exception:
            pass

with col2:
    st.subheader("📝 Recognition Results")
    
    if st.session_state.recognition_result:
        st.success("Speech Recognition Result:")
        st.text_area(
            "Transcribed Text:",
            value=st.session_state.recognition_result,
            height=200,
            key="result_text"
        )
        
        # Download result as text file
        if st.download_button(
            label="📥 Download as Text File",
            data=st.session_state.recognition_result,
            file_name="speech_recognition_result.txt",
            mime="text/plain"
        ):
            st.success("File downloaded successfully!")
        
        # Clear results
        if st.button("🗑️ Clear Results"):
            st.session_state.recognition_result = ""
            st.rerun()
    else:
        st.info("👆 Use the controls on the left to start speech recognition")

# Instructions
st.markdown("---")
st.subheader("📋 Instructions")
st.markdown("""
1. **Live Recording**: Click "Start Live Recording" and speak into your microphone
2. **File Upload**: Upload an audio file (WAV, MP3, FLAC, M4A) and click "Convert Audio to Text"
3. **Language**: Select your preferred language from the sidebar
4. **Results**: View transcribed text in the results panel and download if needed

**Requirements:**
- Microphone access for live recording
- Internet connection for Google Speech Recognition API
- Supported audio formats: WAV, MP3, FLAC, M4A
""")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and Google Speech Recognition API")