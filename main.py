import streamlit as st
import whisper
import tempfile
import os
from pydub import AudioSegment
import io
# ...existing code...

# Set page config
st.set_page_config(
    page_title="Speech Recognition App",
    page_icon="🎤",
    layout="wide"
)

@st.cache_resource
def load_whisper_model(model_size):
    """Load and cache the Whisper model"""
    return whisper.load_model(model_size)

def convert_audio_to_wav(audio_bytes):
    """Convert audio bytes to WAV format"""
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        # Convert to mono and set sample rate to 16kHz (optimal for Whisper)
        audio = audio.set_channels(1).set_frame_rate(16000)
        
        # Export to WAV
        wav_buffer = io.BytesIO()
        audio.export(wav_buffer, format="wav")
        wav_buffer.seek(0)
        return wav_buffer.getvalue()
    except Exception as e:
        st.error(f"Error converting audio: {str(e)}")
        return None

def transcribe_audio(model, audio_data):
    """Transcribe audio using Whisper model"""
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_data)
            tmp_file_path = tmp_file.name
        
        # Transcribe
        result = model.transcribe(tmp_file_path)
        
        # Clean up
        os.unlink(tmp_file_path)
        
        return result
    except Exception as e:
        st.error(f"Error during transcription: {str(e)}")
        return None

def main():
    st.title("🎤 Speech Recognition App")
    st.markdown("Upload an audio file or record your voice to get transcription!")
    
    # Sidebar for model selection
    st.sidebar.header("Model Settings")
    model_size = st.sidebar.selectbox(
        "Select Whisper Model Size:",
        ["tiny", "base", "small", "medium", "large"],
        index=1,
        help="Larger models are more accurate but slower"
    )
    
    # Model size information
    model_info = {
        "tiny": "39 MB - Fastest, least accurate",
        "base": "74 MB - Good balance",
        "small": "244 MB - Better accuracy",
        "medium": "769 MB - High accuracy",
        "large": "1550 MB - Highest accuracy"
    }
    st.sidebar.info(f"**{model_size.title()} Model**: {model_info[model_size]}")
    
    # Load model
    with st.spinner(f"Loading {model_size} model..."):
        model = load_whisper_model(model_size)
    
    st.success(f"✅ {model_size.title()} model loaded successfully!")
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["🎙️ Record Audio", "📁 Upload Audio File"])
    
    with tab1:
        st.header("Record Audio")
        
        # Audio input
        audio_bytes = st.audio_input("Click to record your voice:")
        
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            
            if st.button("🔤 Transcribe Recording", type="primary"):
                with st.spinner("Transcribing your recording..."):
                    # Convert audio to proper format
                    wav_audio = convert_audio_to_wav(audio_bytes)
                    
                    if wav_audio:
                        # Transcribe
                        result = transcribe_audio(model, wav_audio)
                        
                        if result:
                            st.success("Transcription completed!")
                            
                            # Display results
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.subheader("📝 Transcription:")
                                st.write(result["text"])
                                
                                # Copy button
                                st.code(result["text"], language=None)
                            
                            with col2:
                                st.subheader("ℹ️ Details:")
                                st.write(f"**Language:** {result.get('language', 'Unknown')}")
                                
                                # Show segments if available
                                if 'segments' in result and result['segments']:
                                    st.write(f"**Segments:** {len(result['segments'])}")
                                    
                                    with st.expander("View Segments"):
                                        for i, segment in enumerate(result['segments']):
                                            st.write(f"**{i+1}.** [{segment['start']:.1f}s - {segment['end']:.1f}s]")
                                            st.write(f"   {segment['text']}")
    
    with tab2:
        st.header("Upload Audio File")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'm4a', 'flac', 'ogg'],
            help="Supported formats: WAV, MP3, M4A, FLAC, OGG"
        )
        
        if uploaded_file is not None:
            st.audio(uploaded_file, format="audio/wav")
            
            # Show file details
            st.write(f"**File name:** {uploaded_file.name}")
            st.write(f"**File size:** {uploaded_file.size / 1024 / 1024:.2f} MB")
            
            if st.button("🔤 Transcribe File", type="primary"):
                with st.spinner("Transcribing your file..."):
                    # Read the uploaded file
                    audio_bytes = uploaded_file.read()
                    
                    # Convert to WAV if needed
                    wav_audio = convert_audio_to_wav(audio_bytes)
                    
                    if wav_audio:
                        # Transcribe
                        result = transcribe_audio(model, wav_audio)
                        
                        if result:
                            st.success("Transcription completed!")
                            
                            # Display results
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.subheader("📝 Transcription:")
                                st.write(result["text"])
                                
                                # Copy button
                                st.code(result["text"], language=None)
                            
                            with col2:
                                st.subheader("ℹ️ Details:")
                                st.write(f"**Language:** {result.get('language', 'Unknown')}")
                                
                                # Show segments if available
                                if 'segments' in result and result['segments']:
                                    st.write(f"**Segments:** {len(result['segments'])}")
                                    
                                    with st.expander("View Segments"):
                                        for i, segment in enumerate(result['segments']):
                                            st.write(f"**{i+1}.** [{segment['start']:.1f}s - {segment['end']:.1f}s]")
                                            st.write(f"   {segment['text']}")
    
    # Footer
    st.markdown("---")
    st.markdown("Built with Streamlit and OpenAI Whisper")

if __name__ == "__main__":
    main()