import streamlit as st
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import numpy as np
from pydub import AudioSegment
import io

# Set page config
st.set_page_config(
    page_title="Speech Recognition App",
    page_icon="🎤",
    layout="wide"
)

@st.cache_resource
def load_whisper_model():
    """Load and cache the Whisper model from Hugging Face"""
    try:
        model_id = "openai/whisper-base"
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
        
        # Move to CPU (cloud deployment friendly)
        device = "cpu"
        model.to(device)
        
        return processor, model, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def convert_audio_for_processing(audio_bytes):
    """Convert audio bytes to format suitable for Whisper"""
    try:
        # Load audio using pydub
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        
        # Convert to mono and 16kHz sample rate
        audio = audio.set_channels(1).set_frame_rate(16000)
        
        # Convert to numpy array
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        samples = samples / (2**15)  # Normalize to [-1, 1]
        
        return samples
    except Exception as e:
        st.error(f"Error converting audio: {str(e)}")
        return None

def transcribe_audio(processor, model, device, audio_samples):
    """Transcribe audio using Whisper model"""
    try:
        # Process audio
        inputs = processor(audio_samples, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate transcription
        with torch.no_grad():
            predicted_ids = model.generate(**inputs)
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        
        return transcription[0] if transcription else ""
    except Exception as e:
        st.error(f"Error during transcription: {str(e)}")
        return None

def main():
    st.title("🎤 Speech Recognition App")
    st.markdown("Upload an audio file or record your voice to get transcription!")
    
    # Load model
    with st.spinner("Loading Whisper model..."):
        processor, model, device = load_whisper_model()
    
    if processor is None or model is None:
        st.error("❌ Failed to load the speech recognition model. Please check the logs.")
        st.stop()
    
    st.success("✅ Whisper model loaded successfully!")
    
    # Model info
    st.sidebar.header("Model Information")
    st.sidebar.info("**Model**: OpenAI Whisper Base\n**Size**: ~290MB\n**Language**: Multi-language support")
    
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
                    audio_samples = convert_audio_for_processing(audio_bytes)
                    
                    if audio_samples is not None:
                        # Transcribe
                        transcription = transcribe_audio(processor, model, device, audio_samples)
                        
                        if transcription:
                            st.success("Transcription completed!")
                            
                            # Display results
                            st.subheader("📝 Transcription:")
                            st.write(transcription)
                            
                            # Copy-friendly format
                            st.code(transcription, language=None)
                        else:
                            st.error("Failed to transcribe audio. Please try again.")
    
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
                    
                    # Convert audio for processing
                    audio_samples = convert_audio_for_processing(audio_bytes)
                    
                    if audio_samples is not None:
                        # Transcribe
                        transcription = transcribe_audio(processor, model, device, audio_samples)
                        
                        if transcription:
                            st.success("Transcription completed!")
                            
                            # Display results
                            st.subheader("📝 Transcription:")
                            st.write(transcription)
                            
                            # Copy-friendly format
                            st.code(transcription, language=None)
                        else:
                            st.error("Failed to transcribe audio. Please try again.")
    
    # Footer
    st.markdown("---")
    st.markdown("Built with Streamlit and Hugging Face Transformers")
    
    # Usage tips
    with st.expander("💡 Usage Tips"):
        st.markdown("""
        - **Best quality**: Use WAV files with 16kHz sample rate
        - **File size**: Keep files under 25MB for faster processing
        - **Recording**: Speak clearly and avoid background noise
        - **Languages**: Supports multiple languages automatically
        """)

if __name__ == "__main__":
    main()