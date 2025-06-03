import os
import json
import tempfile
import time
import logging
from io import BytesIO
import sounddevice as sd
from scipy.io.wavfile import write
from dotenv import load_dotenv
from groq import Groq
from pydub import AudioSegment
from pydub.playback import play

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class GroqSTT:
    def __init__(self, model="whisper-large-v3-turbo", env_key="sttANDtts"):
        load_dotenv()
        api_key = os.getenv(env_key)
        if not api_key:
            raise ValueError(f"API key not found in .env file under '{env_key}'")
        self.client = Groq(api_key=api_key)
        self.model = model
        logger.info(f"Initialized STT with model: {model}")

    def transcribe_file(self, file_path: str, prompt: str = None):
        start_time = time.time()
        with open(file_path, "rb") as file:
            transcription = self.client.audio.transcriptions.create(
                file=file,
                model=self.model,
                prompt=prompt or "Live microphone transcription",
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"],
                language="en",
                temperature=0.0
            )
        logger.info(f"Transcription completed in {time.time() - start_time:.2f} seconds")
        return transcription

class GroqTTS:
    def __init__(self, voice="Thunder-PlayAI", model="playai-tts", env_key="sttANDtts"):
        load_dotenv()
        api_key = os.getenv(env_key)
        if not api_key:
            raise ValueError(f"API key not found in .env file under '{env_key}'")
        
        self.client = Groq(api_key=api_key)
        self.voice = voice
        self.model = model
        logger.info(f"Initialized TTS with voice: {voice} and model: {model}")

    def speak(self, text: str):
        start_time = time.time()
        logger.info(f"Generating speech for: '{text}'")
        response = self.client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            response_format="wav",
            input=text,
        )
        audio = AudioSegment.from_file(BytesIO(response.read()), format="wav")
        logger.info(f"Speech generation completed in {time.time() - start_time:.2f} seconds")
        logger.info("Playing audio...")
        play(audio)

class GroqChat:
    def __init__(self, model="llama-3.3-70b-versatile", env_key="sttANDtts"):
        load_dotenv()
        api_key = os.getenv(env_key)
        if not api_key:
            raise ValueError(f"API key not found in .env file under '{env_key}'")
        self.client = Groq(api_key=api_key)
        self.model = model
        logger.info(f"Initialized Chat with model: {model}")

    def ask(self, query: str) -> str:
        start_time = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Reply concisely."},
                {"role": "user", "content": query},
            ],
            temperature=0.5,
            max_tokens=200
        )
        logger.info(f"Chat response generated in {time.time() - start_time:.2f} seconds")
        return response.choices[0].message.content.strip()

class SpeechToSpeech:
    def __init__(self, 
                 stt_model="whisper-large-v3-turbo",
                 tts_voice="Jennifer-PlayAI",
                 tts_model="playai-tts",
                 chat_model="llama-3.3-70b-versatile",
                 env_key="sttANDtts"):
        self.stt = GroqSTT(model=stt_model, env_key=env_key)
        self.tts = GroqTTS(voice=tts_voice, model=tts_model, env_key=env_key)
        self.chat = GroqChat(model=chat_model, env_key=env_key)
        logger.info("SpeechToSpeech system initialized")

    def record_audio(self, duration=5, fs=44100):
        """Record audio from microphone and save to temporary file."""
        start_time = time.time()
        logger.info(f"Recording {duration} seconds of audio... Speak now!")
        
        # Optimize recording settings
        recording = sd.rec(
            int(duration * fs),
            samplerate=fs,
            channels=1,
            dtype='int16',
            blocking=True
        )
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        write(temp_file.name, fs, recording)
        logger.info(f"Recording completed in {time.time() - start_time:.2f} seconds")
        return temp_file.name

    def process_conversation(self, duration=5):
        """Record audio, transcribe it, get AI response, and speak it back."""
        total_start_time = time.time()
        
        # Record audio
        audio_file = self.record_audio(duration=duration)
        
        try:
            # Transcribe audio
            transcription = self.stt.transcribe_file(audio_file)
            user_text = transcription.text if hasattr(transcription, 'text') else str(transcription)
            logger.info(f"Transcription: {user_text}")
            
            # Get AI response
            response = self.chat.ask(user_text)
            logger.info(f"AI response: {response}")
            
            # Speak response
            self.tts.speak(response)
            
            logger.info(f"Total conversation cycle completed in {time.time() - total_start_time:.2f} seconds")
            
        finally:
            # Clean up temporary file
            os.unlink(audio_file)

def main():
    logger.info("Starting SpeechToSpeech application")
    sts = SpeechToSpeech()
    
    while True:
        try:
            # Process one conversation cycle
            sts.process_conversation(duration=5)
            
            # Ask if user wants to continue
            continue_chat = input("\nWould you like to continue? (y/n): ").lower()
            if continue_chat != 'y':
                logger.info("Application terminated by user")
                break
                
        except KeyboardInterrupt:
            logger.info("Application terminated by keyboard interrupt")
            break
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}", exc_info=True)
            break

if __name__ == "__main__":
    main()
