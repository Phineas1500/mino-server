from modal import Image, Mount, App, Secret
import whisper
from pathlib import Path
import json
import traceback
import soundfile as sf
import sys
import os
import openai
import time
import requests

app = App("whisper-transcription")

# Update image to include soundfile and its dependencies
image = (
    Image.debian_slim()
    .apt_install(
        "ffmpeg",
        "libsndfile1",
        "git",
        "python3-pip"
    )
    .pip_install(
        "git+https://github.com/openai/whisper.git",
        "ffmpeg-python",
        "soundfile",
        "openai",
        "numpy",
        "torch",
        "tqdm"
    )
)

def log_to_modal(message, level="INFO"):
    """Helper function to format logs for Modal"""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] [{level}] {message}")

@app.function(
    gpu="T4",
    image=image,
    timeout=1800
)
def transcribe_audio(audio_data: bytes, filename: str):
    try:
        print("Starting Modal transcription function")
        print(f"Received filename: {filename}")
        print(f"Received audio data size: {len(audio_data)} bytes")
        
        # Save the uploaded audio content temporarily
        temp_path = Path(f"/tmp/{filename}")
        temp_path.write_bytes(audio_data)
        print(f"Saved audio to: {temp_path}")
        print(f"File exists: {temp_path.exists()}")
        print(f"File size: {temp_path.stat().st_size} bytes")
        
        # Add FFmpeg check for audio file
        import subprocess
        try:
            ffprobe_cmd = ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "default=noprint_wrappers=1", str(temp_path)]
            probe_output = subprocess.check_output(ffprobe_cmd, stderr=subprocess.STDOUT).decode()
            print(f"FFprobe output: {probe_output}")
        except subprocess.CalledProcessError as e:
            print(f"FFprobe error: {e.output.decode()}")

        # Validate audio file
        try:
            data, samplerate = sf.read(str(temp_path))
            print(f"Audio file validated: {samplerate}Hz, shape: {data.shape}")
            print(f"Audio duration: {len(data)/samplerate:.2f} seconds")
            print(f"Audio min/max values: {data.min():.2f}/{data.max():.2f}")
        except Exception as e:
            print(f"Audio validation error: {str(e)}")
            raise ValueError(f"Invalid audio file: {str(e)}")
        
        # Load model and transcribe
        print("Loading Whisper model...")
        model = whisper.load_model("base")
        print("Model loaded, starting transcription...")
        
        result = model.transcribe(
            str(temp_path),
            verbose=True,
            language='en',
            task='transcribe'
        )
        
        print(f"Transcription completed. Text length: {len(result['text'])}")
        print(f"Number of segments: {len(result['segments'])}")
        print(f"First few characters of transcript: {result['text'][:100]}")
        
        return {
            "status": "success",
            "transcript": result["text"],
            "segments": result["segments"]
        }
    except Exception as e:
        print(f"Error in Modal function: {str(e)}")
        print(f"Full traceback: {traceback.format_exc()}")
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@app.function(
    secrets=[Secret.from_name("openai-secret")],
    timeout=1800,
    image=image
)
def process_transcript(transcript: str):
    """Process transcript with OpenAI to generate summary and flashcards"""
    try:
        log_to_modal("Starting transcript processing...")
        log_to_modal(f"Transcript length: {len(transcript)} characters")
        
        # Initialize OpenAI client
        client = get_openai_client()
        
        # Generate summary
        summary_prompt = f"""Please analyze this transcript and provide:
        1. A concise summary (2-3 paragraphs)
        2. 3-5 key points
        3. 5 flashcards in Q&A format
        
        Transcript:
        {transcript}
        
        Format the response as JSON with the following structure:
        {{
            "summary": "...",
            "keyPoints": ["point1", "point2", ...],
            "flashcards": [
                {{"question": "...", "answer": "..."}}
            ]
        }}
        """
        
        log_to_modal("Preparing API request to OpenAI...")
        log_to_modal(f"Prompt length: {len(summary_prompt)} characters")
        
        request_start_time = time.time()
        log_to_modal("Sending request to OpenAI API...")
        
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates educational content from video transcripts."},
                    {"role": "user", "content": summary_prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            request_duration = time.time() - request_start_time
            
            log_to_modal(f"OpenAI API response received in {request_duration:.2f} seconds")
            log_to_modal(f"Response model: {response.model}")
            log_to_modal(f"Response usage: {response.usage}")
            log_to_modal(f"First choice finish reason: {response.choices[0].finish_reason}")
            
            result = json.loads(response.choices[0].message.content)
            log_to_modal("Successfully parsed response as JSON")
            log_to_modal(f"Result keys: {list(result.keys())}")
            return result
            
        except openai.APIError as e:
            log_to_modal(f"OpenAI API error: {str(e)}", "ERROR")
            log_to_modal(f"Error type: {type(e).__name__}", "ERROR")
            raise
            
        except json.JSONDecodeError as e:
            log_to_modal("Failed to parse response as JSON", "ERROR")
            log_to_modal(f"Raw response content: {response.choices[0].message.content}", "ERROR")
            raise
            
    except Exception as e:
        error_msg = f"Error in process_transcript: {str(e)}"
        log_to_modal(error_msg, "ERROR")
        log_to_modal(f"Full traceback: {traceback.format_exc()}", "ERROR")
        return {
            "error": error_msg,
            "traceback": traceback.format_exc()
        }

@app.function(
    secrets=[Secret.from_name("openai-secret")],
    timeout=1800,
    image=image
)
def test_openai_integration():
    """Test function to verify OpenAI API integration"""
    log_to_modal("Starting OpenAI integration test...")
    
    test_transcript = """
    This is a test transcript. In this video, we discuss the basics of machine learning.
    The key topics covered include supervised learning, unsupervised learning, and reinforcement learning.
    We also talk about common algorithms like linear regression and neural networks.
    """
    
    try:
        client = get_openai_client()
        
        log_to_modal("Making test API call to OpenAI...")
        
        summary_prompt = f"""Please analyze this transcript and provide:
        1. A concise summary (2-3 paragraphs)
        2. 3-5 key points
        3. 5 flashcards in Q&A format
        
        Transcript:
        {test_transcript}
        
        Format the response as JSON with the following structure:
        {{
            "summary": "...",
            "keyPoints": ["point1", "point2", ...],
            "flashcards": [
                {{"question": "...", "answer": "..."}}
            ]
        }}
        """
        
        request_start_time = time.time()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that creates educational content from video transcripts."},
                {"role": "user", "content": summary_prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        request_duration = time.time() - request_start_time
        
        log_to_modal(f"OpenAI API response received in {request_duration:.2f} seconds")
        log_to_modal(f"Response model: {response.model}")
        log_to_modal(f"Response usage: {response.usage}")
        log_to_modal(f"First choice finish reason: {response.choices[0].finish_reason}")
        
        result = json.loads(response.choices[0].message.content)
        log_to_modal("Successfully parsed response as JSON")
        log_to_modal(f"Result keys: {list(result.keys())}")
        return result
        
    except Exception as e:
        error_msg = f"Test failed: {str(e)}"
        log_to_modal(error_msg, "ERROR")
        log_to_modal(f"Full traceback: {traceback.format_exc()}", "ERROR")
        return {"error": error_msg}

def get_openai_client():
    """Configure OpenAI client"""
    log_to_modal("Initializing OpenAI client...")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        log_to_modal("No API key found in environment!", "ERROR")
        raise ValueError("OPENAI_API_KEY not found in environment")
    
    log_to_modal(f"API Key found (starts with: {api_key[:8]}...)")
    
    # Initialize standard OpenAI client
    client = openai.OpenAI(
        api_key=api_key
    )
    
    # Test the API key with a simple request
    try:
        models = client.models.list()
        log_to_modal(f"API test successful - available models: {[model.id for model in models.data[:3]]}")
    except Exception as e:
        log_to_modal(f"API test failed: {str(e)}", "ERROR")
        raise
    
    return client

if __name__ == "__main__":
    app.serve() 