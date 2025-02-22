from modal import Image, Mount, App, Secret
import whisper
from pathlib import Path
import json
import traceback
import soundfile as sf
import sys
import os
import openai

app = App("whisper-transcription")

# Update image to include soundfile and its dependencies
image = (
    Image.debian_slim()
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "openai-whisper",
        "ffmpeg-python",
        "soundfile",
        "openai"
    )
)

# Configure OpenAI client for DeepSeek
def get_openai_client():
    client = openai.OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url="https://api.deepseek.com"
    )
    return client

@app.function(gpu="T4", image=image)
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
    timeout=1800
)
def process_transcript(transcript: str):
    """Process transcript with DeepSeek to generate summary and flashcards"""
    try:
        # Initialize DeepSeek client
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
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that creates educational content from video transcripts."},
                {"role": "user", "content": summary_prompt}
            ],
            stream=False
        )
        
        return json.loads(response.choices[0].message.content)
        
    except Exception as e:
        sys.stderr.write(f"Error in process_transcript: {str(e)}\n")
        sys.stderr.write(traceback.format_exc())
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

if __name__ == "__main__":
    app.serve() 