from modal import Image, Secret, App
import whisper
import json
import os
import openai
import time
from pathlib import Path

# Create a Modal app
app = App("whisper-transcription")

def log(message: str, level: str = "INFO") -> None:
    """Helper function for consistent logging"""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [{level}] {message}")

# Define the container image with all necessary dependencies
image = (
    Image.debian_slim(python_version="3.11")
    .apt_install(
        "ffmpeg",
        "git",
        "python3-pip",
        "build-essential",
        "python3-dev"
    )
    .pip_install(
        "torch",
        "numpy",
        "ffmpeg-python",
        "openai",
        "tqdm"
    )
    .run_commands(
        "pip install git+https://github.com/openai/whisper.git"
    )
)

@app.function(
    gpu="T4",  # Request T4 GPU for faster transcription
    image=image,
    timeout=1800,
    secrets=[Secret.from_name("openai-secret")]
)
async def process_video(audio_data: bytes, filename: str):
    """Main function that handles both transcription and summarization"""
    try:
        # Step 1: Save audio data temporarily
        temp_path = Path("/tmp") / filename
        temp_path.write_bytes(audio_data)
        log(f"Saved audio file: {temp_path} ({len(audio_data)} bytes)")

        # Step 2: Transcribe with Whisper
        log("Loading Whisper model...")
        model = whisper.load_model("base")
        
        log("Starting transcription...")
        result = model.transcribe(
            str(temp_path),
            language='en',
            verbose=False
        )
        transcript = result["text"]
        segments = result["segments"]
        log(f"Transcription complete: {len(transcript)} characters")

        # Clean up the temporary file
        temp_path.unlink()
        log("Cleaned up temporary audio file")

        # Step 3: Process with OpenAI
        log("Initializing OpenAI client...")
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        prompt = f"""Analyze this lecture transcript and provide a response in this EXACT JSON format:
        {{
            "summary": "A clear 2-3 paragraph summary of the main concepts",
            "keyPoints": [
                "First key point as a complete sentence",
                "Second key point as a complete sentence",
                "Third key point as a complete sentence",
                "Fourth key point as a complete sentence",
                "Fifth key point as a complete sentence"
            ],
            "flashcards": [
                {{
                    "question": "Basic concept question about a key term or idea?",
                    "answer": "A clear and concise answer to the basic concept."
                }},
                {{
                    "question": "Fundamental question about the content?",
                    "answer": "A straightforward explanation of the fundamental concept."
                }},
                {{
                    "question": "Question testing basic understanding?",
                    "answer": "A comprehensive answer demonstrating basic mastery."
                }},
                {{
                    "question": "More challenging question requiring synthesis of multiple concepts?",
                    "answer": "A detailed answer that connects multiple ideas and demonstrates deeper understanding."
                }},
                {{
                    "question": "Advanced inference question about implications or applications?",
                    "answer": "A sophisticated answer that extends beyond the explicit content to explore implications or real-world applications."
                }}
            ]
        }}

        The response MUST:
        1. Include a detailed summary that captures the main ideas
        2. Have exactly 5 key points as complete sentences
        3. Have exactly 5 flashcards with clear questions and answers, where:
           - First 3 cards test basic understanding of explicit content
           - Fourth card requires connecting multiple concepts
           - Fifth card tests ability to make inferences or applications
        4. Be in valid JSON format

        Here is the transcript to analyze:
        {transcript}
        """

        log("Sending request to OpenAI...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that creates educational content from video transcripts. You MUST return responses in valid JSON format with the exact structure specified. For flashcards, create a progression from basic recall to advanced synthesis and application."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=3000,
            response_format={ "type": "json_object" }
        )

        # Parse OpenAI response
        summary_data = json.loads(response.choices[0].message.content)
        log("Successfully parsed OpenAI response")

        # Combine all results
        final_result = {
            "summary": summary_data["summary"],
            "keyPoints": summary_data["keyPoints"][:5],  # Ensure exactly 5 key points
            "flashcards": summary_data["flashcards"][:5],  # Ensure exactly 5 flashcards
            "transcript": transcript,
            "segments": [
                {
                    "start": s["start"],
                    "end": s["end"],
                    "text": s["text"]
                }
                for s in segments
            ]
        }

        # Validate and provide defaults if needed
        if len(final_result["keyPoints"]) < 5:
            final_result["keyPoints"].extend(["Additional key point"] * (5 - len(final_result["keyPoints"])))

        if len(final_result["flashcards"]) < 5:
            default_cards = [
                {"question": "Basic concept question?", "answer": "Basic answer."},
                {"question": "Fundamental question?", "answer": "Fundamental answer."},
                {"question": "Understanding question?", "answer": "Understanding answer."},
                {"question": "Synthesis question?", "answer": "Synthesis answer."},
                {"question": "Application question?", "answer": "Application answer."}
            ]
            final_result["flashcards"].extend(default_cards[len(final_result["flashcards"]):])

        log("Processing completed successfully")
        return final_result

    except Exception as e:
        log(f"Error in process_video: {str(e)}", "ERROR")
        raise

if __name__ == "__main__":
    app.serve() 