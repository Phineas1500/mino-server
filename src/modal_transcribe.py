from modal import Image, Secret, App
import whisper
import json
import os
import openai
import time
from pathlib import Path
from moviepy.editor import VideoFileClip

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
        "python3-dev",
        "libsndfile1",
        "libglib2.0-0",
        "libgl1-mesa-glx",
        "libsm6",
        "libxext6"
    )
    .pip_install(
        "torch",
        "numpy",
        "ffmpeg-python",
        "openai",
        "tqdm",
        "decorator==4.4.2",
        "imageio-ffmpeg",
        "moviepy==1.0.3",
        "soundfile"
    )
    .run_commands(
        "pip install git+https://github.com/openai/whisper.git"
    )
)

def cut_video_segments(video_path, segments, min_segment_duration=1.0, max_pause_duration=0.5):
    """Cut video based on transcript segments, removing long pauses."""
    try:
        from moviepy.editor import VideoFileClip, concatenate_videoclips
        
        log(f"Opening video file for cutting: {video_path}")
        # Explicitly set audio=True to ensure audio is loaded
        video = VideoFileClip(str(video_path), audio=True)
        
        if not video.fps:
            raise ValueError("Could not determine video FPS")
            
        clips_to_keep = []
        
        for i, segment in enumerate(segments):
            segment_duration = segment["end"] - segment["start"]
            
            if segment_duration < min_segment_duration:
                continue
                
            if i > 0:
                pause_duration = segment["start"] - segments[i-1]["end"]
                if pause_duration > max_pause_duration:
                    segment["start"] = segments[i-1]["end"] + max_pause_duration
            
            log(f"Processing segment {i+1}/{len(segments)}: {segment['start']:.2f}s to {segment['end']:.2f}s")
            clip = video.subclip(segment["start"], segment["end"])
            clips_to_keep.append(clip)
        
        if not clips_to_keep:
            log("No valid segments found - using original video", "WARNING")
            output_path = str(video_path)
            video.close()
            return output_path
            
        log(f"Concatenating {len(clips_to_keep)} clips...")
        final_video = concatenate_videoclips(clips_to_keep)
        
        output_path = str(Path(video_path).with_stem(f"{Path(video_path).stem}_shortened"))
        log(f"Writing shortened video to: {output_path}")
        
        final_video.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            verbose=False,
            logger=None
        )
        
        # Clean up
        video.close()
        final_video.close()
        for clip in clips_to_keep:
            clip.close()
            
        return output_path
        
    except Exception as e:
        log(f"Error in cut_video_segments: {str(e)}", "ERROR")
        # If there's an error, return the original video path
        return str(video_path)

@app.function(
    gpu="T4",
    image=image,
    timeout=1800,
    secrets=[Secret.from_name("openai-secret")]
)
async def process_video(video_data: bytes, filename: str):
    """Main function that handles video processing, transcription, and summarization"""
    temp_files = []  # Keep track of temporary files
    try:
        # Create a safe filename without spaces
        safe_filename = filename.replace(" ", "_")
        
        # Save video data temporarily
        temp_video_path = Path("/tmp") / safe_filename
        temp_files.append(temp_video_path)  # Track for cleanup
        temp_video_path.write_bytes(video_data)
        log(f"Saved video file: {temp_video_path} ({len(video_data)} bytes)")

        # Create directories if they don't exist
        temp_video_path.parent.mkdir(parents=True, exist_ok=True)

        # Extract audio for transcription
        log("Extracting audio for transcription...")
        video = VideoFileClip(str(temp_video_path))
        temp_audio_path = temp_video_path.with_suffix('.wav')
        temp_files.append(temp_audio_path)  # Track for cleanup
        
        video.audio.write_audiofile(
            str(temp_audio_path),
            codec='pcm_s16le',
            fps=16000,
            ffmpeg_params=["-ac", "1"],
            verbose=False,
            logger=None
        )
        video.close()

        # Transcribe with Whisper
        log("Loading Whisper model...")
        model = whisper.load_model("base")
        
        log("Starting transcription...")
        result = model.transcribe(
            str(temp_audio_path),
            language='en',
            verbose=False
        )
        transcript = result["text"]
        segments = result["segments"]
        log(f"Transcription complete: {len(transcript)} characters")

        # Clean up audio file
        if temp_audio_path.exists():
            temp_audio_path.unlink()

        # If no transcript, return original video
        if not transcript.strip():
            log("No transcript found - returning original video", "WARNING")
            return {
                "status": "success",
                "summary": "Video contains no spoken content.",
                "keyPoints": ["No key points available."] * 5,
                "flashcards": [{"question": "No content available.", "answer": "No content available."}] * 5,
                "transcript": "",
                "segments": [],
                "shortened_video": video_data
            }

        # Process with OpenAI
        log("Processing with OpenAI...")
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
        log("Successfully processed with OpenAI")

        # Cut video - now handles empty segments gracefully
        log("Starting video cutting process...")
        shortened_path = cut_video_segments(
            temp_video_path,
            segments,
            min_segment_duration=1.0,
            max_pause_duration=0.5
        )
        
        # Read the video data
        with open(shortened_path, 'rb') as f:
            shortened_video_data = f.read()

        # Clean up files
        if temp_video_path.exists():
            temp_video_path.unlink()
        if shortened_path != str(temp_video_path) and Path(shortened_path).exists():
            Path(shortened_path).unlink()

        # Return the processed results
        return {
            "status": "success",
            "summary": summary_data["summary"],
            "keyPoints": summary_data["keyPoints"][:5],
            "flashcards": summary_data["flashcards"][:5],
            "transcript": transcript,
            "segments": [
                {
                    "start": s["start"],
                    "end": s["end"],
                    "text": s["text"]
                }
                for s in segments
            ],
            "shortened_video": shortened_video_data if "shortened_video_data" in locals() else video_data
        }

    except Exception as e:
        log(f"Error in process_video: {str(e)}", "ERROR")
        # Return the original video in case of error
        return {
            "status": "success",
            "summary": "Error processing video.",
            "keyPoints": ["Error processing video."] * 5,
            "flashcards": [{"question": "Error processing video.", "answer": "Error processing video."}] * 5,
            "transcript": "",
            "segments": [],
            "shortened_video": video_data
        }
    finally:
        # Clean up all temporary files
        for temp_file in temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except Exception as e:
                log(f"Error cleaning up {temp_file}: {str(e)}", "WARNING")


if __name__ == "__main__":
    app.serve() 