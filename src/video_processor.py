import sys
import json
import os
import traceback
import tempfile
import subprocess
from pathlib import Path
from modal import Function

def process_video(input_path, output_path):
    """
    Extract audio from video and send only audio to Modal for processing
    """
    try:
        input_path = Path(input_path)
        filename = input_path.name
        
        sys.stderr.write(f"Starting process_video with input: {input_path}\n")
        
        # Create a temporary file for the audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
        
        try:
            # Extract audio using ffmpeg (much faster than moviepy)
            sys.stderr.write(f"Extracting audio to {temp_audio_path}...\n")
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', str(input_path),
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # PCM format
                '-ar', '16000',  # 16kHz sample rate
                '-ac', '1',  # Mono
                '-y',  # Overwrite output file
                temp_audio_path
            ]
            
            # Run ffmpeg
            subprocess.run(ffmpeg_cmd, check=True, stderr=subprocess.PIPE)
            
            # Check the size of the extracted audio
            audio_size = os.path.getsize(temp_audio_path)
            sys.stderr.write(f"Extracted audio: {audio_size} bytes\n")
            
            # Read the audio file (much smaller than the video)
            with open(temp_audio_path, 'rb') as audio_file:
                audio_data = audio_file.read()
                
            # Connect to Modal
            sys.stderr.write("Connecting to Modal service...\n")
            modal_fn = Function.lookup("whisper-transcription", "process_audio")
            
            # Send just the audio data to Modal
            sys.stderr.write(f"Sending audio to Modal for processing from: {filename}\n")
            result = modal_fn.remote(audio_data, filename)
            sys.stderr.write("Received result from Modal\n")
            
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
                sys.stderr.write(f"Cleaned up temporary audio file: {temp_audio_path}\n")
        
        # Write transcript to file
        transcript_path = input_path.with_suffix('.txt')
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(result["transcript"])
            
        output = {
            "status": "success",
            "transcript_file": str(transcript_path),
            "transcript": result["transcript"],
            "summary": result["summary"],
            "keyPoints": result["keyPoints"],
            "flashcards": result["flashcards"],
            "segments": result["segments"],
            "stats": result.get("stats", {
                "total_segments": 0,
                "skippable_segments": 0,
                "total_duration": 0,
                "skippable_duration": 0,
                "skippable_percentage": 0
            })
        }
        
        sys.stderr.write("Processing completed successfully\n")
        print(json.dumps(output, ensure_ascii=False))
        sys.stdout.flush()
        
    except Exception as e:
        sys.stderr.write(f"Error in process_video: {str(e)}\n")
        sys.stderr.write(traceback.format_exc())
        error_output = {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        print(json.dumps(error_output, ensure_ascii=False))
        sys.stdout.flush()
        raise

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.stderr.write("Insufficient arguments\n")
        print(json.dumps({
            "status": "error",
            "error": "Insufficient arguments"
        }))
        sys.exit(1)
    
    # Add a test option for easier debugging
    if sys.argv[1] == "test":
        test_output = {
            "status": "success",
            "transcript": "This is a test transcript.",
            "summary": "Test summary",
            "keyPoints": ["Test point 1", "Test point 2"],
            "flashcards": [{"question": "Test?", "answer": "Answer"}],
            "segments": [{"start": 0, "end": 10, "text": "Test segment"}]
        }
        print(json.dumps(test_output))
        sys.exit(0)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    process_video(input_path, output_path)