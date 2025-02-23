import moviepy.editor as mp
from pathlib import Path
import json
import sys
import os
import traceback
from modal import Function

def process_video(input_path, output_path):
    """
    Process video and send to Modal for transcription, content analysis, and segment analysis
    """
    try:
        sys.stderr.write(f"Starting process_video with input: {input_path}\n")
        input_path = Path(input_path)
        
        # Read the video file
        sys.stderr.write("Reading video file for Modal...\n")
        with open(input_path, 'rb') as video_file:
            video_data = video_file.read()
        sys.stderr.write(f"Read {len(video_data)} bytes of video data\n")
        
        sys.stderr.write("Connecting to Modal service...\n")
        modal_fn = Function.lookup("whisper-transcription", "process_video")
        
        # Just send the video data and filename without path
        filename = input_path.name
        sys.stderr.write(f"Sending to Modal for processing with filename: {filename}\n")
        result = modal_fn.remote(video_data, filename)
        sys.stderr.write("Received result from Modal\n")

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
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    process_video(input_path, output_path)