import moviepy.editor as mp
from pathlib import Path
import json
import sys
import os
import traceback
from modal import Function
from concurrent.futures import ThreadPoolExecutor
import tempfile

def compress_video(input_path):
    """Compress video before sending to Modal"""
    try:
        video = mp.VideoFileClip(str(input_path))
        
        # Calculate target size (compress to 720p if larger)
        target_width = min(1280, video.size[0])
        target_height = min(720, video.size[1])
        
        if target_width < video.size[0] or target_height < video.size[1]:
            video = video.resize(width=target_width, height=target_height)
        
        # Create temporary file for compressed video
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_path = temp_file.name
        
        # Compress with lower bitrate and h264 codec
        video.write_videofile(
            temp_path,
            codec='libx264',
            audio_codec='aac',
            preset='faster',  # Faster encoding, slightly larger file
            bitrate='2000k'  # Adjust bitrate based on your needs
        )
        
        video.close()
        return temp_path
    except Exception as e:
        sys.stderr.write(f"Error in compress_video: {str(e)}\n")
        return str(input_path)  # Return original path if compression fails

def process_video(input_path, output_path):
    """
    Process video and send to Modal for transcription, content analysis, and segment analysis
    """
    try:
        sys.stderr.write(f"Starting process_video with input: {input_path}\n")
        input_path = Path(input_path)
        
        # Compress video in a separate thread
        with ThreadPoolExecutor() as executor:
            compressed_path_future = executor.submit(compress_video, input_path)
            
            # While video is compressing, set up Modal connection
            sys.stderr.write("Connecting to Modal service...\n")
            modal_fn = Function.lookup("whisper-transcription", "process_video")
            
            # Wait for compression to complete
            compressed_path = compressed_path_future.result()
            sys.stderr.write(f"Video compressed: {compressed_path}\n")
        
        # Read the compressed video file
        sys.stderr.write("Reading compressed video file...\n")
        with open(compressed_path, 'rb') as video_file:
            video_data = video_file.read()
        sys.stderr.write(f"Read {len(video_data)} bytes of compressed video data\n")
        
        # Clean up temporary compressed file if it's different from input
        if compressed_path != str(input_path):
            try:
                os.remove(compressed_path)
            except:
                pass
        
        # Send to Modal for processing
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