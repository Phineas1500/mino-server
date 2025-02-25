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
        
        # More aggressive resizing - target 480p for faster processing
        target_width = min(854, video.size[0])  # 480p equivalent width
        target_height = min(480, video.size[1])
        
        # Only resize if the video is larger than target
        if target_width < video.size[0] or target_height < video.size[1]:
            # Calculate aspect ratio
            aspect = video.size[0] / video.size[1]
            if aspect > (16/9):
                # Wide video, fit to width
                target_width = 854
                target_height = int(854 / aspect)
            else:
                # Tall video, fit to height
                target_height = 480
                target_width = int(480 * aspect)
            video = video.resize(width=target_width, height=target_height)
        
        # Reduce framerate if it's higher than 24fps
        if video.fps > 24:
            video = video.set_fps(24)
        
        # Create temporary file for compressed video
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_path = temp_file.name
        
        # Compress with more aggressive settings
        video.write_videofile(
            temp_path,
            codec='libx264',
            audio_codec='aac',
            preset='ultrafast',  # Fastest encoding preset
            bitrate='1000k',    # Lower bitrate
            audio_bitrate='96k', # Lower audio quality
            threads=4,          # Use multiple threads
            ffmpeg_params=[
                '-tune', 'fastdecode',  # Optimize for decoding speed
                '-maxrate', '1500k',    # Maximum bitrate
                '-bufsize', '2000k',    # Buffer size
                '-crf', '28',           # Constant Rate Factor (higher = lower quality, 28 is still acceptable)
                '-level', '3.0'         # H.264 level (helps with compatibility)
            ]
        )
        
        video.close()
        
        # Verify the compressed file size
        original_size = os.path.getsize(input_path)
        compressed_size = os.path.getsize(temp_path)
        sys.stderr.write(f"Original size: {original_size/1024/1024:.2f}MB, Compressed: {compressed_size/1024/1024:.2f}MB\n")
        
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
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Start video compression
            compressed_path_future = executor.submit(compress_video, input_path)
            
            # While video is compressing, prepare Modal connection
            modal_setup_future = executor.submit(Function.lookup, "whisper-transcription", "process_video")
            
            # Wait for both tasks to complete
            compressed_path = compressed_path_future.result()
            modal_fn = modal_setup_future.result()
            
            sys.stderr.write(f"Video compressed: {compressed_path}\n")
        
        # Read the compressed video file in chunks for memory efficiency
        sys.stderr.write("Reading compressed video file...\n")
        chunk_size = 1024 * 1024  # 1MB chunks
        video_chunks = []
        with open(compressed_path, 'rb') as video_file:
            while chunk := video_file.read(chunk_size):
                video_chunks.append(chunk)
        video_data = b''.join(video_chunks)
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