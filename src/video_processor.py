import sys
import json
import os
import traceback
import tempfile
import subprocess
from pathlib import Path
# Remove FunctionCall import if not needed elsewhere
from modal import Function 

def process_video(input_path, output_path):
    """
    Extract audio from video, send audio to Modal for processing,
    and return the final result. (No live streaming)
    """
    try:
        input_path = Path(input_path)
        filename = input_path.name
        
        sys.stderr.write(f"Starting process_video with input: {input_path}\n")
        
        # Create a temporary file for the audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
        
        result = None # Initialize result

        try:
            # First, check if the input file exists and has content
            if not input_path.exists():
                raise Exception(f"Input video file does not exist: {input_path}")
            
            file_size = input_path.stat().st_size
            sys.stderr.write(f"Input video file size: {file_size} bytes\n")
            if file_size == 0:
                raise Exception(f"Input video file is empty: {input_path}")
            
            # Try to get video info first to check if it's valid
            sys.stderr.write("Checking video file with ffprobe...\n")
            try:
                ffprobe_cmd = [
                    'ffprobe', '-v', 'quiet', '-print_format', 'json', 
                    '-show_streams', '-show_format', str(input_path)
                ]
                probe_result = subprocess.run(ffprobe_cmd, capture_output=True, text=True, check=True)
                probe_info = json.loads(probe_result.stdout)
                
                # Check if there are any audio streams
                audio_streams = [s for s in probe_info.get('streams', []) if s.get('codec_type') == 'audio']
                sys.stderr.write(f"Found {len(audio_streams)} audio stream(s)\n")
                
                if not audio_streams:
                    sys.stderr.write("Warning: No audio streams found in video\n")
                
            except subprocess.CalledProcessError as e:
                sys.stderr.write(f"ffprobe failed: {e}\n")
                sys.stderr.write(f"ffprobe stderr: {e.stderr}\n")
                # Continue anyway, might still work
            except Exception as e:
                sys.stderr.write(f"Error running ffprobe: {e}\n")
                # Continue anyway
            
            # Extract audio using ffmpeg with better error handling
            sys.stderr.write(f"Extracting audio to {temp_audio_path}...\n")
            
            # Try multiple ffmpeg configurations
            ffmpeg_configs = [
                # Config 1: Standard extraction
                [
                    'ffmpeg', '-i', str(input_path),
                    '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                    '-y', temp_audio_path
                ],
                # Config 2: More permissive with auto codec detection
                [
                    'ffmpeg', '-i', str(input_path),
                    '-vn', '-ar', '16000', '-ac', '1',
                    '-y', temp_audio_path
                ],
                # Config 3: Force decode any audio present
                [
                    'ffmpeg', '-i', str(input_path),
                    '-map', '0:a?', '-ar', '16000', '-ac', '1',
                    '-y', temp_audio_path
                ]
            ]
            
            extraction_success = False
            last_error = ""
            
            for i, ffmpeg_cmd in enumerate(ffmpeg_configs):
                try:
                    sys.stderr.write(f"Trying ffmpeg config {i+1}...\n")
                    result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
                    
                    # Check if audio file was created and has content
                    if os.path.exists(temp_audio_path) and os.path.getsize(temp_audio_path) > 0:
                        extraction_success = True
                        audio_size = os.path.getsize(temp_audio_path)
                        sys.stderr.write(f"Audio extraction successful with config {i+1}: {audio_size} bytes\n")
                        break
                    else:
                        sys.stderr.write(f"Config {i+1} produced no output\n")
                        
                except subprocess.CalledProcessError as e:
                    last_error = f"Config {i+1} failed: {e.stderr}"
                    sys.stderr.write(f"{last_error}\n")
                    continue
            
            if not extraction_success:
                # Final fallback: if video has no audio, generate silent audio track
                if not audio_streams:
                    sys.stderr.write("No audio found - generating silent audio track...\n")
                    try:
                        # Get video duration for silent audio generation
                        duration_cmd = [
                            'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                            '-of', 'csv=p=0', str(input_path)
                        ]
                        duration_result = subprocess.run(duration_cmd, capture_output=True, text=True, check=True)
                        duration = float(duration_result.stdout.strip())
                        
                        # Generate silent audio track matching video duration
                        silent_cmd = [
                            'ffmpeg', '-f', 'lavfi', '-i', f'anullsrc=channel_layout=mono:sample_rate=16000',
                            '-t', str(duration), '-y', temp_audio_path
                        ]
                        subprocess.run(silent_cmd, check=True, capture_output=True, text=True)
                        
                        if os.path.exists(temp_audio_path) and os.path.getsize(temp_audio_path) > 0:
                            extraction_success = True
                            audio_size = os.path.getsize(temp_audio_path)
                            sys.stderr.write(f"Generated silent audio track: {audio_size} bytes\n")
                        
                    except Exception as e:
                        sys.stderr.write(f"Failed to generate silent audio: {e}\n")
                
                if not extraction_success:
                    raise Exception(f"All ffmpeg configurations failed and unable to generate silent audio. Last error: {last_error}")
            
            audio_size = os.path.getsize(temp_audio_path)
            sys.stderr.write(f"Final extracted audio: {audio_size} bytes\n")
            
            with open(temp_audio_path, 'rb') as audio_file:
                audio_data = audio_file.read()
                
            # Connect to Modal
            sys.stderr.write("Connecting to Modal service...\n")
            modal_fn = Function.from_name("whisper-transcription", "process_audio") 
            
            # --- Use blocking call (.remote()) ---
            sys.stderr.write(f"Calling Modal function for: {filename} (blocking using .remote())...\n")
            # --- Revert back to using .remote() ---
            result = modal_fn.remote(audio_data, filename) 
            sys.stderr.write("Modal function call completed.\n")
            # --- End blocking call ---
            
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
                sys.stderr.write(f"Cleaned up temporary audio file: {temp_audio_path}\n")
        
        # --- Process the final result ---
        if result is None:
             raise Exception("Did not receive a result from Modal function.")

        # Check if Modal function returned an error status internally
        if not isinstance(result, dict) or result.get("status") == "error":
             error_detail = result.get('error', 'Unknown error') if isinstance(result, dict) else str(result)
             raise Exception(f"Modal function failed: {error_detail}")
            
        output = {
            "status": "success",
            "transcript_file": None, # Not saving locally anymore
            "transcript": result.get("transcript", ""),
            "summary": result.get("summary", ""),
            "keyPoints": result.get("keyPoints", []),
            "flashcards": result.get("flashcards", []),
            "segments": result.get("segments", []),
            "stats": result.get("stats", {})
        }
        
        sys.stderr.write("Processing completed successfully\n")
        # Print the FINAL JSON result to stdout for Node.js
        # Ensure ONLY the final JSON goes to stdout
        print(json.dumps(output, ensure_ascii=False)) 
        sys.stdout.flush()
        
    except Exception as e:
        sys.stderr.write(f"Error in process_video: {str(e)}\n")
        sys.stderr.write(traceback.format_exc())
        
        # No Modal call object 'call' exists here to cancel

        error_output = {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        # Print the ERROR JSON result to stdout for Node.js
        print(json.dumps(error_output, ensure_ascii=False))
        sys.stdout.flush()

# --- Main execution block remains the same ---
if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.stderr.write("Insufficient arguments\n")
        print(json.dumps({
            "status": "error",
            "error": "Insufficient arguments"
        }))
        sys.exit(1)
    
    if sys.argv[1] == "test":
        # ... (test code remains the same) ...
        sys.exit(0)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] # output_path is not really used now, but kept for argument compatibility
    process_video(input_path, output_path)