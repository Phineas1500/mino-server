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
            # Extract audio using ffmpeg
            sys.stderr.write(f"Extracting audio to {temp_audio_path}...\n")
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', str(input_path),
                '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                '-y', temp_audio_path
            ]
            # Redirect ffmpeg output to prevent interfering with final JSON
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE) 
            
            audio_size = os.path.getsize(temp_audio_path)
            sys.stderr.write(f"Extracted audio: {audio_size} bytes\n")
            
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