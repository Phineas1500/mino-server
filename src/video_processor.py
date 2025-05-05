import sys
import json
import os
import traceback
import tempfile
import subprocess
from pathlib import Path
# Use Stub for generator support
from modal import Function # Import Stub

def process_video(input_path, output_path):
    """
    Extract audio from video, send audio to Modal for processing (streaming progress),
    and return the final result.
    """
    final_result = None # Initialize final_result
    temp_audio_path = None # Initialize path
    try:
        input_path = Path(input_path)
        filename = input_path.name

        sys.stderr.write(f"Starting process_video with input: {input_path}\n")

        # Create a temporary file for the audio
        # Use try...finally for temp file creation/deletion
        try:
            # Create temp file within the main try block
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                temp_audio_path = temp_audio.name

            # Extract audio using ffmpeg
            sys.stderr.write(f"Extracting audio to {temp_audio_path}...\n")
            ffmpeg_cmd = [
                'ffmpeg', '-i', str(input_path),
                '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                '-y', temp_audio_path
            ]
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

            audio_size = os.path.getsize(temp_audio_path)
            sys.stderr.write(f"Extracted audio: {audio_size} bytes\n")

            with open(temp_audio_path, 'rb') as audio_file:
                audio_data = audio_file.read()

            # Connect to Modal
            sys.stderr.write("Connecting to Modal service...\n")
            # Ensure the function name matches the one in modal_transcribe.py
            modal_fn = Function.from_name("whisper-transcription", "process_audio")

            sys.stderr.write(f"Calling Modal function for: {filename} (streaming using .remote_gen())...\n")

            # --- Use streaming call (.remote_gen()) ---
            for update in modal_fn.remote_gen(audio_data, filename):
                update_type = update.get("type")

                if update_type == "progress":
                    # Format and print progress for Node.js
                    percentage = update.get("percentage", 0)
                    stage = update.get("stage", "unknown")
                    message = update.get("message", "")
                    progress_line = f"PROGRESS: {percentage} STAGE: {stage}"
                    if message:
                        progress_line += f" MESSAGE: {message}"
                    print(progress_line, flush=True) # Print to stdout
                    sys.stdout.flush() # Explicit flush just in case
                    sys.stderr.write(f"Modal Progress: {percentage}% - {stage} - {message}\n") # Also log to stderr for debugging

                elif update_type == "result":
                    # Store the final result
                    final_result = update.get("data")
                    sys.stderr.write("Modal function returned final result.\n")
                    # Don't break here, let the generator finish naturally

                elif update_type == "error":
                    # Handle errors yielded by Modal function
                    error_msg = update.get("error", "Unknown error from Modal function")
                    sys.stderr.write(f"Modal function yielded error: {error_msg}\n")
                    raise Exception(f"Modal function failed: {error_msg}")

                else:
                    sys.stderr.write(f"Received unknown update type from Modal: {update_type}\n")
            # --- End streaming call ---

            sys.stderr.write("Modal function call completed.\n")

        finally:
            # Clean up the temporary audio file
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
                sys.stderr.write(f"Cleaned up temporary audio file: {temp_audio_path}\n")

        # --- Process the final result ---
        if final_result is None:
             raise Exception("Did not receive a final result from Modal function.")

        # Check if Modal function returned an error status internally (redundant if errors are yielded)
        if not isinstance(final_result, dict) or final_result.get("status") != "success":
             error_detail = final_result.get('error', 'Unknown error in final result') if isinstance(final_result, dict) else str(final_result)
             raise Exception(f"Modal function failed (final result status): {error_detail}")

        # Output is already structured correctly in final_result['data']
        output = final_result # Use the data directly

        sys.stderr.write("Processing completed successfully\n")
        # Print the FINAL JSON result to stdout for Node.js
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
        # Print the ERROR JSON result to stdout for Node.js
        # Ensure no progress lines interfere here
        print(json.dumps(error_output, ensure_ascii=False))
        sys.stdout.flush()

# --- Main execution block remains the same ---
if __name__ == "__main__":
    # ... (argument parsing remains the same) ...
    if len(sys.argv) < 3:
        # ... error handling ...
        sys.exit(1)

    if sys.argv[1] == "test":
        # ... test code ...
        sys.exit(0)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    process_video(input_path, output_path)