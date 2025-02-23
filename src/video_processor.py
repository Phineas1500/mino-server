import moviepy.editor as mp
from pathlib import Path
import json
import sys
import os
import traceback
from modal import Function

def process_video(input_path, output_path):
    """
    Extract audio and send to Modal for transcription
    """
    try:
        sys.stderr.write(f"Starting process_video with input: {input_path}\n")
        input_path = Path(input_path)
        
        # In the process_video function, after creating WAV file:
        sys.stderr.write("Extracting audio...\n")
        video = mp.VideoFileClip(str(input_path))
        temp_audio_path = input_path.with_suffix('.wav')
        video.audio.write_audiofile(
            str(temp_audio_path),
            codec='pcm_s16le',
            ffmpeg_params=["-ac", "1"],  # Convert to mono
            fps=16000,  # Use 16kHz sampling rate
            logger=None
        )
        video.close()

        # Add these debug checks:
        sys.stderr.write(f"Audio file created at: {temp_audio_path}\n")
        sys.stderr.write(f"Audio file exists: {os.path.exists(temp_audio_path)}\n")
        sys.stderr.write(f"Audio file size: {os.path.getsize(temp_audio_path)} bytes\n")

        sys.stderr.write("Reading audio file for Modal...\n")
        with open(temp_audio_path, 'rb') as audio_file:
            audio_data = audio_file.read()
        sys.stderr.write(f"Read {len(audio_data)} bytes of audio data\n")
        
        sys.stderr.write("Connecting to Modal service...\n")
        transcribe_fn = Function.lookup("whisper-transcription", "transcribe_audio")
        
        sys.stderr.write("Sending to Modal for transcription...\n")
        result = transcribe_fn.remote(audio_data, temp_audio_path.name)
        sys.stderr.write(f"Received result from Modal: {result}\n")
        
        if result.get("status") == "error":
            raise Exception(f"Modal transcription failed: {result.get('error')}")
        
        # Clean up temporary audio file
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            sys.stderr.write("Cleaned up temporary audio file\n")

        # Write transcript
        transcript_path = input_path.with_suffix('.html')
        with open(transcript_path, 'w', encoding='utf-8') as f:
            # f.write("Full Transcription:\n")
            # f.write("=======+=========\n\n")
            # f.write(result["transcript"])
            # f.write("\n\n")

            
            # f.write("Segments with Timestamps:\n")
            # f.write("=======================\n<br>")
            # segments_text = []
            # for segment in result["segments"]:
            #     text = f"[{segment['start']:.2f}s -> {segment['end']:.2f}s] {segment['text']}\n"
            #     segments_text.append(text)
            #     f.write("\n" + text + "<br>")

            # f.write("""<!DOCTYPE html>
            #     <html lang="en">
            #     <head>
            #         <meta charset="UTF-8">
            #         <title>Transcript</title>
            #         <style>
            #             body { font-family: Arial, sans-serif; }
            #             pre { white-space: pre-wrap; }
            #         </style>
            #     </head>
            #     <body>
            #         <h2>Segments with Timestamps:</h2>
            #         <hr>
            #         <pre>
            #     """)
            # f.write("<html><body>\n")
            # f.write("<h2>Segments with Timestamps:</h2>\n")
            # f.write("<hr>\n")
            segments_text = []
            for segment in result["segments"]:
                text = f"[{segment['start']:.2f}s -> {segment['end']:.2f}s] {segment['text']}\n"
                segments_text.append(text)
                f.write(text)
            # f.write("</body></html>")
            # f.write("""    </pre>
            #     </body>
            #     </html>
            #     """)
        
        output = {
            "status": "success",
            "transcript_file": str(transcript_path),
            "segments": segments_text
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