import whisper
import moviepy.editor as mp
from pathlib import Path
import json
import sys
import os

def process_video(input_path, output_path):
    """
    Simple video processing: just extract audio and transcribe
    """
    try:
        input_path = Path(input_path)
        
        # Extract audio for transcription
        sys.stderr.write("Extracting audio...\n")
        video = mp.VideoFileClip(str(input_path))
        temp_audio_path = input_path.with_suffix('.wav')
        video.audio.write_audiofile(str(temp_audio_path), codec='pcm_s16le', logger=None)
        video.close()
        
        # Load Whisper model and transcribe
        sys.stderr.write("Loading Whisper model...\n")
        model = whisper.load_model("base")
        
        sys.stderr.write("Transcribing audio...\n")
        result = model.transcribe(str(temp_audio_path))
        
        # Clean up temporary audio file
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            sys.stderr.write("Cleaned up temporary audio file\n")

        # Write transcript with timestamps
        transcript_path = input_path.with_suffix('.txt')
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write("Full Transcription:\n")
            f.write("=================\n\n")
            f.write(result["text"])
            f.write("\n\n")
            
            f.write("Segments with Timestamps:\n")
            f.write("=======================\n\n")
            segments_text = []
            for segment in result["segments"]:
                start = segment["start"]
                end = segment["end"]
                text = segment["text"]
                segments_text.append(f"[{start:.2f}s -> {end:.2f}s] {text}")
                f.write(f"[{start:.2f}s -> {end:.2f}s] {text}\n")
        
        # Return simple response
        output = {
            "status": "success",
            "transcript_file": str(transcript_path),
            "segments": segments_text
        }
        print(json.dumps(output, separators=(',', ':')))
        sys.stdout.flush()
        return
        
    except Exception as e:
        error_output = {
            "status": "error",
            "error": str(e)
        }
        print(json.dumps(error_output, separators=(',', ':')))
        sys.stdout.flush()
        return

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_data = {
            "status": "success",
            "transcript_file": "test.txt",
            "segments": [
                "[0.00s -> 5.00s] This is a test transcript."
            ]
        }
        print(json.dumps(test_data, separators=(',', ':')))
        sys.stdout.flush()
    else:
        if len(sys.argv) < 3:
            print(json.dumps({
                "status": "error",
                "error": "Insufficient arguments"
            }, separators=(',', ':')))
            sys.exit(1)
        
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        process_video(input_path, output_path) 