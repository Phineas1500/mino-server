import cv2
import numpy as np
import whisper
import moviepy.editor as mp
from pathlib import Path
import json
import sys
import os
import re
import logging

# Configure logging to write to stderr
logging.basicConfig(level=logging.INFO, stream=sys.stderr, format='%(message)s')
logger = logging.getLogger(__name__)

def extract_key_points(text):
    """Extract key points from the transcription text"""
    sentences = re.split('[.!?]', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    key_points = []
    seen = set()
    for sentence in sentences:
        if len(key_points) >= 4:
            break
        if len(sentence) > 20 and sentence not in seen:
            key_points.append(sentence)
            seen.add(sentence)
    return key_points

def create_flashcards(text, segments):
    """Create flashcards from the transcription"""
    flashcards = []
    current_segment = ""
    
    for segment in segments:
        current_segment += segment["text"] + " "
        if len(current_segment) > 100:
            question = f"What is discussed in this segment: '{current_segment[:50]}...'"
            answer = current_segment.strip()
            flashcards.append({
                "question": question,
                "answer": answer
            })
            current_segment = ""
            
            if len(flashcards) >= 3:
                break
    
    return flashcards

def process_video(input_path, output_path, params_json=None):
    """Process video with various effects and transcription"""
    try:
        params = json.loads(params_json) if params_json else {}
        transcription_only = params.get('transcriptionOnly', False)
        input_path = Path(input_path)
        
        logger.info("Starting transcription process...")
        
        # Extract audio for transcription
        video = mp.VideoFileClip(str(input_path))
        temp_audio_path = input_path.with_suffix('.wav')
        video.audio.write_audiofile(str(temp_audio_path), codec='pcm_s16le')
        video.close()
        
        logger.info("Loading Whisper model...")
        model = whisper.load_model("base")
        
        logger.info("Transcribing audio...")
        transcription_result = model.transcribe(str(temp_audio_path))
        
        # Clean up temporary audio file
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            logger.info("Cleaned up temporary audio file")
        
        full_text = transcription_result["text"]
        key_points = extract_key_points(full_text)
        flashcards = create_flashcards(full_text, transcription_result["segments"])
        
        summary = ". ".join(re.split('[.!?]', full_text)[:2]) + "."
        if len(summary) > 200:
            summary = summary[:197] + "..."

        # Get video info using moviepy instead of OpenCV
        if transcription_only:
            try:
                # Use moviepy to get video info
                with mp.VideoFileClip(str(input_path)) as video:
                    fps = video.fps
                    frame_width = int(video.size[0])
                    frame_height = int(video.size[1])
                    frame_count = int(video.duration * video.fps)
            except Exception as e:
                logger.warning(f"Error getting video info: {e}")
                # Fallback values
                fps = 30.0
                frame_width = 1280
                frame_height = 720
                frame_count = 0
        else:
            # Original video processing code
            logger.info("Processing video...")
            cap = cv2.VideoCapture(str(input_path))
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                blurred = cv2.GaussianBlur(frame, (5, 5), 0)
                
                lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                cl = clahe.apply(l)
                enhanced = cv2.merge((cl,a,b))
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
                
                out.write(enhanced)
                frame_count += 1
                
            cap.release()
            out.release()
        
        result = {
            "status": "success",
            "frames_processed": frame_count,
            "fps": fps,
            "dimensions": f"{frame_width}x{frame_height}",
            "transcription_data": {
                "summary": summary,
                "keyPoints": key_points,
                "flashcards": flashcards,
                "transcript": full_text,
                "segments": transcription_result["segments"]
            }
        }
        # Print only the JSON result to stdout
        print(json.dumps(result))
        return result
        
    except Exception as e:
        error_result = {
            "status": "error",
            "error": str(e)
        }
        print(json.dumps(error_result))
        return error_result
        
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(json.dumps({"status": "error", "error": "Insufficient arguments"}))
        sys.exit(1)
        
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    params_json = sys.argv[3] if len(sys.argv) > 3 else None
    
    process_video(input_path, output_path, params_json)