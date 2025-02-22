import cv2
import numpy as np
import whisper
import moviepy.editor as mp
from pathlib import Path
import json
import sys
import os
import re

def extract_key_points(text):
    """Extract key points from the transcription text"""
    # Split into sentences
    sentences = re.split('[.!?]', text)
    # Filter out empty sentences and trim
    sentences = [s.strip() for s in sentences if s.strip()]
    # Select important sentences (for this simple version, we'll take the first few unique ones)
    key_points = []
    seen = set()
    for sentence in sentences:
        if len(key_points) >= 4:  # Limit to 4 key points
            break
        # Only add if sentence is substantial and not too similar to existing points
        if len(sentence) > 20 and sentence not in seen:
            key_points.append(sentence)
            seen.add(sentence)
    return key_points

def create_flashcards(text, segments):
    """Create flashcards from the transcription"""
    flashcards = []
    current_segment = ""
    
    # Combine segments into meaningful chunks
    for segment in segments:
        current_segment += segment["text"] + " "
        if len(current_segment) > 100:  # Create a flashcard after collecting enough text
            # Create a question from the segment
            question = f"What is discussed in this segment: '{current_segment[:50]}...'"
            answer = current_segment.strip()
            flashcards.append({
                "question": question,
                "answer": answer
            })
            current_segment = ""
            
            if len(flashcards) >= 3:  # Limit to 3 flashcards
                break
    
    return flashcards

def process_video(input_path, output_path, params_json=None):
    """
    Process video with various effects and transcription
    params_json: JSON string containing processing parameters
    """
    try:
        # Load parameters if provided
        params = json.loads(params_json) if params_json else {}
        input_path = Path(input_path)
        
        # First do the transcription
        print("Starting transcription process...")
        
        # Extract audio for transcription
        video = mp.VideoFileClip(str(input_path))
        temp_audio_path = input_path.with_suffix('.wav')
        video.audio.write_audiofile(str(temp_audio_path), codec='pcm_s16le')
        video.close()
        
        # Load Whisper model and transcribe
        print("Loading Whisper model...")
        model = whisper.load_model("base")
        
        print("Transcribing audio...")
        transcription_result = model.transcribe(str(temp_audio_path))
        
        # Clean up temporary audio file
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            print("Cleaned up temporary audio file")
        
        # Process transcription into required format
        full_text = transcription_result["text"]
        key_points = extract_key_points(full_text)
        flashcards = create_flashcards(full_text, transcription_result["segments"])
        
        # Create summary (first 2 sentences or 200 characters)
        summary = ". ".join(re.split('[.!?]', full_text)[:2]) + "."
        if len(summary) > 200:
            summary = summary[:197] + "..."
        
        # Now process the video
        print("Processing video...")
        cap = cv2.VideoCapture(str(input_path))
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Example processing steps (can be customized based on params)
            
            # 1. Apply Gaussian blur for noise reduction
            blurred = cv2.GaussianBlur(frame, (5, 5), 0)
            
            # 2. Enhance contrast
            lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            enhanced = cv2.merge((cl,a,b))
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # Write the processed frame
            out.write(enhanced)
            frame_count += 1
            
        # Release everything
        cap.release()
        out.release()
        
        # Return processing stats and transcription
        return {
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
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

if __name__ == "__main__":
    # Expect arguments: input_path output_path [params_json]
    if len(sys.argv) < 3:
        print(json.dumps({"status": "error", "error": "Insufficient arguments"}))
        sys.exit(1)
        
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    params_json = sys.argv[3] if len(sys.argv) > 3 else None
    
    result = process_video(input_path, output_path, params_json)
    print(json.dumps(result)) 