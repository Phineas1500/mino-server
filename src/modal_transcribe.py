from modal import Image, Secret, App
import whisper
import json
import os
import openai
import time
from pathlib import Path
from moviepy.editor import VideoFileClip

# Create a Modal app
app = App("whisper-transcription")

def log(message: str, level: str = "INFO") -> None:
    """Helper function for consistent logging"""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [{level}] {message}")

# Define the container image with all necessary dependencies
image = (
    Image.debian_slim(python_version="3.11")
    .apt_install(
        "ffmpeg",
        "git",
        "python3-pip",
        "build-essential",
        "python3-dev",
        "libsndfile1",
        "libglib2.0-0",
        "libgl1-mesa-glx",
        "libsm6",
        "libxext6"
    )
    .pip_install(
        "torch",
        "numpy",
        "ffmpeg-python",
        "openai",
        "tqdm",
        "decorator==4.4.2",
        "imageio-ffmpeg",
        "moviepy==1.0.3",
        "soundfile"
    )
    .run_commands(
        "pip install git+https://github.com/openai/whisper.git"
    )
)

def cleanup_segments(segments, min_duration=1.0):
    """Combine very short segments with neighbors."""
    cleaned = []
    current = None
    
    for segment in segments:
        duration = segment["end"] - segment["start"]
        
        if duration < min_duration and current:
            # Merge with current if too short
            current["end"] = segment["end"]
            current["text"] += " " + segment["text"]
            if "importance_score" in segment and "importance_score" in current:
                # Average the importance scores when merging
                current["importance_score"] = (float(current["importance_score"]) + float(segment["importance_score"])) / 2
        else:
            if current:
                cleaned.append(current)
            current = segment.copy()
    
    if current:
        cleaned.append(current)
    
    return cleaned

def fix_overlapping_segments(segments):
    """Fix any overlapping segment timestamps."""
    fixed = []
    last_end = 0
    
    for segment in sorted(segments, key=lambda x: x["start"]):
        if segment["start"] < last_end:
            segment["start"] = last_end
        
        if segment["start"] < segment["end"]:
            fixed.append(segment)
            last_end = segment["end"]
            
    return fixed
    
def smooth_speed_transition(clip1, clip2, transition_duration=0.5):
    """Create a smooth transition between clips of different speeds."""
    from moviepy.editor import CompositeAudioClip
    
    if clip1 is None or clip2 is None:
        return clip2
        
    # Skip if either clip is too short for transition
    if clip1.duration < transition_duration or clip2.duration < transition_duration:
        return clip2
    
    try:
        # Create a simpler crossfade without volume adjustment
        clip = clip2.crossfadein(transition_duration)
        return clip
    except Exception as e:
        log(f"Error in smooth transition: {str(e)}", "WARNING")
        # Return original clip if transition fails
        return clip2

def analyze_segments_importance(segments, client, summary_data):
    """Analyze segments using GPT-3.5-turbo to determine importance scores."""
    
    # Break segments into smaller chunks to avoid token limits
    CHUNK_SIZE = 10  # Process 10 segments at a time
    all_analyzed_segments = []
    
    for i in range(0, len(segments), CHUNK_SIZE):
        chunk = segments[i:i + CHUNK_SIZE]
        segments_for_analysis = []
        for segment in chunk:
            segments_for_analysis.append({
                "text": segment["text"],
                "start": segment["start"],
                "end": segment["end"]
            })
        
        prompt = f"""Here is a summary of the entire video:
        {summary_data["summary"]}

        The key points of the video are:
        {json.dumps(summary_data["keyPoints"], indent=2)}

        Given this context, analyze these lecture video segments and assign each one an importance score from 0.0 to 1.0, where:
        - 1.0: Absolutely crucial content that directly states a key point
        - 0.8-0.9: Important content that elaborates on key points
        - 0.5-0.7: Supporting content that provides context or examples
        - 0.3-0.4: Background information that could be sped up
        - 0.0-0.2: Filler content, repetition, or verbal pauses

        BE STRICT with scores - not everything can be important. Most segments should be in the middle range.
        Remember that high scores (0.8-1.0) should be rare and reserved only for the most crucial content.

        Return your analysis in this exact JSON format:
        {{
            "segments": [
                {{
                    "start": start_time,
                    "end": end_time,
                    "importance_score": importance_score,
                    "reason": "Brief explanation of the score",
                    "text": "original segment text"
                }},
                ...
            ]
        }}

        Here are the segments to analyze:
        {json.dumps(segments_for_analysis, indent=2)}
        """

        log(f"Analyzing chunk {i//CHUNK_SIZE + 1} of {(len(segments) + CHUNK_SIZE - 1)//CHUNK_SIZE}")
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at analyzing content and identifying truly important information. Be very critical and selective. Most content should not be marked as highly important."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,  # Lower temperature for more consistent scoring
            max_tokens=3000,
            response_format={ "type": "json_object" }
        )

        # Parse the response
        analysis = json.loads(response.choices[0].message.content)
        all_analyzed_segments.extend(analysis["segments"])
        
        # Debug logging for this chunk
        for segment in analysis["segments"]:
            log(f"""Initial Segment Analysis:
            Text: {segment['text'][:100]}...
            Score: {segment['importance_score']}
            Time: {segment['start']:.2f}s - {segment['end']:.2f}s
            Reason: {segment['reason']}""", "DEBUG")

    return all_analyzed_segments
    
def cut_video_segments(video_path, segments, summary_data, min_segment_duration=1.0, max_pause_duration=0.5):
    """Cut video based on importance scores with smooth transitions."""
    try:
        from moviepy.editor import VideoFileClip, concatenate_videoclips
        
        video = VideoFileClip(str(video_path), audio=True)
        video_duration = video.duration
        log(f"Video duration: {video_duration:.2f}s")
        
        # Get OpenAI client
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        
        # Analyze segments importance
        log("Analyzing segments importance with GPT-3.5-turbo...")
        analyzed_segments = analyze_segments_importance(segments, client, summary_data)
        
        # Clean up segments
        analyzed_segments = fix_overlapping_segments(analyzed_segments)
        analyzed_segments = cleanup_segments(analyzed_segments, min_duration=min_segment_duration)
        
        # Ensure segments don't exceed video duration
        cleaned_segments = []
        for segment in analyzed_segments:
            if segment["start"] >= video_duration:
                log(f"Skipping segment starting at {segment['start']:.2f}s as it exceeds video duration", "WARNING")
                continue
            
            # Clip end time to video duration
            segment["end"] = min(segment["end"], video_duration)
            cleaned_segments.append(segment)
        
        analyzed_segments = cleaned_segments
        
        # Group segments by continuous importance levels
        grouped_segments = []
        current_group = []
        
        for i, segment in enumerate(analyzed_segments):
            if not current_group:
                current_group.append(segment)
                continue
                
            prev_score = float(current_group[-1]["importance_score"])
            curr_score = float(segment["importance_score"])
            
            # Stricter grouping logic
            same_thought = (
                abs(prev_score - curr_score) < 0.15 and  # Stricter importance difference
                len(current_group) < 3 and  # Limit group size
                not segment["text"].startswith(('.', '!', '?', 'but', 'however', 'therefore'))  # More transition words
            )
            
            if same_thought:
                current_group.append(segment)
            else:
                grouped_segments.append(current_group)
                current_group = [segment]
        
        if current_group:
            grouped_segments.append(current_group)
        
        # Process each group
        clips_to_keep = []
        total_original_duration = 0
        total_kept_duration = 0
        prev_clip = None
        
        for group in grouped_segments:
            avg_importance = sum(float(s["importance_score"]) for s in group) / len(group)
            start_time = group[0]["start"]
            end_time = min(group[-1]["end"], video_duration)
            
            if start_time >= end_time or start_time >= video_duration:
                log(f"Skipping invalid segment: {start_time:.2f}s - {end_time:.2f}s", "WARNING")
                continue
                
            duration = end_time - start_time
            total_original_duration += duration
            
            log(f"""Processing Group:
            Time: {start_time:.2f}s - {end_time:.2f}s
            Duration: {duration:.2f}s
            Avg Importance: {avg_importance:.2f}
            Text: {' '.join(s['text'] for s in group)[:100]}...""", "DEBUG")
            
            try:
                clip = video.subclip(start_time, end_time)
                
                # More granular speed adjustments with safety checks
                if avg_importance >= 0.8:  # Key content
                    log("Decision: Keep normal speed", "DEBUG")
                    effective_duration = duration
                else:
                    # Calculate speed factor
                    if avg_importance >= 0.6:
                        speed_factor = 1.25
                        log(f"Decision: Speed up slightly {speed_factor:.1f}x", "DEBUG")
                    elif avg_importance >= 0.4:
                        speed_factor = 1.75
                        log(f"Decision: Speed up moderately {speed_factor:.1f}x", "DEBUG")
                    elif avg_importance >= 0.2:
                        speed_factor = 2.25
                        log(f"Decision: Speed up significantly {speed_factor:.1f}x", "DEBUG")
                    else:
                        log("Decision: Skipping segment", "DEBUG")
                        continue

                    try:
                        # First try the simpler speed change method
                        clip = clip.speedx(speed_factor)
                        
                        # If audio exists, try to preserve pitch
                        if clip.audio is not None:
                            try:
                                new_fps = clip.audio.fps * speed_factor
                                clip = clip.set_audio(clip.audio.set_fps(new_fps))
                            except Exception as audio_e:
                                log(f"Warning: Could not adjust audio pitch: {str(audio_e)}", "WARNING")
                        
                        effective_duration = duration / speed_factor
                        
                    except Exception as speed_e:
                        log(f"Warning: Failed to adjust speed, using original clip: {str(speed_e)}", "WARNING")
                        effective_duration = duration
                
                # Add crossfade if not the first clip
                if prev_clip is not None:
                    try:
                        clip = clip.crossfadein(0.3)
                    except Exception as fade_e:
                        log(f"Warning: Could not add crossfade: {str(fade_e)}", "WARNING")
                
                total_kept_duration += effective_duration
                clips_to_keep.append(clip)
                prev_clip = clip
                
            except Exception as e:
                log(f"Error processing segment {start_time:.2f}s - {end_time:.2f}s: {str(e)}", "ERROR")
                continue
        
        # Log overall statistics
        if total_original_duration > 0:
            reduction_percent = (1 - total_kept_duration / total_original_duration) * 100
            log(f"""Video Processing Statistics:
            Original Duration: {total_original_duration:.2f}s
            Final Duration: {total_kept_duration:.2f}s
            Reduction: {reduction_percent:.1f}%""", "INFO")

        if not clips_to_keep:
            log("No valid segments found - using original video", "WARNING")
            output_path = str(video_path)
            video.close()
            return output_path
        
        # Only attempt concatenation if we have valid clips
        if clips_to_keep:
            try:
                log(f"Concatenating {len(clips_to_keep)} clips...")
                final_video = concatenate_videoclips(clips_to_keep)
                
                output_path = str(Path(video_path).with_stem(f"{Path(video_path).stem}_shortened"))
                log(f"Writing shortened video to: {output_path}")
                
                final_video.write_videofile(
                    output_path,
                    codec="libx264",
                    audio_codec="aac",
                    verbose=False,
                    logger=None
                )
                
                # Clean up
                final_video.close()
                
            except Exception as e:
                log(f"Error during concatenation: {str(e)}", "ERROR")
                # If concatenation fails, return original video
                output_path = str(video_path)
        else:
            log("No valid clips to concatenate - using original video", "WARNING")
            output_path = str(video_path)
        
        return output_path
        
    except Exception as e:
        log(f"Error in cut_video_segments: {str(e)}", "ERROR")
        return str(video_path)

@app.function(
    gpu="H100:4",
    image=image,
    timeout=1800,
    secrets=[Secret.from_name("openai-secret")]
)
async def process_video(video_data: bytes, filename: str):
    """Main function that handles video processing, transcription, and summarization"""
    temp_files = []  # Keep track of temporary files
    try:
        # Create a safe filename without spaces
        safe_filename = filename.replace(" ", "_")
        
        # Save video data temporarily
        temp_video_path = Path("/tmp") / safe_filename
        temp_files.append(temp_video_path)  # Track for cleanup
        temp_video_path.write_bytes(video_data)
        log(f"Saved video file: {temp_video_path} ({len(video_data)} bytes)")

        # Create directories if they don't exist
        temp_video_path.parent.mkdir(parents=True, exist_ok=True)

        # Extract audio for transcription
        log("Extracting audio for transcription...")
        video = VideoFileClip(str(temp_video_path))
        temp_audio_path = temp_video_path.with_suffix('.wav')
        temp_files.append(temp_audio_path)  # Track for cleanup
        
        video.audio.write_audiofile(
            str(temp_audio_path),
            codec='pcm_s16le',
            fps=16000,
            ffmpeg_params=["-ac", "1"],
            verbose=False,
            logger=None
        )
        video.close()

        # Transcribe with Whisper
        log("Loading Whisper model...")
        model = whisper.load_model("base")
        
        log("Starting transcription...")
        result = model.transcribe(
            str(temp_audio_path),
            language='en',
            verbose=False
        )
        transcript = result["text"]
        segments = result["segments"]
        log(f"Transcription complete: {len(transcript)} characters")

        # Clean up audio file
        if temp_audio_path.exists():
            temp_audio_path.unlink()

        # If no transcript, return original video
        if not transcript.strip():
            log("No transcript found - returning original video", "WARNING")
            return {
                "status": "success",
                "summary": "Video contains no spoken content.",
                "keyPoints": ["No key points available."] * 5,
                "flashcards": [{"question": "No content available.", "answer": "No content available."}] * 5,
                "transcript": "",
                "segments": [],
                "shortened_video": video_data
            }

        # Process with OpenAI
        log("Processing with OpenAI...")
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        prompt = f"""Analyze this lecture transcript and provide a response in this EXACT JSON format:
        {{
            "summary": "A clear 2-3 paragraph summary of the main concepts",
            "keyPoints": [
                "First key point as a complete sentence",
                "Second key point as a complete sentence",
                "Third key point as a complete sentence",
                "Fourth key point as a complete sentence",
                "Fifth key point as a complete sentence"
            ],
            "flashcards": [
                {{
                    "question": "Basic concept question about a key term or idea?",
                    "answer": "A clear and concise answer to the basic concept."
                }},
                {{
                    "question": "Fundamental question about the content?",
                    "answer": "A straightforward explanation of the fundamental concept."
                }},
                {{
                    "question": "Question testing basic understanding?",
                    "answer": "A comprehensive answer demonstrating basic mastery."
                }},
                {{
                    "question": "More challenging question requiring synthesis of multiple concepts?",
                    "answer": "A detailed answer that connects multiple ideas and demonstrates deeper understanding."
                }},
                {{
                    "question": "Advanced inference question about implications or applications?",
                    "answer": "A sophisticated answer that extends beyond the explicit content to explore implications or real-world applications."
                }}
            ]
        }}

        The response MUST:
        1. Include a detailed summary that captures the main ideas
        2. Have exactly 5 key points as complete sentences
        3. Have exactly 5 flashcards with clear questions and answers, where:
           - First 3 cards test basic understanding of explicit content
           - Fourth card requires connecting multiple concepts
           - Fifth card tests ability to make inferences or applications
        4. Be in valid JSON format

        Here is the transcript to analyze:
        {transcript}
        """

        log("Sending request to OpenAI...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that creates educational content from video transcripts. You MUST return responses in valid JSON format with the exact structure specified. For flashcards, create a progression from basic recall to advanced synthesis and application."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=3000,
            response_format={ "type": "json_object" }
        )

        # Parse OpenAI response
        summary_data = json.loads(response.choices[0].message.content)
        log("Successfully processed with OpenAI")

        # Cut video - now handles empty segments gracefully
        log("Starting video cutting process...")
        shortened_path = cut_video_segments(
            temp_video_path,
            segments,
            summary_data,
            min_segment_duration=1.0,
            max_pause_duration=0.5
        )
        
        # Read the video data
        with open(shortened_path, 'rb') as f:
            shortened_video_data = f.read()

        # Clean up files
        if temp_video_path.exists():
            temp_video_path.unlink()
        if shortened_path != str(temp_video_path) and Path(shortened_path).exists():
            Path(shortened_path).unlink()

        # Return the processed results
        return {
            "status": "success",
            "summary": summary_data["summary"],
            "keyPoints": summary_data["keyPoints"][:5],
            "flashcards": summary_data["flashcards"][:5],
            "transcript": transcript,
            "segments": [
                {
                    "start": s["start"],
                    "end": s["end"],
                    "text": s["text"]
                }
                for s in segments
            ],
            "shortened_video": shortened_video_data if "shortened_video_data" in locals() else video_data
        }

    except Exception as e:
        log(f"Error in process_video: {str(e)}", "ERROR")
        # Return the original video in case of error
        return {
            "status": "success",
            "summary": "Error processing video.",
            "keyPoints": ["Error processing video."] * 5,
            "flashcards": [{"question": "Error processing video.", "answer": "Error processing video."}] * 5,
            "transcript": "",
            "segments": [],
            "shortened_video": video_data
        }
    finally:
        # Clean up all temporary files
        for temp_file in temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except Exception as e:
                log(f"Error cleaning up {temp_file}: {str(e)}", "WARNING")


if __name__ == "__main__":
    app.serve() 