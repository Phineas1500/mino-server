from modal import Image, Secret, App
import whisper
import json
import os
import openai
import time
from pathlib import Path
from moviepy.editor import VideoFileClip
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import traceback

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
        "soundfile",
        "transformers"
    )
    .run_commands(
        "pip install git+https://github.com/openai/whisper.git"
    )
)

def cut_video_segments(video_path, segments, min_segment_duration=1.0, max_pause_duration=0.5):
    """Cut video based on transcript segments, removing long pauses."""
    try:
        from moviepy.editor import VideoFileClip, concatenate_videoclips
        
        log(f"Opening video file for cutting: {video_path}")
        # Explicitly set audio=True to ensure audio is loaded
        video = VideoFileClip(str(video_path), audio=True)
        
        if not video.fps:
            raise ValueError("Could not determine video FPS")
            
        clips_to_keep = []
        
        for i, segment in enumerate(segments):
            segment_duration = segment["end"] - segment["start"]
            
            if segment_duration < min_segment_duration:
                continue
                
            if i > 0:
                pause_duration = segment["start"] - segments[i-1]["end"]
                if pause_duration > max_pause_duration:
                    segment["start"] = segments[i-1]["end"] + max_pause_duration
            
            log(f"Processing segment {i+1}/{len(segments)}: {segment['start']:.2f}s to {segment['end']:.2f}s")
            clip = video.subclip(segment["start"], segment["end"])
            clips_to_keep.append(clip)
        
        if not clips_to_keep:
            log("No valid segments found - using original video", "WARNING")
            output_path = str(video_path)
            video.close()
            return output_path
            
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
        video.close()
        final_video.close()
        for clip in clips_to_keep:
            clip.close()
            
        return output_path
        
    except Exception as e:
        log(f"Error in cut_video_segments: {str(e)}", "ERROR")
        # If there's an error, return the original video path
        return str(video_path)

@app.function(
    gpu="H100",
    image=image,
    timeout=1800,
    secrets=[Secret.from_name("openai-secret")]
)
async def optimize_playback_speed(segments):
    """Analyze transcript segments and determine optimal playback speeds"""
    try:
        log("Starting playback speed optimization...")
        
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        optimized_segments = []
        total_duration = sum(seg["end"] - seg["start"] for seg in segments)
        optimized_duration = 0

        for i, segment in enumerate(segments):
            try:
                log(f"Processing segment {i+1}/{len(segments)}")
                
                prompt = f"""Rate the educational importance of this lecture segment on a scale of 1-10 and explain why.

                Guidelines:
                10 = Crucial concept, key definition, or fundamental principle
                7-9 = Important examples, core ideas, or detailed explanations
                4-6 = Supporting information, context, or basic examples
                1-3 = Repetition, tangents, or very basic content

                Segment: "{segment['text']}"

                Respond in this exact JSON format:
                {{
                    "importance_score": <number 1-10>,
                    "reason": "<brief explanation of the rating>"
                }}"""

                # Get OpenAI's analysis
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert in educational content analysis. Analyze lecture segments and rate their importance. Always respond in valid JSON format."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.3,
                    max_tokens=150,
                    response_format={ "type": "json_object" }
                )

                # Parse response
                analysis = json.loads(response.choices[0].message.content)
                
                # Validate and normalize importance score
                importance_score = float(analysis["importance_score"])
                importance_score = max(1, min(10, importance_score))  # Ensure between 1 and 10
                
                # Calculate playback speed based on importance score
                # Formula: speed ranges from 1.0 (score 10) to 2.0 (score 1)
                speed = 1.0 + ((10 - importance_score) / 9)  # Linear scaling
                speed = round(min(max(speed, 1.0), 2.0), 2)  # Ensure between 1.0 and 2.0
                
                # Calculate optimized duration for this segment
                segment_duration = segment["end"] - segment["start"]
                optimized_duration += segment_duration / speed
                
                # Add optimized segment
                optimized_segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"],
                    "speed": speed,
                    "importance_score": importance_score,
                    "reason": analysis["reason"]
                })
                
                log(f"Processed segment {i+1}: Score={importance_score}, Speed={speed:.2f}x")
                
            except Exception as e:
                log(f"Error processing segment {i+1}: {str(e)}", "WARNING")
                # Use conservative defaults for this segment
                optimized_segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"],
                    "speed": 1.0,
                    "importance_score": 5,
                    "reason": "Processing error, using default speed"
                })
                optimized_duration += segment["end"] - segment["start"]

        # Calculate statistics
        time_saved = total_duration - optimized_duration
        time_saved_percentage = (time_saved / total_duration) * 100 if total_duration > 0 else 0
        
        log(f"Optimization complete. Time saved: {time_saved:.2f} seconds ({time_saved_percentage:.1f}%)")
        
        return {
            "segments": optimized_segments,
            "stats": {
                "original_duration": total_duration,
                "optimized_duration": optimized_duration,
                "time_saved": time_saved,
                "time_saved_percentage": time_saved_percentage
            }
        }

    except Exception as e:
        log(f"Error in optimize_playback_speed: {str(e)}", "ERROR")
        # Return unoptimized segments if optimization fails
        return {
            "segments": [{
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "speed": 1.0,
                "importance_score": 5,
                "reason": "Optimization failed, using default speed"
            } for seg in segments],
            "stats": {
                "original_duration": sum(seg["end"] - seg["start"] for seg in segments),
                "optimized_duration": sum(seg["end"] - seg["start"] for seg in segments),
                "time_saved": 0,
                "time_saved_percentage": 0
            }
        }

@app.function(
    gpu="H100",
    image=image,
    timeout=1800,
    secrets=[Secret.from_name("openai-secret")]
)
async def process_video(video_data: bytes, filename: str):
    """Main function that handles video processing, transcription, and content analysis"""
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

        try:
            # Extract audio for transcription
            log("Extracting audio for transcription...")
            video = VideoFileClip(str(temp_video_path))
            if not video.audio:
                raise ValueError("Video has no audio track")
                
            temp_audio_path = temp_video_path.with_suffix('.wav')
            temp_files.append(temp_audio_path)  # Track for cleanup
            
            log(f"Writing audio to {temp_audio_path}")
            video.audio.write_audiofile(
                str(temp_audio_path),
                codec='pcm_s16le',
                fps=16000,
                ffmpeg_params=["-ac", "1"],
                verbose=True,
                logger=None
            )
            video.close()
            log("Audio extraction complete")

            # Verify audio file exists and has content
            if not temp_audio_path.exists():
                raise FileNotFoundError(f"Audio file not created at {temp_audio_path}")
            audio_size = temp_audio_path.stat().st_size
            log(f"Audio file size: {audio_size} bytes")
            if audio_size == 0:
                raise ValueError("Audio file is empty")

            # Transcribe with Whisper
            log("Loading Whisper model...")
            # Use tiny model instead of base for faster processing
            model = whisper.load_model("tiny")
            
            log("Starting transcription...")
            result = model.transcribe(
                str(temp_audio_path),
                language='en',
                verbose=True
            )
            
            if not result or not isinstance(result, dict):
                raise ValueError(f"Invalid Whisper result: {result}")
                
            transcript = result.get("text", "")
            segments = result.get("segments", [])
            
            log(f"Transcription complete: {len(transcript)} characters, {len(segments)} segments")

            # Clean up audio file early
            if temp_audio_path.exists():
                temp_audio_path.unlink()
                log("Cleaned up audio file")

            # Process with OpenAI for summary, key points, and flashcards
            log("Generating educational content...")
            client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

            content_prompt = f"""Analyze this lecture transcript and provide a response in this EXACT JSON format:
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

            The response MUST be in valid JSON format.
            Here is the transcript: {transcript}"""

            log("Generating educational content with OpenAI...")
            content_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that creates educational content from video transcripts. You MUST return responses in valid JSON format with the exact structure specified."
                    },
                    {
                        "role": "user",
                        "content": content_prompt
                    }
                ],
                temperature=0.7,
                max_tokens=2000,
                response_format={ "type": "json_object" }
            )

            # Parse OpenAI response for educational content
            content_data = json.loads(content_response.choices[0].message.content)
            log("Successfully generated educational content")

            # Process segments in batches for faster analysis
            log("Analyzing segments for importance ratings...")
            analyzed_segments = []
            total_duration = 0
            skippable_duration = 0
            
            # Process segments in batches of 5
            batch_size = 5
            for i in range(0, len(segments), batch_size):
                batch = segments[i:i + batch_size]
                batch_texts = [seg["text"] for seg in batch]
                
                try:
                    importance_prompt = f"""Rate each lecture segment's importance in context of the entire lecture on a scale of 1-10 and explain why. Be extra lenient with the rating.

                    Guidelines:
                    10 = Crucial concept, key definition, or fundamental principle
                    7-9 = Important examples, core ideas, or detailed explanations
                    4-6 = Supporting information, context, or basic examples
                    1-3 = Repetitive information, tangents, or very basic content

                    Segments to analyze:
                    {json.dumps(batch_texts)}

                    Respond in JSON format with an array of ratings:
                    {{
                        "ratings": [
                            {{
                                "importance_score": <number 1-10>,
                                "reason": "Brief explanation"
                            }},
                            ...
                        ]
                    }}"""

                    importance_response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are an expert at analyzing educational content importance. Rate content based on its educational value and relevance."
                            },
                            {
                                "role": "user",
                                "content": importance_prompt
                            }
                        ],
                        temperature=0.3,
                        max_tokens=500,
                        response_format={ "type": "json_object" }
                    )

                    analysis = json.loads(importance_response.choices[0].message.content)
                    
                    for seg, rating in zip(batch, analysis["ratings"]):
                        segment_duration = seg["end"] - seg["start"]
                        total_duration += segment_duration
                        
                        importance_score = max(1, min(10, float(rating["importance_score"])))
                        can_skip = importance_score < 4
                        
                        if can_skip:
                            skippable_duration += segment_duration
                        
                        analyzed_segments.append({
                            "start": seg["start"],
                            "end": seg["end"],
                            "text": seg["text"],
                            "can_skip": can_skip,
                            "importance_score": importance_score,
                            "reason": rating["reason"]
                        })
                    
                    log(f"Processed batch {i//batch_size + 1}/{(len(segments) + batch_size - 1)//batch_size}")

                except Exception as e:
                    log(f"Error analyzing batch starting at segment {i}: {str(e)}", "WARNING")
                    # Handle failed batch with default values
                    for seg in batch:
                        analyzed_segments.append({
                            "start": seg["start"],
                            "end": seg["end"],
                            "text": seg["text"],
                            "can_skip": False,
                            "importance_score": 5,
                            "reason": "Error in batch analysis"
                        })
                        total_duration += seg["end"] - seg["start"]

            # Calculate statistics
            skippable_segments = [s for s in analyzed_segments if s["can_skip"]]
            skippable_percentage = (skippable_duration/total_duration)*100 if total_duration > 0 else 0
            
            log(f"Analysis complete: {len(skippable_segments)} skippable segments found")
            log(f"Potentially skippable content: {skippable_percentage:.1f}% of video")

            # Return the final result with both educational content and skippable segments
            return {
                "status": "success",
                "summary": content_data["summary"],
                "keyPoints": content_data["keyPoints"][:5],
                "flashcards": content_data["flashcards"][:5],
                "transcript": transcript,
                "segments": analyzed_segments,
                "stats": {
                    "total_segments": len(segments),
                    "skippable_segments": len(skippable_segments),
                    "total_duration": total_duration,
                    "skippable_duration": skippable_duration,
                    "skippable_percentage": skippable_percentage
                }
            }

        except Exception as e:
            log(f"Error during video processing: {str(e)}", "ERROR")
            log(f"Stack trace: {traceback.format_exc()}", "ERROR")
            raise

    except Exception as e:
        log(f"Error in process_video: {str(e)}", "ERROR")
        log(f"Stack trace: {traceback.format_exc()}", "ERROR")
        return {
            "status": "error",
            "error": str(e),
            "summary": "",
            "keyPoints": [],
            "flashcards": [],
            "transcript": "",
            "segments": [],
            "stats": {
                "total_segments": 0,
                "skippable_segments": 0,
                "total_duration": 0,
                "skippable_duration": 0,
                "skippable_percentage": 0
            }
        }
    finally:
        # Clean up all temporary files
        for temp_file in temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
                    log(f"Cleaned up temporary file: {temp_file}")
            except Exception as e:
                log(f"Error cleaning up {temp_file}: {str(e)}", "WARNING")


if __name__ == "__main__":
    app.serve() 