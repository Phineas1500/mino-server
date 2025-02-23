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
import numpy as np
import soundfile as sf

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

def cut_video_segments(video_path, segments, summary_data, min_segment_duration=1.0, max_pause_duration=0.2):
    """Cut video based on importance scores with smooth transitions and pause removal."""
    try:
        from moviepy.editor import VideoFileClip, concatenate_videoclips
        
        log("Loading video file...")
        video = VideoFileClip(str(video_path), audio=True)
        video_duration = video.duration
        log(f"Video duration: {video_duration:.2f}s")
        
        # First, detect and remove pauses
        log("Detecting pauses in audio...")
        temp_audio = str(Path(video_path).with_suffix('.wav'))
        video.audio.write_audiofile(
            temp_audio,
            codec='pcm_s16le',
            fps=16000,
            ffmpeg_params=["-ac", "1"],
            verbose=False,
            logger=None
        )
        
        # Read audio and detect silent segments
        audio_data, sample_rate = sf.read(temp_audio)
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
            
        # Calculate frame energy in dB
        frame_length = int(0.02 * sample_rate)  # 20ms frames
        hop_length = frame_length // 2
        frames = []
        
        for i in range(0, len(audio_data) - frame_length, hop_length):
            frame = audio_data[i:i + frame_length]
            energy = 20 * np.log10(np.mean(frame ** 2) + 1e-10)
            frames.append(energy)
            
        frames = np.array(frames)
        threshold_db = -40  # Threshold for silence detection
        is_silent = frames < threshold_db
        
        # Convert frame indices to time
        time_per_frame = hop_length / sample_rate
        silent_segments = []
        start_frame = None
        
        # Find continuous silent segments
        for i in range(len(is_silent)):
            if is_silent[i] and start_frame is None:
                start_frame = i
            elif not is_silent[i] and start_frame is not None:
                duration = (i - start_frame) * time_per_frame
                if duration >= max_pause_duration:
                    start_time = start_frame * time_per_frame
                    end_time = i * time_per_frame
                    silent_segments.append((start_time, end_time))
                start_frame = None
        
        # Check final segment
        if start_frame is not None:
            duration = (len(is_silent) - start_frame) * time_per_frame
            if duration >= max_pause_duration:
                start_time = start_frame * time_per_frame
                end_time = len(is_silent) * time_per_frame
                silent_segments.append((start_time, end_time))
        
        # Clean up audio file
        Path(temp_audio).unlink()
        
        # Create segments without pauses
        log(f"Found {len(silent_segments)} pauses to remove")
        active_segments = []
        current_time = 0
        
        for pause_start, pause_end in silent_segments:
            if pause_start > current_time:
                active_segments.append({
                    "start": current_time,
                    "end": pause_start
                })
            current_time = pause_end
        
        # Add final segment if needed
        if current_time < video_duration:
            active_segments.append({
                "start": current_time,
                "end": video_duration
            })
        
        # Get OpenAI client for importance analysis
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
                continue
            segment["end"] = min(segment["end"], video_duration)
            cleaned_segments.append(segment)
        
        analyzed_segments = cleaned_segments
        
        # Process each active segment with appropriate speed
        clips_to_keep = []
        total_original_duration = 0
        total_kept_duration = 0
        prev_clip = None
        
        for active_seg in active_segments:
            # Find analyzed segments that overlap with this active segment
            relevant_segments = [
                seg for seg in analyzed_segments
                if seg["end"] > active_seg["start"] and seg["start"] < active_seg["end"]
            ]
            
            if not relevant_segments:
                continue
                
            # Use average importance of overlapping segments
            avg_importance = sum(float(s["importance_score"]) for s in relevant_segments) / len(relevant_segments)
            duration = active_seg["end"] - active_seg["start"]
            total_original_duration += duration
            
            try:
                clip = video.subclip(active_seg["start"], active_seg["end"])
                
                # Apply speed adjustment based on importance
                if avg_importance >= 0.8:  # Key content
                    speed_factor = 1.0
                elif avg_importance >= 0.6:
                    speed_factor = 1.25
                elif avg_importance >= 0.4:
                    speed_factor = 1.75
                elif avg_importance >= 0.2:
                    speed_factor = 2.25
                else:
                    continue  # Skip very low importance segments
                
                try:
                    clip = clip.speedx(speed_factor)
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
                log(f"Error processing segment {active_seg['start']:.2f}s - {active_seg['end']:.2f}s: {str(e)}", "ERROR")
                continue
        
        # Log statistics
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
        
        # Concatenate clips
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
            for clip in clips_to_keep:
                clip.close()
            
        except Exception as e:
            log(f"Error during concatenation: {str(e)}", "ERROR")
            output_path = str(video_path)
        
        video.close()
        return output_path
        
    except Exception as e:
        log(f"Error in cut_video_segments: {str(e)}", "ERROR")
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
                        
                        # Calculate playback speed based on importance score
                        # Linear scale: 10 -> 1x speed, 1 -> 2.5x speed
                        playback_speed = 1.0 + (1.5 * (10 - importance_score) / 9)
                        playback_speed = round(playback_speed, 2)
                        
                        # Calculate adjusted duration with speed
                        adjusted_duration = segment_duration / playback_speed
                        
                        if can_skip:
                            skippable_duration += segment_duration
                        
                        analyzed_segments.append({
                            "start": seg["start"],
                            "end": seg["end"],
                            "text": seg["text"],
                            "can_skip": can_skip,
                            "importance_score": importance_score,
                            "playback_speed": playback_speed,
                            "original_duration": segment_duration,
                            "adjusted_duration": adjusted_duration,
                            "reason": rating["reason"]
                        })
                    
                    log(f"Processed batch {i//batch_size + 1}/{(len(segments) + batch_size - 1)//batch_size}")

                except Exception as e:
                    log(f"Error analyzing batch starting at segment {i}: {str(e)}", "WARNING")
                    # Handle failed batch with default values
                    for seg in batch:
                        segment_duration = seg["end"] - seg["start"]
                        analyzed_segments.append({
                            "start": seg["start"],
                            "end": seg["end"],
                            "text": seg["text"],
                            "can_skip": False,
                            "importance_score": 5,
                            "playback_speed": 1.75,  # Middle speed for default
                            "original_duration": segment_duration,
                            "adjusted_duration": segment_duration / 1.75,
                            "reason": "Error in batch analysis"
                        })
                        total_duration += segment_duration

            # Calculate statistics
            skippable_segments = [s for s in analyzed_segments if s["can_skip"]]
            total_adjusted_duration = sum(s["adjusted_duration"] for s in analyzed_segments)
            time_saved = total_duration - total_adjusted_duration
            skippable_percentage = (skippable_duration/total_duration)*100 if total_duration > 0 else 0
            time_saved_percentage = (time_saved/total_duration)*100 if total_duration > 0 else 0
            
            log(f"Analysis complete: {len(skippable_segments)} skippable segments found")
            log(f"Original duration: {total_duration:.1f}s, Adjusted duration: {total_adjusted_duration:.1f}s")
            log(f"Time saved through speed adjustments: {time_saved:.1f}s ({time_saved_percentage:.1f}%)")
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