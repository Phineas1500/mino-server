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
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

def fix_overlapping_segments(segments):
    """Fix any overlapping segments by adjusting end times"""
    if not segments:
        return []
        
    # Sort segments by start time
    sorted_segments = sorted(segments, key=lambda x: x["start"])
    
    # Fix overlaps
    fixed_segments = [sorted_segments[0]]
    for current in sorted_segments[1:]:
        previous = fixed_segments[-1]
        
        # Check for overlap
        if current["start"] < previous["end"]:
            # Adjust the previous segment's end time
            overlap = previous["end"] - current["start"]
            previous["end"] = current["start"]
            
            # Adjust original and adjusted durations if they exist
            if "original_duration" in previous:
                previous["original_duration"] -= overlap
            if "adjusted_duration" in previous:
                # Scale down adjusted duration proportionally
                if previous["original_duration"] > 0:
                    ratio = previous["adjusted_duration"] / (previous["original_duration"] + overlap)
                    previous["adjusted_duration"] = previous["original_duration"] * ratio
        
        fixed_segments.append(current)
    
    return fixed_segments

def cleanup_segments(segments, min_duration=0.5):
    """Remove segments that are too short and ensure all values are valid"""
    if not segments:
        return []
        
    cleaned = []
    for segment in segments:
        # Skip segments that are too short
        if segment["end"] - segment["start"] < min_duration:
            continue
            
        # Ensure all required fields exist and have valid values
        cleaned_segment = {
            "start": float(segment["start"]),
            "end": float(segment["end"]),
            "text": str(segment.get("text", "")),
            "importance_score": float(segment.get("importance_score", 5)),
        }
        
        # Add optional fields if they exist
        if "can_skip" in segment:
            cleaned_segment["can_skip"] = bool(segment["can_skip"])
        if "playback_speed" in segment:
            cleaned_segment["playback_speed"] = max(1.0, min(2.5, float(segment.get("playback_speed", 1.5))))
        if "reason" in segment:
            cleaned_segment["reason"] = str(segment["reason"])
        if "original_duration" in segment:
            cleaned_segment["original_duration"] = float(segment["original_duration"])
        if "adjusted_duration" in segment:
            cleaned_segment["adjusted_duration"] = float(segment["adjusted_duration"])
        if "speed" in segment:
            cleaned_segment["speed"] = float(segment["speed"])
            
        cleaned.append(cleaned_segment)
    
    return cleaned

def analyze_segments_importance(segments, client, summary_data=None):
    """Legacy function signature for compatibility - delegates to the new parallel method"""
    log("Using parallel processing for segment importance analysis")
    loop = asyncio.get_event_loop()
    analyzed_segments, _ = loop.run_until_complete(process_segments_parallel(segments, client))
    return analyzed_segments

async def process_batch_async(client, batch, batch_idx, total_batches):
    """Process a batch of segments asynchronously with OpenAI API"""
    try:
        batch_texts = [
            f"{i+1}. [{seg['start']:.1f}s - {seg['end']:.1f}s] Duration: {seg['end']-seg['start']:.1f}s\nContent: {seg['text']}"
            for i, seg in enumerate(batch)
        ]
        log(f"Processing batch {batch_idx+1}/{total_batches} with {len(batch)} segments")
        
        segments_text = "\n\n".join(batch_texts)
        
        importance_prompt = f"""Rate these lecture segments and return the analysis in JSON format.

        RATING RULES:
        - 10-8: Essential content (core concepts, definitions, key principles)
        - 7-5: Supporting content (examples, explanations, context)
        - 4-1: Supplementary content (repetition, tangents, filler)

        SEGMENTS TO ANALYZE:
        {segments_text}

        REQUIRED JSON RESPONSE FORMAT:
        {{
            "ratings": [
                {{
                    "score": number,          // 1-10 rating
                    "speed": number,          // 1.0-2.5 playback speed
                    "skip": boolean,          // true if score ≤ 3
                    "key_point": "string"     // Brief summary
                }}
            ]
        }}"""

        def execute_openai_call():
            return client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{
                    "role": "system",
                    "content": "You analyze educational content and return ratings in JSON format. Always respond with valid JSON matching the requested structure."
                }, {
                    "role": "user",
                    "content": importance_prompt
                }],
                temperature=0.3,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
        
        with ThreadPoolExecutor() as executor:
            response = await asyncio.get_event_loop().run_in_executor(
                executor, execute_openai_call
            )
        
        # Parse JSON response
        result = json.loads(response.choices[0].message.content)
        ratings = result.get('ratings', [])
        
        processed_segments = []
        batch_duration = 0
        batch_skippable = 0
        
        for seg, rating in zip(batch, ratings[:len(batch)]):
            try:
                segment_duration = seg["end"] - seg["start"]
                batch_duration += segment_duration
                
                importance_score = float(rating.get('score', 5))
                importance_score = max(1, min(10, importance_score))
                
                playback_speed = float(rating.get('speed', 1.5))
                playback_speed = max(1.0, min(2.5, playback_speed))
                
                can_skip = bool(rating.get('skip', importance_score <= 3))
                
                if can_skip:
                    batch_skippable += segment_duration
                
                processed_segments.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                    "can_skip": can_skip,
                    "importance_score": importance_score,
                    "playback_speed": playback_speed,
                    "original_duration": segment_duration,
                    "adjusted_duration": segment_duration / playback_speed,
                    "reason": rating.get('key_point', '')
                })
                
            except Exception as e:
                log(f"Error processing segment: {str(e)}", "WARNING")
                # Add default values for failed segments
                segment_duration = seg["end"] - seg["start"]
                processed_segments.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                    "can_skip": False,
                    "importance_score": 5,
                    "playback_speed": 1.5,
                    "original_duration": segment_duration,
                    "adjusted_duration": segment_duration / 1.5,
                    "reason": f"Error processing segment: {str(e)[:50]}"
                })
        
        return {
            "segments": processed_segments,
            "total_duration": batch_duration,
            "skippable_duration": batch_skippable
        }
        
    except Exception as e:
        log(f"Error in batch {batch_idx+1}: {str(e)}", "ERROR")
        return {
            "segments": [{
                **seg,
                "can_skip": False,
                "importance_score": 5,
                "playback_speed": 1.5,
                "original_duration": seg["end"] - seg["start"],
                "adjusted_duration": (seg["end"] - seg["start"]) / 1.5,
                "reason": f"Batch error: {str(e)[:50]}"
            } for seg in batch],
            "total_duration": sum(seg["end"] - seg["start"] for seg in batch),
            "skippable_duration": 0
        }

async def process_segments_parallel(segments, client, batch_size=40, max_concurrent=3):
    """Process segments in parallel with larger batches and controlled concurrency"""
    log(f"Starting parallel processing of {len(segments)} segments")
    
    # Merge segments based on more aggressive criteria
    merged_segments = []
    current_segment = None
    MIN_SEGMENT_DURATION = 5.0  # Minimum desired segment duration in seconds
    MAX_SEGMENT_DURATION = 15.0  # Maximum segment duration to prevent too-long segments
    
    for segment in segments:
        if not current_segment:
            current_segment = segment.copy()
            continue
        
        current_duration = current_segment["end"] - current_segment["start"]
        segment_duration = segment["end"] - segment["start"]
        combined_duration = current_duration + segment_duration
        
        # Merge criteria:
        # 1. Current segment is too short OR
        # 2. Combined duration is less than maximum AND
        #    - There's semantic continuity (no long pause) OR
        #    - Current segment is very short
        time_gap = segment["start"] - current_segment["end"]
        should_merge = (
            current_duration < MIN_SEGMENT_DURATION or
            (combined_duration < MAX_SEGMENT_DURATION and
             (time_gap < 1.0 or current_duration < 2.0))
        )
        
        if should_merge:
            # Merge segments
            current_segment["end"] = segment["end"]
            current_segment["text"] += " " + segment["text"]
        else:
            # Add current segment and start a new one
            merged_segments.append(current_segment)
            current_segment = segment.copy()
    
    # Add the last segment
    if current_segment:
        merged_segments.append(current_segment)
    
    # Log merge results
    original_count = len(segments)
    merged_count = len(merged_segments)
    average_duration_before = sum(s["end"] - s["start"] for s in segments) / original_count
    average_duration_after = sum(s["end"] - s["start"] for s in merged_segments) / merged_count
    
    log(f"Merged {original_count} segments into {merged_count} segments")
    log(f"Average segment duration: {average_duration_before:.1f}s → {average_duration_after:.1f}s")
    
    # Adjust batch size based on segment count
    dynamic_batch_size = min(max(10, merged_count // 4), batch_size)
    batches = [merged_segments[i:i+dynamic_batch_size] for i in range(0, len(merged_segments), dynamic_batch_size)]
    
    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(batch, idx):
        async with semaphore:
            # Add exponential backoff between batches to prevent rate limiting
            if idx > 0:
                delay = min(0.2 * (2 ** (idx // 3)), 2.0)  # Cap at 2 seconds
                await asyncio.sleep(delay)
            return await process_batch_async(client, batch, idx, len(batches))
    
    # Process batches
    tasks = [process_with_semaphore(batch, i) for i, batch in enumerate(batches)]
    batch_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Combine results
    all_segments = []
    total_duration = 0
    skippable_duration = 0
    
    for result in batch_results:
        if isinstance(result, Exception):
            log(f"Batch failed: {str(result)}", "ERROR")
            continue
            
        all_segments.extend(result["segments"])
        total_duration += result["total_duration"]
        skippable_duration += result["skippable_duration"]
    
    # Calculate statistics
    total_adjusted_duration = sum(seg["adjusted_duration"] for seg in all_segments)
    skippable_segments = [s for s in all_segments if s["can_skip"]]
    
    # Safely calculate percentages, avoiding division by zero
    time_saved = total_duration - total_adjusted_duration if total_duration > 0 else 0
    skippable_percentage = (skippable_duration/total_duration)*100 if total_duration > 0 else 0
    time_saved_percentage = (time_saved/total_duration)*100 if total_duration > 0 else 0
    
    if total_duration > 0:
        log(f"Processing complete: {len(all_segments)} segments analyzed")
        log(f"Original: {total_duration:.1f}s, Adjusted: {total_adjusted_duration:.1f}s")
        log(f"Time saved: {time_saved:.1f}s ({time_saved_percentage:.1f}%)")
    else:
        log("Warning: No valid segments with duration found")
    
    return all_segments, {
        "total_segments": len(all_segments),
        "skippable_segments": len(skippable_segments),
        "total_duration": total_duration,
        "skippable_duration": skippable_duration,
        "skippable_percentage": skippable_percentage,
        "time_saved": time_saved,
        "time_saved_percentage": time_saved_percentage
    }

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
    """Analyze transcript segments in parallel and determine optimal playback speeds"""
    try:
        log("Starting playback speed optimization with parallel batch processing...")
        
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        
        # Define a specialized version of process_batch_async for this function
        async def process_speed_batch(batch, idx, total):
            try:
                batch_texts = []
                for j, segment in enumerate(batch):
                    batch_texts.append(f"Segment {j+1}: \"{segment['text']}\"")
                
                segments_text = "\n\n".join(batch_texts)
                
                prompt = f"""Analyze these {len(batch)} educational segments and return JSON ratings for their importance.

                RATING CRITERIA:
                - 10: Core concept that defines the entire topic
                - 8-9: Critical information necessary for understanding
                - 6-7: Important examples or detailed explanations
                - 4-5: Supporting context or secondary information
                - 2-3: Basic examples or repetitive content
                - 1: Fillers, tangential remarks, or redundancies

                SEGMENTS TO RATE:
                {segments_text}

                REQUIRED JSON RESPONSE FORMAT:
                {{
                    "ratings": [
                        {{ "importance_score": number, "reason": "3-5 word justification" }},
                        ...exactly {len(batch)} ratings in original order...
                    ]
                }}"""
                
                # Run OpenAI call in a thread pool
                def execute_openai_call():
                    return client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {
                                "role": "system",
                                "content": "You analyze educational content and return ratings in JSON format. Always respond with valid JSON matching the requested structure."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        temperature=0.3,
                        max_tokens=1500,
                        response_format={"type": "json_object"}
                    )
                
                with ThreadPoolExecutor() as executor:
                    response = await asyncio.get_event_loop().run_in_executor(
                        executor, execute_openai_call
                    )
                
                # Parse response
                analysis_result = json.loads(response.choices[0].message.content)
                
                # Validate response
                if "ratings" not in analysis_result or not isinstance(analysis_result["ratings"], list):
                    raise ValueError("Invalid response format: missing 'ratings' array")
                
                if len(analysis_result["ratings"]) != len(batch):
                    log(f"Warning: Got {len(analysis_result['ratings'])} ratings for {len(batch)} segments in batch {idx+1}", "WARNING")
                    
                    # Adjust ratings array size if needed
                    if len(analysis_result["ratings"]) < len(batch):
                        analysis_result["ratings"].extend([
                            {
                                "importance_score": 5,
                                "reason": "Default rating (missing in API response)"
                            } for _ in range(len(batch) - len(analysis_result["ratings"]))
                        ])
                    else:
                        analysis_result["ratings"] = analysis_result["ratings"][:len(batch)]
                
                # Process each segment with its corresponding rating
                optimized_batch = []
                batch_duration = 0
                batch_optimized_duration = 0
                
                for segment, rating in zip(batch, analysis_result["ratings"]):
                    segment_duration = segment["end"] - segment["start"]
                    batch_duration += segment_duration
                    
                    # Validate importance score
                    importance_score = float(rating.get("importance_score", 5))
                    importance_score = max(1, min(10, importance_score))
                    
                    # Calculate playback speed
                    speed = 1.0 + ((10 - importance_score) / 9)  # Linear scaling
                    speed = round(min(max(speed, 1.0), 2.0), 2)
                    
                    # Calculate optimized duration
                    optimized_duration = segment_duration / speed
                    batch_optimized_duration += optimized_duration
                    
                    # Add optimized segment
                    optimized_batch.append({
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": segment["text"],
                        "speed": speed,
                        "importance_score": importance_score,
                        "reason": rating.get("reason", "")
                    })
                
                log(f"Successfully processed batch {idx+1}/{total}")
                
                return {
                    "segments": optimized_batch,
                    "original_duration": batch_duration,
                    "optimized_duration": batch_optimized_duration
                }
                
            except Exception as e:
                log(f"Error processing batch {idx+1}: {str(e)}", "WARNING")
                
                # Return default values
                default_segments = []
                batch_duration = 0
                
                for segment in batch:
                    segment_duration = segment["end"] - segment["start"]
                    batch_duration += segment_duration
                    
                    default_segments.append({
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": segment["text"],
                        "speed": 1.0,
                        "importance_score": 5,
                        "reason": f"Batch processing error: {str(e)[:50]}"
                    })
                
                return {
                    "segments": default_segments,
                    "original_duration": batch_duration,
                    "optimized_duration": batch_duration  # No optimization applied
                }
        
        # Split segments into batches
        BATCH_SIZE = 20
        MAX_CONCURRENT = 5  # Maximum number of concurrent API calls
        
        batches = []
        for i in range(0, len(segments), BATCH_SIZE):
            batches.append(segments[i:i+BATCH_SIZE])
        
        total_batches = len(batches)
        log(f"Processing {len(segments)} segments in {total_batches} batches with {MAX_CONCURRENT} concurrent requests")
        
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        
        async def process_with_semaphore(batch, idx):
            async with semaphore:
                # Add a small delay to prevent rate limiting
                if idx > 0:
                    await asyncio.sleep(0.5)
                return await process_speed_batch(batch, idx, total_batches)
        
        # Create tasks for all batches
        tasks = [process_with_semaphore(batch, i) for i, batch in enumerate(batches)]
        
        # Execute all tasks and gather results
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        optimized_segments = []
        total_duration = 0
        optimized_duration = 0
        
        for result in batch_results:
            if isinstance(result, Exception):
                log(f"A batch failed completely: {str(result)}", "ERROR")
                continue
                
            optimized_segments.extend(result["segments"])
            total_duration += result["original_duration"]
            optimized_duration += result["optimized_duration"]
        
        # Calculate statistics
        time_saved = total_duration - optimized_duration
        time_saved_percentage = (time_saved / total_duration) * 100 if total_duration > 0 else 0
        
        log(f"Optimization complete: Processed {len(optimized_segments)} segments")
        log(f"Original duration: {total_duration:.2f}s, Optimized duration: {optimized_duration:.2f}s")
        log(f"Time saved: {time_saved:.2f}s ({time_saved_percentage:.1f}%)")
        
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
        log(f"Stack trace: {traceback.format_exc()}", "ERROR")
        
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

            # Transcribe with Whisper using supported options
            log("Loading Whisper model...")
            model = whisper.load_model("tiny")
            
            log("Starting transcription...")
            result = model.transcribe(
                str(temp_audio_path),
                language='en',
                verbose=True,
                condition_on_previous_text=True,  # Help maintain context between segments
                word_timestamps=False,  # Faster processing
                # Using only supported options
                fp16=False,  # Avoid GPU memory issues
                beam_size=1,  # Faster processing
                best_of=1,    # Faster processing
                temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],  # Multiple temperatures for better results
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

            # More focused prompt for larger segments
            content_prompt = f"""Extract the core educational value from this transcript and return as JSON.

            ANALYSIS REQUIREMENTS:
            1. Focus on identifying the central thesis and major supporting arguments
            2. Prioritize conceptual understanding over details
            3. Extract information that would appear on an exam
            4. Ignore repetitions, examples, and tangents
            5. Connect related ideas across different parts of the transcript

            REQUIRED JSON FORMAT:
            {{
                "summary": "Clear, concise 2-paragraph summary of core concepts only",
                "keyPoints": [
                    "5 essential insights that represent the most important takeaways",
                    ...4 more key points...
                ],
                "flashcards": [
                    {{
                        "question": "Conceptual question testing understanding",
                        "answer": "Precise, factual answer focusing on core concept"
                    }},
                    ...4 more flashcards covering different concepts...
                ]
            }}

            TRANSCRIPT:
            {transcript[:4000]}"""  # Limit transcript length to avoid token limits

            log("Generating educational content with OpenAI...")
            content_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing educational content and returning results in JSON format. Always respond with valid JSON that matches the requested structure."
                    },
                    {
                        "role": "user",
                        "content": content_prompt
                    }
                ],
                temperature=0.5,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )

            # Parse OpenAI response with error handling
            try:
                content_data = json.loads(content_response.choices[0].message.content)
                required_fields = ["summary", "keyPoints", "flashcards"]
                missing_fields = [field for field in required_fields if field not in content_data]
                
                if missing_fields:
                    raise ValueError(f"Missing required fields in API response: {missing_fields}")
                
                if not isinstance(content_data["keyPoints"], list) or len(content_data["keyPoints"]) == 0:
                    raise ValueError("Invalid or empty keyPoints in response")
                    
                if not isinstance(content_data["flashcards"], list) or len(content_data["flashcards"]) == 0:
                    raise ValueError("Invalid or empty flashcards in response")
                    
            except (json.JSONDecodeError, KeyError, IndexError, ValueError) as e:
                log(f"Error parsing OpenAI response: {str(e)}", "ERROR")
                content_data = {
                    "summary": "Error generating summary",
                    "keyPoints": ["Error generating key points"],
                    "flashcards": [{"question": "Error", "answer": "Error generating flashcards"}]
                }
                
            log("Successfully generated educational content")

            # Process segments using parallel processing
            log("Analyzing segments using parallel processing...")
            try:
                analyzed_segments, stats = await process_segments_parallel(
                    segments, 
                    client,
                    batch_size=20,      # Number of segments per batch
                    max_concurrent=5    # Maximum number of concurrent API calls
                )
            except Exception as e:
                log(f"Error in segment analysis: {str(e)}", "ERROR")
                # Return basic segment info if analysis fails
                analyzed_segments = [{
                    "start": s["start"],
                    "end": s["end"],
                    "text": s["text"],
                    "importance_score": 5,
                    "playback_speed": 1.0,
                    "can_skip": False,
                    "reason": "Error in analysis"
                } for s in segments]
                stats = {
                    "total_segments": len(segments),
                    "skippable_segments": 0,
                    "total_duration": sum(s["end"] - s["start"] for s in segments),
                    "skippable_duration": 0,
                    "skippable_percentage": 0
                }

            # Return the final result with educational content and analyzed segments
            return {
                "status": "success",
                "summary": content_data["summary"],
                "keyPoints": content_data["keyPoints"][:5],
                "flashcards": content_data["flashcards"][:5],
                "transcript": transcript,
                "segments": analyzed_segments,
                "stats": stats
            }

        except Exception as api_error:
            log(f"API Error: {str(api_error)}", "ERROR")
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