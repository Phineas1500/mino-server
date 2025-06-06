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
import sys # Import sys for flushing output

# --- 1. Define Progress Reporting Function ---
def report_progress(percentage: int, stage: str, message: str = None):
    """Prints progress updates to stdout for the calling process."""
    log_msg = f"PROGRESS: {percentage} STAGE: {stage}"
    if message:
        log_msg += f" MESSAGE: {message}"
    print(log_msg, flush=True) # Use flush=True
    sys.stdout.flush() # Explicit flush

def calculate_optimal_flashcard_count(segments, transcript):
    """Calculate optimal number of flashcards based on video length and content richness"""
    # Calculate total video duration
    total_duration = sum(seg["end"] - seg["start"] for seg in segments) if segments else 0
    
    # Calculate content metrics
    transcript_length = len(transcript) if transcript else 0
    segment_count = len(segments) if segments else 0
    
    # Base number of flashcards (minimum)
    base_count = 3
    
    # Add flashcards based on duration (1 per 2 minutes)
    duration_bonus = int(total_duration / 120)  # Every 2 minutes
    
    # Add flashcards based on transcript length (1 per 800 characters)
    content_bonus = int(transcript_length / 800)
    
    # Add flashcards based on segment density (indicates information richness)
    # More segments per minute = more dense content
    if total_duration > 0:
        segments_per_minute = (segment_count / total_duration) * 60
        density_bonus = int(segments_per_minute / 3)  # Every 3 segments per minute
    else:
        density_bonus = 0
    
    # Calculate total with bonuses
    total_flashcards = base_count + duration_bonus + content_bonus + density_bonus
    
    # Cap between 3 and 20 flashcards
    optimal_count = max(3, min(20, total_flashcards))
    
    log(f"Flashcard calculation: duration={total_duration:.1f}s, transcript_len={transcript_length}, segments={segment_count}")
    log(f"Bonuses: duration_bonus={duration_bonus}, content_bonus={content_bonus}, density_bonus={density_bonus}")
    log(f"Optimal flashcard count: {optimal_count}")
    
    return optimal_count

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

async def process_batch_async(client, batch, batch_idx, total_batches, key_points): # Added video_type, video_title, video_description
    """Process a batch of segments asynchronously with OpenAI API (Original structure with updated prompts)"""
    try:
        batch_texts = [
            f"{i+1}. [{seg['start']:.1f}s - {seg['end']:.1f}s] Duration: {seg['end']-seg['start']:.1f}s\nContent: {seg['text']}"
            for i, seg in enumerate(batch)
        ]
        log(f"Processing batch {batch_idx+1}/{total_batches} with {len(batch)} segments")

        segments_text = "\n\n".join(batch_texts)
        key_points_text = "\n".join([f"- {kp}" for kp in key_points]) # Format key points

        # --- NEW Enhanced Prompt ---
        importance_prompt = f"""Analyze these video segments based on their importance to the viewer. Return the analysis in JSON format.

        OVERALL KEY POINTS/THEMES:
        {key_points_text}

        RATING GUIDELINES FOR VIDEO'S CONTENT (Consider relevance to the Key Points/Themes above, as well as what you determine about the video's type):
        - 10-8: Essential content - The main value viewers are seeking based on the video type.
          * For educational: Core concepts, key definitions, primary explanations
          * For entertainment: Major punchlines, key plot developments, highlight moments
          * For tutorials: Critical steps, important demonstrations, essential techniques
          * For interviews: Major revelations, key insights, defining statements
          → 1.0-1.1x speed (Approx range for calculation guidance)

        - 7-5: Supporting content - Enhances understanding or enjoyment but not the primary value.
          * For educational: Supporting examples, context, secondary details
          * For entertainment: Setup, context building, secondary moments
          * For tutorials: Preparatory steps, additional context, alternative approaches
          * For interviews: Context, follow-up questions, elaborations
          → 1.3-1.7x speed (Approx range for calculation guidance)

        - 4-1: Supplementary content - Minimal essential information, potentially distracting.
          * For educational: Repetition, tangents, filler content, long pauses
          * For entertainment: Excessive pauses, minor transitions, redundant elements
          * For tutorials: Verbose explanations, non-critical asides, repetitive warnings
          * For interviews: Small talk, extended introductions, tangential discussions
          → 1.9-2.5x speed or potentially skippable if ≤ 3 (Approx range for calculation guidance)

        PLAYBACK SPEED FORMULA (Strict - MUST be followed in JSON output):
        - Score 10: Speed 1.0
        - Score 9: Speed 1.0
        - Score 8: Speed 1.1
        - Score 7: Speed 1.3
        - Score 6: Speed 1.5
        - Score 5: Speed 1.7
        - Score 4: Speed 1.9
        - Score 3: Speed 2.1
        - Score 2: Speed 2.3
        - Score 1: Speed 2.5

        SEGMENTS TO ANALYZE:
        {segments_text}

        REQUIRED JSON RESPONSE FORMAT:
        {{
            "ratings": [
                {{
                    "score": number,          // 1-10 rating based on importance relative to this specific video's purpose
                    "speed": number,          // MUST follow the strict formula above based on score
                    "skip": boolean,          // true if score ≤ 3
                    "key_point": "string"     // Brief summary justifying the rating based on the segment content and video context
                }}
            ]
        }}"""
        # --- End NEW Enhanced Prompt ---

        def execute_openai_call():
            return client.chat.completions.create(
                model="gpt-4.1-nano-2025-04-14", # Or your preferred model
                messages=[{
                    "role": "system",
                    # --- NEW Enhanced System Prompt ---
                    "content": "You are an expert video content analyst. Your task is to analyze video segments based on the overall key points/themes. Rate each segment's importance relative to the *specific purpose* of *this* video, not against an absolute standard. Adhere strictly to the provided rating guidelines and the specific score-to-speed mapping formula. Respond ONLY with valid JSON matching the requested structure. The 'key_point' field in the JSON should briefly explain *why* the segment received its score in the context of the video's goals."
                    # --- End NEW Enhanced System Prompt ---
                }, {
                    "role": "user",
                    "content": importance_prompt
                }],
                temperature=0.3,
                max_tokens=2000, # Adjust if needed based on batch size
                response_format={"type": "json_object"}
            )

        with ThreadPoolExecutor() as executor:
            response = await asyncio.get_event_loop().run_in_executor(
                executor, execute_openai_call
            )

        # Parse JSON response
        result = json.loads(response.choices[0].message.content)
        ratings = result.get('ratings', [])

        # --- Logging and Mismatch Handling (Kept from original) ---
        log(f"[Batch {batch_idx+1}] Received {len(ratings)} ratings for {len(batch)} segments.")
        if len(ratings) != len(batch):
            log(f"[Batch {batch_idx+1}] Warning: Rating count mismatch. Will adjust.", "WARNING")
            # Adjust ratings array size if needed
            if len(ratings) < len(batch):
                ratings.extend([
                    {"score": 5, "speed": 1.7, "skip": False, "key_point": "Default rating due to mismatch"}
                    for _ in range(len(batch) - len(ratings))
                ])
            else:
                ratings = ratings[:len(batch)]
        # --- End Logging and Mismatch Handling ---

        processed_segments = []
        batch_duration = 0
        batch_skippable = 0

        # --- Segment Processing Loop (Kept identical to original logic) ---
        for i, (seg, rating) in enumerate(zip(batch, ratings)): # Use enumerate for index
            try:
                segment_duration = seg["end"] - seg["start"]
                batch_duration += segment_duration

                log(f"[Batch {batch_idx+1}, Seg {i+1}] Original Segment: start={seg['start']:.1f}, end={seg['end']:.1f}, text='{seg['text'][:50]}...'")
                log(f"[Batch {batch_idx+1}, Seg {i+1}] Received Rating: {rating}")

                importance_score = 5 # Default score
                api_speed = 0.0 # Default speed indicator
                can_skip = False # Default skip status
                reason_text = "Default rating" # Default reason

                # Safely get rating components
                try:
                    importance_score = float(rating.get('score', 5))
                    importance_score = max(1, min(10, importance_score)) # Clamp score 1-10
                except (ValueError, TypeError):
                     log(f"[Batch {batch_idx+1}, Seg {i+1}] Warning: Invalid score type in rating ({rating.get('score')}). Using default score 5.", "WARNING")
                     importance_score = 5

                try:
                    api_speed = float(rating.get('speed', 0.0))
                except (ValueError, TypeError):
                     log(f"[Batch {batch_idx+1}, Seg {i+1}] Warning: Invalid speed type in rating ({rating.get('speed')}). Will calculate speed from score.", "WARNING")
                     api_speed = 0.0 # Indicate calculation needed

                try:
                     # Use 'skip' field if present, otherwise derive from score
                    if 'skip' in rating:
                         can_skip = bool(rating.get('skip'))
                    else:
                         can_skip = importance_score <= 3
                except Exception: # Catch potential errors converting to bool
                    log(f"[Batch {batch_idx+1}, Seg {i+1}] Warning: Invalid skip type in rating ({rating.get('skip')}). Deriving from score.", "WARNING")
                    can_skip = importance_score <= 3

                reason_text = rating.get('key_point', 'No reason provided')

                # Calculate playback speed based on score using the strict formula
                playback_speed_map = {
                    10: 1.0, 9: 1.0, 8: 1.1,
                    7: 1.3, 6: 1.5, 5: 1.7,
                    4: 1.9, 3: 2.1, 2: 2.3, 1: 2.5
                }
                default_speed = 1.5 # Fallback speed

                # Verify API speed against the formula for the score, or calculate if invalid/missing
                expected_speed = playback_speed_map.get(int(importance_score), default_speed)
                if 1.0 <= api_speed <= 2.5 and abs(api_speed - expected_speed) < 0.01: # Check if API speed is valid and matches formula
                     playback_speed = api_speed
                     log(f"[Batch {batch_idx+1}, Seg {i+1}] Using valid speed from API: {api_speed}", "DEBUG")
                else:
                    if api_speed != 0.0: # Log if API provided an invalid speed
                         log(f"[Batch {batch_idx+1}, Seg {i+1}] Warning: API speed ({api_speed}) is invalid or doesn't match score ({importance_score}). Calculating speed based on score.", "WARNING")
                    playback_speed = expected_speed # Calculate speed based on score
                    log(f"[Batch {batch_idx+1}, Seg {i+1}] Calculated speed based on score: {playback_speed}", "DEBUG")


                # Ensure speed is within bounds (redundant check, but safe)
                playback_speed = max(1.0, min(2.5, playback_speed))

                if can_skip:
                    batch_skippable += segment_duration

                processed_segment_data = {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"], # Ensure text is always carried over
                    "can_skip": can_skip,
                    "importance_score": importance_score,
                    "playback_speed": playback_speed,
                    "original_duration": segment_duration,
                    "adjusted_duration": segment_duration / playback_speed if playback_speed > 0 else segment_duration, # Avoid division by zero
                    "reason": reason_text # Use the reason from the API or default
                }
                processed_segments.append(processed_segment_data)

                log(f"[Batch {batch_idx+1}, Seg {i+1}] Processed Segment: score={importance_score}, speed={playback_speed}, skip={can_skip}, text='{processed_segment_data['text'][:50]}...'")
                if importance_score <= 1:
                    log(f"[Batch {batch_idx+1}, Seg {i+1}] Low score segment (<=1) processed. Text included: {'Yes' if processed_segment_data['text'] else 'No'}", "DEBUG")

            except Exception as e:
                log(f"Error processing segment {i+1} in batch {batch_idx+1}: {str(e)}", "WARNING")
                # Add default values for failed segments (Kept identical)
                segment_duration = seg.get("end", 0) - seg.get("start", 0)
                default_segment = {
                    "start": seg.get("start", 0),
                    "end": seg.get("end", 0),
                    "text": seg.get("text", "[Text missing due to segment error]"), # Ensure text is included
                    "can_skip": False,
                    "importance_score": 5,
                    "playback_speed": 1.5,
                    "original_duration": segment_duration,
                    "adjusted_duration": segment_duration / 1.5 if segment_duration > 0 else 0,
                    "reason": f"Error processing segment: {str(e)[:50]}"
                }
                processed_segments.append(default_segment)
                log(f"[Batch {batch_idx+1}, Seg {i+1}] Added default segment due to error. Text included: {'Yes' if default_segment['text'] else 'No'}", "WARNING")
        # --- End Segment Processing Loop ---

        log(f"[Batch {batch_idx+1}] Finished processing. Produced {len(processed_segments)} segments.")
        return {
            "segments": processed_segments,
            "total_duration": batch_duration,
            "skippable_duration": batch_skippable
        }

    except Exception as e:
        log(f"Error in batch {batch_idx+1}: {str(e)}", "ERROR")
        log(f"[Batch {batch_idx+1}] Returning default segments due to batch error. Input count: {len(batch)}")
        # Return default structure on batch error (Kept identical)
        default_segments = []
        batch_total_duration = 0
        for seg in batch:
             duration = seg.get("end", 0) - seg.get("start", 0)
             batch_total_duration += duration
             default_segments.append({
                 "start": seg.get("start", 0),
                 "end": seg.get("end", 0),
                 "text": seg.get("text", "[Text missing due to batch error]"), # Ensure text field exists
                 "can_skip": False,
                 "importance_score": 5,
                 "playback_speed": 1.5,
                 "original_duration": duration,
                 "adjusted_duration": duration / 1.5 if duration > 0 else 0,
                 "reason": f"Batch error: {str(e)[:50]}",
                 # Ensure all keys expected by later code are present, even in error case
                 # Add **seg if you need absolutely all original keys, but be cautious
             })
        return {
            "segments": default_segments,
            "total_duration": batch_total_duration,
            "skippable_duration": 0
        }
        
async def process_segments_parallel(segments, client, key_points, batch_size=40, max_concurrent=3, base_progress=80, progress_range=14): # Add progress params
    """Process segments in parallel with controlled concurrency and progress reporting"""
    log(f"Starting parallel processing of {len(segments)} segments with key points context.")
    report_progress(base_progress, "analyzing", f"Starting segment analysis...") # Initial progress for this step

    # We're reverting to the original approach that doesn't merge segments
    # This will keep the original whisper transcript segments as is
    merged_segments = segments.copy()
    
    # Log merge results
    original_count = len(segments)
    merged_count = len(merged_segments)
    average_duration = sum(s["end"] - s["start"] for s in merged_segments) / merged_count if merged_count > 0 else 0
    
    log(f"Using {merged_count} original segments with average duration: {average_duration:.1f}s")
    
    # Adjust batch size based on segment count
    dynamic_batch_size = min(max(10, merged_count // 4), batch_size)
    batches = [merged_segments[i:i+dynamic_batch_size] for i in range(0, len(merged_segments), dynamic_batch_size)]
    total_batches = len(batches)

    semaphore = asyncio.Semaphore(max_concurrent)
    processed_batches_count = 0 # Counter for progress

    async def process_with_semaphore(batch, idx):
        nonlocal processed_batches_count
        async with semaphore:
            if idx > 0:
                delay = min(0.2 * (2 ** (idx // 3)), 2.0)
                await asyncio.sleep(delay)
            result = await process_batch_async(client, batch, idx, total_batches, key_points)
            # --- Report progress after each batch ---
            processed_batches_count += 1
            current_progress = base_progress + int((processed_batches_count / total_batches) * progress_range)
            report_progress(current_progress, "analyzing", f"Analyzed batch {idx+1}/{total_batches}")
            # --- End progress reporting ---
            return result

    tasks = [process_with_semaphore(batch, i) for i, batch in enumerate(batches)]
    batch_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Combine results
    all_segments = []
    total_duration = 0
    skippable_duration = 0
    
    # --- Added Logging ---
    log(f"Aggregating results from {len(batch_results)} batches.")
    # --- End Added Logging ---
    for i, result in enumerate(batch_results):
        if isinstance(result, Exception):
            log(f"Batch {i+1} failed: {str(result)}", "ERROR")
            continue
            
        # --- Added Logging ---
        log(f"Batch {i+1} result: {len(result.get('segments', []))} segments.")
        # --- End Added Logging ---
        all_segments.extend(result["segments"])
        total_duration += result["total_duration"]
        skippable_duration += result["skippable_duration"]
    
    # --- Added Logging ---
    log(f"Aggregation complete. Total segments processed: {len(all_segments)}. Original input count: {len(segments)}")
    if len(all_segments) != len(segments):
         log(f"Warning: Segment count mismatch after aggregation! Input={len(segments)}, Output={len(all_segments)}", "WARNING")
    if all_segments:
        log(f"Sample - First processed segment text: '{all_segments[0].get('text', 'N/A')[:50]}...' score: {all_segments[0].get('importance_score', 'N/A')}")
        log(f"Sample - Last processed segment text: '{all_segments[-1].get('text', 'N/A')[:50]}...' score: {all_segments[-1].get('importance_score', 'N/A')}")
    # --- End Added Logging ---

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
    
    # Final progress update for this step (slightly before Node parsing)
    report_progress(base_progress + progress_range, "analyzing", "Segment analysis complete.")
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
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [{level}] {message}", file=sys.stderr, flush=True) # Log to stderr

# Define the container image with all necessary dependencies
image = (
        Image.from_registry("ubuntu:24.04", add_python="3.11", force_build=True)
.apt_install(
        "ffmpeg",
        "git",
        "python3-pip",
        "build-essential",
        "python3-dev",
        "libsndfile1",
        "libglib2.0-0",
        #"libgl1-mesa-glx",
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

@app.function(
    gpu="T4",
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
                        model="gpt-4.1-nano-2025-04-14",
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
    gpu="A10G",
    image=image,
    timeout=1800,
    secrets=[Secret.from_name("openai-secret")]
)
async def process_audio(audio_data: bytes, filename: str):
    """Process just the audio track for faster upload and processing"""
    temp_files = []
    # --- Define Progress Mapping for Audio ---
    # Assuming audio processing maps to 40%-94% overall progress
    PROGRESS_START = 40
    PROGRESS_LOAD_MODEL_END = 42
    PROGRESS_TRANSCRIBE_END = 75
    PROGRESS_CONTENT_GEN_END = 80
    PROGRESS_SEGMENT_ANALYSIS_START = 80
    PROGRESS_SEGMENT_ANALYSIS_END = 94
    # --- End Progress Mapping ---
    try:
        report_progress(PROGRESS_START, "preparing", "Starting audio processing")
        # Create a safe filename without spaces
        safe_filename = filename.replace(" ", "_")
        
        # Save audio data temporarily
        temp_audio_path = Path("/tmp") / f"{safe_filename}.wav"
        temp_files.append(temp_audio_path)  # Track for cleanup
        temp_audio_path.write_bytes(audio_data)
        log(f"Saved audio file: {temp_audio_path} ({len(audio_data)} bytes)")

        # Create directories if they don't exist
        temp_audio_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Verify audio file exists and has content
            if not temp_audio_path.exists():
                raise FileNotFoundError(f"Audio file not created at {temp_audio_path}")
            audio_size = temp_audio_path.stat().st_size
            log(f"Audio file size: {audio_size} bytes")
            if audio_size == 0:
                raise ValueError("Audio file is empty")

            # --- Transcribe with Whisper ---
            report_progress(PROGRESS_START, "loading_model", "Loading transcription model...")
            log("Loading Whisper model...")
            model = whisper.load_model("small")
            report_progress(PROGRESS_LOAD_MODEL_END, "loading_model", "Transcription model loaded.")

            report_progress(PROGRESS_LOAD_MODEL_END, "transcribing", "Starting transcription...")
            log("Starting transcription...")
            result = model.transcribe(
                str(temp_audio_path),
                language='en',
                verbose=True # Keep verbose for Whisper's own logs (to stderr)
            )
            report_progress(PROGRESS_TRANSCRIBE_END, "transcribing", "Transcription complete.")

            if not result or not isinstance(result, dict):
                raise ValueError(f"Invalid Whisper result: {result}")
                
            transcript = result.get("text", "")
            segments = result.get("segments", [])
            
            log(f"Transcription complete: {len(transcript)} characters, {len(segments)} segments")
            
            # --- Added Logging ---
            log(f"Whisper produced {len(segments)} segments.")
            if segments:
                 log(f"Sample - First Whisper segment text: '{segments[0].get('text', 'N/A')[:50]}...'")
                 log(f"Sample - Last Whisper segment text: '{segments[-1].get('text', 'N/A')[:50]}...'")
            # --- End Added Logging ---

            # --- Process with OpenAI ---
            report_progress(PROGRESS_TRANSCRIBE_END, "generating_summary", "Generating summary...")
            log("Generating educational content...")
            client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

            # Calculate optimal number of flashcards based on content
            optimal_flashcard_count = calculate_optimal_flashcard_count(segments, transcript)
            
            # Adjust key points count based on flashcard count (minimum 3, scale with flashcards)
            key_points_count = max(3, min(optimal_flashcard_count, 8))

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
                    "{key_points_count} essential insights that represent the most important takeaways"
                ],
                "flashcards": [
                    {{
                        "question": "Conceptual question testing understanding",
                        "answer": "Precise, factual answer focusing on core concept"
                    }}
                    ...exactly {optimal_flashcard_count} flashcards covering different concepts, ensuring comprehensive coverage of the material...
                ]
            }}

            TRANSCRIPT:
            {transcript[:4000]}"""  # Limit transcript length to avoid token limits

            log(f"Generating educational content with {optimal_flashcard_count} flashcards and {key_points_count} key points...")
            content_response = client.chat.completions.create(
                model="gpt-4.1-nano-2025-04-14",
                messages=[
                    {
                        "role": "system",
                        "content": f"You are an expert at analyzing educational content and returning results in JSON format. Generate exactly {optimal_flashcard_count} diverse flashcards that comprehensively cover the material. Always respond with valid JSON matching the requested structure."
                    },
                    {
                        "role": "user",
                        "content": content_prompt
                    }
                ],
                temperature=0.5,
                max_tokens=3000,  # Increased token limit to accommodate more flashcards
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
                
                # Log the actual counts vs expected
                actual_flashcards = len(content_data["flashcards"])
                actual_keypoints = len(content_data["keyPoints"])
                log(f"Generated {actual_flashcards} flashcards (expected {optimal_flashcard_count}) and {actual_keypoints} key points (expected {key_points_count})")
                
                # Warn if counts don't match but don't fail - the API did its best
                if actual_flashcards != optimal_flashcard_count:
                    log(f"Warning: Got {actual_flashcards} flashcards instead of requested {optimal_flashcard_count}", "WARNING")
                if actual_keypoints != key_points_count:
                    log(f"Warning: Got {actual_keypoints} key points instead of requested {key_points_count}", "WARNING")
                    
            except (json.JSONDecodeError, KeyError, IndexError, ValueError) as e:
                log(f"Error parsing OpenAI response: {str(e)}", "ERROR")
                # Create fallback content with the optimal counts
                content_data = {
                    "summary": "Error generating summary - content analysis failed",
                    "keyPoints": [f"Key point {i+1}: Error generating content" for i in range(key_points_count)],
                    "flashcards": [{"question": f"Question {i+1}: Error generating content", "answer": f"Answer {i+1}: Error generating content"} for i in range(optimal_flashcard_count)]
                }
                
            log("Successfully generated educational content")
            key_points_list = content_data.get("keyPoints", []) # Get the key points
            log(f"Final content: {len(content_data.get('flashcards', []))} flashcards, {len(key_points_list)} key points")
            report_progress(PROGRESS_CONTENT_GEN_END, "generating_keypoints", "Generated summary and key points.") # Update stage

            # --- Process segments ---
            # Pass progress parameters to the parallel function
            log("Analyzing segments using parallel processing...")
            analyzed_segments, stats = await process_segments_parallel(
                segments, 
                client,
                key_points_list, # Pass the extracted key points
                batch_size=20,
                max_concurrent=3,
                base_progress=PROGRESS_SEGMENT_ANALYSIS_START,
                progress_range=(PROGRESS_SEGMENT_ANALYSIS_END - PROGRESS_SEGMENT_ANALYSIS_START)
            )
            # report_progress is called inside process_segments_parallel

            # --- Added Logging ---
            log(f"Parallel processing returned {len(analyzed_segments)} segments.")
            if analyzed_segments:
                 log(f"Sample - First analyzed segment text: '{analyzed_segments[0].get('text', 'N/A')[:50]}...' score: {analyzed_segments[0].get('importance_score', 'N/A')}")
                 log(f"Sample - Last analyzed segment text: '{analyzed_segments[-1].get('text', 'N/A')[:50]}...' score: {analyzed_segments[-1].get('importance_score', 'N/A')}")
            # --- End Added Logging ---

            # --- Return final result ---
            # Node.js will report 95% (parsing_results) after receiving this JSON
            return {
                "status": "success",
                "summary": content_data["summary"],
                "keyPoints": content_data["keyPoints"],
                "flashcards": content_data["flashcards"],
                "transcript": transcript,
                "segments": analyzed_segments,
                "stats": stats
            }

        except Exception as e:
            log(f"Error during audio processing: {str(e)}", "ERROR")
            log(f"Stack trace: {traceback.format_exc()}", "ERROR")
            raise

    except Exception as e:
        log(f"Error in process_audio: {str(e)}", "ERROR")
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
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
                    log(f"Cleaned up temporary file: {temp_file}")
            except Exception as e:
                log(f"Error cleaning up {temp_file}: {str(e)}", "WARNING")

@app.function(
    gpu="T4",
    image=image,
    timeout=1800,
    secrets=[Secret.from_name("openai-secret")]
)
async def process_video(video_data: bytes, filename: str):
    """Main function that handles video processing, transcription, and content analysis"""
    temp_files = []
    # --- Define Progress Mapping for Video ---
    # Assuming video processing maps to 35%-94% overall progress
    PROGRESS_START = 35
    PROGRESS_EXTRACT_AUDIO_START = 36
    PROGRESS_EXTRACT_AUDIO_END = 40
    PROGRESS_LOAD_MODEL_END = 42
    PROGRESS_TRANSCRIBE_END = 75
    PROGRESS_CONTENT_GEN_END = 80
    PROGRESS_SEGMENT_ANALYSIS_START = 80
    PROGRESS_SEGMENT_ANALYSIS_END = 94
    # --- End Progress Mapping ---
    try:
        report_progress(PROGRESS_START, "preparing_script", "Starting video processing")
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
            # --- Extract audio ---
            report_progress(PROGRESS_EXTRACT_AUDIO_START, "extracting_audio", "Extracting audio...")
            log("Extracting audio for transcription using ffmpeg...")
            temp_audio_path = temp_video_path.with_suffix('.wav')
            temp_files.append(temp_audio_path)  # Track for cleanup
            
            # FFmpeg command for audio extraction
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', str(temp_video_path),
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # PCM format
                '-ar', '16000',  # 16kHz sample rate
                '-ac', '1',  # Mono
                '-y',  # Overwrite output file
                str(temp_audio_path)
            ]
            
            # Run ffmpeg
            import subprocess
            subprocess.run(ffmpeg_cmd, check=True, stderr=subprocess.PIPE)
            report_progress(PROGRESS_EXTRACT_AUDIO_END, "extracting_audio", "Audio extraction complete.")
            
            log(f"Audio extraction complete to {temp_audio_path}")

            # Verify audio file exists and has content
            if not temp_audio_path.exists():
                raise FileNotFoundError(f"Audio file not created at {temp_audio_path}")
            audio_size = temp_audio_path.stat().st_size
            log(f"Audio file size: {audio_size} bytes")
            if audio_size == 0:
                raise ValueError("Audio file is empty")

            # --- Transcribe with Whisper ---
            report_progress(PROGRESS_EXTRACT_AUDIO_END, "loading_model", "Loading transcription model...")
            log("Loading Whisper model...")
            model = whisper.load_model("small")
            report_progress(PROGRESS_LOAD_MODEL_END, "loading_model", "Transcription model loaded.")

            report_progress(PROGRESS_LOAD_MODEL_END, "transcribing", "Starting transcription...")
            log("Starting transcription...")
            result = model.transcribe(
                str(temp_audio_path),
                language='en',
                verbose=True # Keep verbose for Whisper's own logs (to stderr)
            )
            report_progress(PROGRESS_TRANSCRIBE_END, "transcribing", "Transcription complete.")

            if not result or not isinstance(result, dict):
                raise ValueError(f"Invalid Whisper result: {result}")
                
            transcript = result.get("text", "")
            segments = result.get("segments", [])
            
            log(f"Transcription complete: {len(transcript)} characters, {len(segments)} segments")
            
            # --- Added Logging ---
            log(f"Whisper produced {len(segments)} segments.")
            if segments:
                 log(f"Sample - First Whisper segment text: '{segments[0].get('text', 'N/A')[:50]}...'")
                 log(f"Sample - Last Whisper segment text: '{segments[-1].get('text', 'N/A')[:50]}...'")
            # --- End Added Logging ---

            # Clean up audio file early
            if temp_audio_path.exists():
                temp_audio_path.unlink()
                log("Cleaned up audio file")

            # --- Process with OpenAI ---
            report_progress(PROGRESS_TRANSCRIBE_END, "generating_summary", "Generating summary...")
            log("Generating educational content...")
            client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

            # Calculate optimal number of flashcards based on content
            optimal_flashcard_count = calculate_optimal_flashcard_count(segments, transcript)
            
            # Adjust key points count based on flashcard count (minimum 3, scale with flashcards)
            key_points_count = max(3, min(optimal_flashcard_count, 8))

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
                    "{key_points_count} essential insights that represent the most important takeaways"
                ],
                "flashcards": [
                    {{
                        "question": "Conceptual question testing understanding",
                        "answer": "Precise, factual answer focusing on core concept"
                    }}
                    ...exactly {optimal_flashcard_count} flashcards covering different concepts, ensuring comprehensive coverage of the material...
                ]
            }}

            TRANSCRIPT:
            {transcript[:4000]}"""  # Limit transcript length to avoid token limits

            log(f"Generating educational content with {optimal_flashcard_count} flashcards and {key_points_count} key points...")
            content_response = client.chat.completions.create(
                model="gpt-4.1-nano-2025-04-14",
                messages=[
                    {
                        "role": "system",
                        "content": f"You are an expert at analyzing educational content and returning results in JSON format. Generate exactly {optimal_flashcard_count} diverse flashcards that comprehensively cover the material. Always respond with valid JSON matching the requested structure."
                    },
                    {
                        "role": "user",
                        "content": content_prompt
                    }
                ],
                temperature=0.5,
                max_tokens=3000,  # Increased token limit to accommodate more flashcards
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
                
                # Log the actual counts vs expected
                actual_flashcards = len(content_data["flashcards"])
                actual_keypoints = len(content_data["keyPoints"])
                log(f"Generated {actual_flashcards} flashcards (expected {optimal_flashcard_count}) and {actual_keypoints} key points (expected {key_points_count})")
                
                # Warn if counts don't match but don't fail - the API did its best
                if actual_flashcards != optimal_flashcard_count:
                    log(f"Warning: Got {actual_flashcards} flashcards instead of requested {optimal_flashcard_count}", "WARNING")
                if actual_keypoints != key_points_count:
                    log(f"Warning: Got {actual_keypoints} key points instead of requested {key_points_count}", "WARNING")
                    
            except (json.JSONDecodeError, KeyError, IndexError, ValueError) as e:
                log(f"Error parsing OpenAI response: {str(e)}", "ERROR")
                # Create fallback content with the optimal counts
                content_data = {
                    "summary": "Error generating summary - content analysis failed",
                    "keyPoints": [f"Key point {i+1}: Error generating content" for i in range(key_points_count)],
                    "flashcards": [{"question": f"Question {i+1}: Error generating content", "answer": f"Answer {i+1}: Error generating content"} for i in range(optimal_flashcard_count)]
                }
                
            log("Successfully generated educational content")
            key_points_list = content_data.get("keyPoints", []) # Get the key points
            log(f"Final content: {len(content_data.get('flashcards', []))} flashcards, {len(key_points_list)} key points")
            report_progress(PROGRESS_CONTENT_GEN_END, "generating_keypoints", "Generated summary and key points.") # Update stage

            # --- Process segments ---
            # Pass progress parameters to the parallel function
            log("Analyzing segments using parallel processing...")
            analyzed_segments, stats = await process_segments_parallel(
                segments, 
                client,
                key_points_list, # Pass the extracted key points
                batch_size=20,      # Number of segments per batch
                max_concurrent=3,    # Maximum number of concurrent API calls
                base_progress=PROGRESS_SEGMENT_ANALYSIS_START,
                progress_range=(PROGRESS_SEGMENT_ANALYSIS_END - PROGRESS_SEGMENT_ANALYSIS_START)
            )
            # report_progress is called inside process_segments_parallel

            # --- Added Logging ---
            log(f"Parallel processing returned {len(analyzed_segments)} segments.")
            if analyzed_segments:
                 log(f"Sample - First analyzed segment text: '{analyzed_segments[0].get('text', 'N/A')[:50]}...' score: {analyzed_segments[0].get('importance_score', 'N/A')}")
                 log(f"Sample - Last analyzed segment text: '{analyzed_segments[-1].get('text', 'N/A')[:50]}...' score: {analyzed_segments[-1].get('importance_score', 'N/A')}")
            # --- End Added Logging ---

            # --- Return final result ---
            # Node.js will report 95% (parsing_results) after receiving this JSON
            return {
                "status": "success",
                "summary": content_data["summary"],
                "keyPoints": content_data["keyPoints"],
                "flashcards": content_data["flashcards"],
                "transcript": transcript,
                "segments": analyzed_segments,
                "stats": stats
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
