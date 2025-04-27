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
        - 10-8: Essential content (core concepts, definitions, key principles) → 1.0x speed
        - 7-5: Supporting content (examples, explanations, context) → 1.5x speed
        - 4-1: Supplementary content (repetition, tangents, filler) → 2.0-2.5x speed

        PLAYBACK SPEED MUST follow this strict formula:
        - For score 10: Use speed 1.0
        - For score 9: Use speed 1.0
        - For score 8: Use speed 1.1
        - For score 7: Use speed 1.3
        - For score 6: Use speed 1.5
        - For score 5: Use speed 1.7
        - For score 4: Use speed 1.9
        - For score 3: Use speed 2.1
        - For score 2: Use speed 2.3
        - For score 1: Use speed 2.5

        SEGMENTS TO ANALYZE:
        {segments_text}

        REQUIRED JSON RESPONSE FORMAT:
        {{
            "ratings": [
                {{
                    "score": number,          // 1-10 rating
                    "speed": number,          // MUST follow the formula above
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
                    "content": "You analyze educational content and return ratings in JSON format. Follow the playback speed formula precisely based on the score. Always respond with valid JSON matching the requested structure."
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
        
        # --- Added Logging ---
        log(f"[Batch {batch_idx+1}] Received {len(ratings)} ratings for {len(batch)} segments.")
        if len(ratings) != len(batch):
             log(f"[Batch {batch_idx+1}] Warning: Rating count mismatch. Will adjust.", "WARNING")
             # Adjust ratings array size if needed (existing logic)
             if len(ratings) < len(batch):
                 ratings.extend([
                     {"score": 5, "speed": 1.7, "skip": False, "key_point": "Default (missing)"} 
                     for _ in range(len(batch) - len(ratings))
                 ])
             else:
                 ratings = ratings[:len(batch)]
        # --- End Added Logging ---

        processed_segments = []
        batch_duration = 0
        batch_skippable = 0
        
        for i, (seg, rating) in enumerate(zip(batch, ratings)): # Use enumerate for index
            try:
                segment_duration = seg["end"] - seg["start"]
                batch_duration += segment_duration
                
                # --- Added Logging ---
                log(f"[Batch {batch_idx+1}, Seg {i+1}] Original Segment: start={seg['start']:.1f}, end={seg['end']:.1f}, text='{seg['text'][:50]}...'")
                log(f"[Batch {batch_idx+1}, Seg {i+1}] Received Rating: {rating}")
                # --- End Added Logging ---

                importance_score = float(rating.get('score', 5))
                importance_score = max(1, min(10, importance_score))
                
                # Calculate playback speed based on score using our formula
                # This ensures a direct relationship between score and speed
                playback_speed_map = {
                    10: 1.0, 9: 1.0, 8: 1.1,
                    7: 1.3, 6: 1.5, 5: 1.7,
                    4: 1.9, 3: 2.1, 2: 2.3, 1: 2.5
                }
                default_speed = 1.5
                
                # First try to use the speed from the API response
                api_speed = float(rating.get('speed', 0.0))
                
                # If API speed is valid, use it, otherwise calculate from the score
                if 1.0 <= api_speed <= 2.5:
                    playback_speed = api_speed
                else:
                    # Fallback to calculating from score
                    playback_speed = playback_speed_map.get(int(importance_score), default_speed)
                
                # Ensure speed is within bounds
                playback_speed = max(1.0, min(2.5, playback_speed))
                
                can_skip = bool(rating.get('skip', importance_score <= 3))
                
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
                    "adjusted_duration": segment_duration / playback_speed,
                    "reason": rating.get('key_point', '')
                }
                processed_segments.append(processed_segment_data)

                # --- Added Logging ---
                log(f"[Batch {batch_idx+1}, Seg {i+1}] Processed Segment: score={importance_score}, speed={playback_speed}, skip={can_skip}, text='{processed_segment_data['text'][:50]}...'")
                if importance_score <= 1:
                     log(f"[Batch {batch_idx+1}, Seg {i+1}] Low score segment (<=1) processed. Text included: {'Yes' if processed_segment_data['text'] else 'No'}", "DEBUG")
                # --- End Added Logging ---
                
            except Exception as e:
                log(f"Error processing segment {i+1} in batch {batch_idx+1}: {str(e)}", "WARNING")
                # Add default values for failed segments
                segment_duration = seg["end"] - seg["start"]
                default_segment = {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"], # Ensure text is included even in default
                    "can_skip": False,
                    "importance_score": 5,
                    "playback_speed": 1.5,
                    "original_duration": segment_duration,
                    "adjusted_duration": segment_duration / 1.5,
                    "reason": f"Error processing segment: {str(e)[:50]}"
                }
                processed_segments.append(default_segment)
                log(f"[Batch {batch_idx+1}, Seg {i+1}] Added default segment due to error. Text included: {'Yes' if default_segment['text'] else 'No'}", "WARNING")
        
        # --- Added Logging ---
        log(f"[Batch {batch_idx+1}] Finished processing. Produced {len(processed_segments)} segments.")
        # --- End Added Logging ---
        return {
            "segments": processed_segments,
            "total_duration": batch_duration,
            "skippable_duration": batch_skippable
        }
        
    except Exception as e:
        log(f"Error in batch {batch_idx+1}: {str(e)}", "ERROR")
        # --- Added Logging ---
        log(f"[Batch {batch_idx+1}] Returning default segments due to batch error. Input count: {len(batch)}")
        # --- End Added Logging ---
        return {
            "segments": [{
                **seg,
                "can_skip": False,
                "importance_score": 5,
                "playback_speed": 1.5,
                "original_duration": seg["end"] - seg["start"],
                "adjusted_duration": (seg["end"] - seg["start"]) / 1.5,
                "reason": f"Batch error: {str(e)[:50]}",
                "text": seg.get("text", "[Text missing due to batch error]") # Ensure text field exists
            } for seg in batch],
            "total_duration": sum(seg["end"] - seg["start"] for seg in batch),
            "skippable_duration": 0
        }
        
async def process_segments_parallel(segments, client, batch_size=40, max_concurrent=3):
    """Process segments in parallel with controlled concurrency"""
    log(f"Starting parallel processing of {len(segments)} segments")
    
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
    gpu="T4",
    image=image,
    timeout=1800,
    secrets=[Secret.from_name("openai-secret")]
)
async def process_audio(audio_data: bytes, filename: str):
    """Process just the audio track for faster upload and processing"""
    temp_files = []  # Keep track of temporary files
    try:
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

            # Transcribe with Whisper - using tiny model for speed
            log("Loading Whisper model...")
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
            
            # --- Added Logging ---
            log(f"Whisper produced {len(segments)} segments.")
            if segments:
                 log(f"Sample - First Whisper segment text: '{segments[0].get('text', 'N/A')[:50]}...'")
                 log(f"Sample - Last Whisper segment text: '{segments[-1].get('text', 'N/A')[:50]}...'")
            # --- End Added Logging ---

            # Process with OpenAI for summary, key points, and flashcards
            log("Generating educational content...")
            client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

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
            analyzed_segments, stats = await process_segments_parallel(
                segments, 
                client,
                batch_size=20,
                max_concurrent=3
            )

            # --- Added Logging ---
            log(f"Parallel processing returned {len(analyzed_segments)} segments.")
            if analyzed_segments:
                 log(f"Sample - First analyzed segment text: '{analyzed_segments[0].get('text', 'N/A')[:50]}...' score: {analyzed_segments[0].get('importance_score', 'N/A')}")
                 log(f"Sample - Last analyzed segment text: '{analyzed_segments[-1].get('text', 'N/A')[:50]}...' score: {analyzed_segments[-1].get('importance_score', 'N/A')}")
            # --- End Added Logging ---

            # Return the final result
            return {
                "status": "success",
                "summary": content_data["summary"],
                "keyPoints": content_data["keyPoints"][:5],
                "flashcards": content_data["flashcards"][:5],
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
            # Extract audio for transcription using ffmpeg directly instead of moviepy
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
            
            log(f"Audio extraction complete to {temp_audio_path}")

            # Verify audio file exists and has content
            if not temp_audio_path.exists():
                raise FileNotFoundError(f"Audio file not created at {temp_audio_path}")
            audio_size = temp_audio_path.stat().st_size
            log(f"Audio file size: {audio_size} bytes")
            if audio_size == 0:
                raise ValueError("Audio file is empty")

            # Transcribe with Whisper using tiny model
            log("Loading Whisper model...")
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

            # Process with OpenAI for summary, key points, and flashcards
            log("Generating educational content...")
            client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

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
            analyzed_segments, stats = await process_segments_parallel(
                segments, 
                client,
                batch_size=20,      # Number of segments per batch
                max_concurrent=3    # Maximum number of concurrent API calls
            )

            # --- Added Logging ---
            log(f"Parallel processing returned {len(analyzed_segments)} segments.")
            if analyzed_segments:
                 log(f"Sample - First analyzed segment text: '{analyzed_segments[0].get('text', 'N/A')[:50]}...' score: {analyzed_segments[0].get('importance_score', 'N/A')}")
                 log(f"Sample - Last analyzed segment text: '{analyzed_segments[-1].get('text', 'N/A')[:50]}...' score: {analyzed_segments[-1].get('importance_score', 'N/A')}")
            # --- End Added Logging ---

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
