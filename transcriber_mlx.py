#!/usr/bin/env python3
# transcriber_mlx.py
#
# Author: Alex Mittell
# GitHub: amittell
# Date: 2025-05-12
# License: MIT License
#
# Copyright (c) 2025 Alex Mittell
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import os
import sys
import logging
import pandas as pd
import gc
from datetime import timedelta
import shutil
import time # For timing the process
import threading # Added for heartbeat

# Attempt to load .env file if python-dotenv is installed
# This initial attempt is basic; more detailed logging occurs after logger setup.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass # python-dotenv not installed, .env will be handled later if present

# --- MLX Specific Imports ---
try:
    import mlx.core as mx
    import mlx_whisper
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger("MLX_Whisper_Processor").error("MLX or mlx-whisper library not found. Please install it (e.g., 'pip install mlx mlx-whisper')")
    sys.exit(1)

# --- PyTorch Import (needed early for MPS check) ---
try:
    import torch
except ImportError:
    # Log later once logger is configured, but note the absence for MPS check
    pass

# --- WhisperX components ---
import whisperx
from whisperx.diarize import DiarizationPipeline, assign_word_speakers


# --- Configuration & Defaults ---
DEFAULT_MODEL_NAME_SCRIPT = "mlx-community/whisper-small-mlx"
DEFAULT_LANGUAGE_SCRIPT = "en"
DEFAULT_HF_TOKEN_SCRIPT = None
# DEFAULT_MODEL_DOWNLOAD_ROOT_SCRIPT = None # Removed as not used by MLX or this diarization setup
DEFAULT_OUTPUT_DIR_SCRIPT = "."
DEFAULT_DIARIZATION_DEVICE_SCRIPT = "mps"


DEFAULT_MODEL_NAME = os.getenv("WHISPERX_PROCESSOR_MODEL_NAME", DEFAULT_MODEL_NAME_SCRIPT)
DEFAULT_LANGUAGE = os.getenv("WHISPERX_PROCESSOR_LANGUAGE", DEFAULT_LANGUAGE_SCRIPT)
DEFAULT_HF_TOKEN = os.getenv("WHISPERX_PROCESSOR_HF_TOKEN", DEFAULT_HF_TOKEN_SCRIPT)
# DEFAULT_MODEL_DOWNLOAD_ROOT = os.getenv("WHISPERX_PROCESSOR_MODEL_DOWNLOAD_ROOT", DEFAULT_MODEL_DOWNLOAD_ROOT_SCRIPT) # Removed
DEFAULT_OUTPUT_DIR = os.getenv("WHISPERX_PROCESSOR_OUTPUT_DIR", DEFAULT_OUTPUT_DIR_SCRIPT)
DEFAULT_DIARIZATION_DEVICE = os.getenv("WHISPERX_PROCESSOR_DIARIZATION_DEVICE", DEFAULT_DIARIZATION_DEVICE_SCRIPT)


# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("MLX_Whisper_Processor")

# Load .env file if python-dotenv is installed and .env exists
try:
    from dotenv import load_dotenv
    if load_dotenv(): # python-dotenv's load_dotenv returns True if .env was loaded
        logger.info("Configuration from .env file has been loaded.")
    elif not os.path.exists(".env"):
        logger.debug("No .env file found. Environment variables will be used if set.")
    else: # .env exists but load_dotenv returned False (e.g., empty or malformed)
        logger.debug(".env file found but may not have been loaded (possibly empty or malformed).")
except ImportError:
    logger.debug("python-dotenv not installed, .env file will not be loaded. Relying on environment variables.")


# --- Helper Functions ---
def check_ffmpeg_availability():
    return shutil.which("ffmpeg") is not None

def format_timestamp_srt(seconds_float):
    if seconds_float is None or seconds_float < 0: seconds_float = 0
    delta = timedelta(seconds=seconds_float)
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int(delta.microseconds / 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def get_unique_filename(filepath):
    if not os.path.exists(filepath):
        return filepath
    base, ext = os.path.splitext(filepath)
    counter = 1
    new_filepath = f"{base}_{counter}{ext}"
    while os.path.exists(new_filepath):
        counter += 1
        new_filepath = f"{base}_{counter}{ext}"
    logger.info(f"Output file {filepath} exists. Saving as {new_filepath} instead.")
    return new_filepath

def save_srt(result_with_speakers, output_filename):
    output_filename = get_unique_filename(output_filename)
    with open(output_filename, 'w', encoding='utf-8') as srt_file:
        segment_idx = 1
        if "segments" not in result_with_speakers:
            logger.warning("No segments found to create SRT file.")
            return
        for segment in result_with_speakers["segments"]:
            words_in_segment = segment.get("words", [])
            if not words_in_segment and all(k in segment for k in ['start', 'end', 'text']):
                # Handle segments without word-level details (e.g., if alignment failed or wasn't fine-grained)
                text_line = f"[{segment.get('speaker', 'SPEAKER_?')}] {segment['text'].strip()}"
                srt_file.write(f"{segment_idx}\n{format_timestamp_srt(segment['start'])} --> {format_timestamp_srt(segment['end'])}\n{text_line}\n\n")
                segment_idx += 1
                continue

            if not words_in_segment: continue # Skip if no words and not a valid segment-level entry

            current_speaker_block = []
            for word_info in words_in_segment:
                if not all(k in word_info for k in ['start', 'word']): continue # Basic check for essential word data

                current_word_speaker = word_info.get("speaker", "SPEAKER_?")
                # Fallback for end timestamp if missing, though mlx_whisper should provide it
                word_end = word_info.get("end", word_info["start"] + 0.5)
                if word_info.get("end") is None:
                    logger.debug(f"Word '{word_info['word']}' at {word_info['start']:.2f}s missing 'end' timestamp in SRT. Defaulting duration.")


                if not current_speaker_block or current_speaker_block[-1]["speaker"] == current_word_speaker:
                    current_speaker_block.append({"text": word_info["word"].strip(), "start": word_info["start"], "end": word_end, "speaker": current_word_speaker})
                else:
                    # Speaker changed, write out the previous block
                    if current_speaker_block:
                        block_text = " ".join(w["text"] for w in current_speaker_block)
                        srt_file.write(f"{segment_idx}\n{format_timestamp_srt(current_speaker_block[0]['start'])} --> {format_timestamp_srt(current_speaker_block[-1]['end'])}\n[{current_speaker_block[0]['speaker']}] {block_text}\n\n")
                        segment_idx += 1
                    current_speaker_block = [{"text": word_info["word"].strip(), "start": word_info["start"], "end": word_end, "speaker": current_word_speaker}]
            
            # Write out the last block for the current segment
            if current_speaker_block:
                block_text = " ".join(w["text"] for w in current_speaker_block)
                srt_file.write(f"{segment_idx}\n{format_timestamp_srt(current_speaker_block[0]['start'])} --> {format_timestamp_srt(current_speaker_block[-1]['end'])}\n[{current_speaker_block[0]['speaker']}] {block_text}\n\n")
                segment_idx += 1
    logger.info(f"SRT file saved to {output_filename}")

def save_csv(result_with_speakers, output_filename):
    output_filename = get_unique_filename(output_filename)
    rows = []
    if "segments" not in result_with_speakers:
        logger.warning("No segments found to create CSV file.")
        return

    for seg_idx, segment in enumerate(result_with_speakers["segments"]):
        seg_speaker = segment.get('speaker', 'SPEAKER_?')
        words = segment.get("words", [])
        if not words: # Handle segments without word details but with overall text
            if all(k in segment for k in ['start', 'end', 'text']):
                rows.append({"segment_id": seg_idx, "word_id": None, "speaker": seg_speaker, "start_time": segment['start'], "end_time": segment['end'], "text": segment['text'].strip()})
            continue # Move to next segment

        for word_idx, word in enumerate(words):
            # Expect 'start', 'end', and 'word' from mlx_whisper word_timestamps
            if not all(k in word for k in ['start', 'end', 'word']):
                logger.debug(f"Skipping word due to missing keys (start, end, or word): {word}")
                continue
            rows.append({
                "segment_id": seg_idx,
                "word_id": word_idx,
                "speaker": word.get('speaker', seg_speaker), # Fallback to segment speaker
                "start_time": word['start'],
                "end_time": word['end'],
                "text": word['word'].strip()
            })

    if rows:
        pd.DataFrame(rows).to_csv(output_filename, index=False, encoding='utf-8')
        logger.info(f"CSV file saved to {output_filename}")
    else:
        logger.warning(f"No data to write to CSV for {output_filename}")

def save_txt(result_with_speakers, output_filename):
    output_filename = get_unique_filename(output_filename)
    with open(output_filename, 'w', encoding='utf-8') as txt_file:
        if "segments" not in result_with_speakers:
            logger.warning("No segments found to create TXT file.")
            return

        current_speaker = None
        for segment in result_with_speakers["segments"]:
            words = segment.get("words", [])
            if not words and 'text' in segment: # Segment-level text
                speaker = segment.get('speaker', 'SPEAKER_?')
                if speaker != current_speaker:
                    txt_file.write(f"\n\n[{speaker}]:\n" if current_speaker else f"[{speaker}]:\n")
                    current_speaker = speaker
                txt_file.write(segment['text'].strip() + " ")
                continue

            if not words: continue # Should be caught by above, but as a safeguard

            # Check if speaker changes within the words of this segment
            # This can happen if diarization assigns different speakers within an ASR segment
            if current_speaker is not None:
                first_word_speaker = words[0].get('speaker', 'SPEAKER_?') if words else None
                if first_word_speaker and first_word_speaker != current_speaker:
                     txt_file.write("\n") # Add a line break before new speaker block starts if current line had text

            for word_info in words:
                if 'word' not in word_info: continue
                speaker = word_info.get('speaker', 'SPEAKER_?')
                if speaker != current_speaker:
                    txt_file.write(f"\n\n[{speaker}]:\n" if current_speaker else f"[{speaker}]:\n")
                    current_speaker = speaker
                txt_file.write(word_info['word'].strip() + " ")
        txt_file.write("\n") # Final newline for cleanliness
    logger.info(f"TXT file saved to {output_filename}")

# --- Main Processing Function ---
def process_audio_mlx(args):
    start_time_total = time.time()

    if not os.path.exists(args.input_audio):
        logger.error(f"Input audio file not found: {args.input_audio}"); sys.exit(1)

    video_extensions = ['.mp4', '.mov', '.mkv', '.avi', '.flv', '.webm']
    _, input_ext = os.path.splitext(args.input_audio)
    if input_ext.lower() in video_extensions and not check_ffmpeg_availability():
        logger.error("Input is video, but ffmpeg not found. Please install ffmpeg."); sys.exit(1)

    if args.diarize:
        token_for_diarization = args.hf_token or os.environ.get("WHISPERX_PROCESSOR_HF_TOKEN") or os.environ.get("HF_TOKEN")
        if not token_for_diarization:
            logger.error("Diarization enabled but no Hugging Face token provided. Please use --hf_token or set WHISPERX_PROCESSOR_HF_TOKEN/HF_TOKEN."); sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Processing audio file: {args.input_audio}")
    logger.info(f"Using MLX. Parameters: Model='{args.model_name}', Lang='{args.language}', Diarize={args.diarize}, PrintProgress={args.print_progress}")

    loaded_audio_waveform = None
    try:
        logger.info(f"Loading audio file '{args.input_audio}' for duration check and potential diarization...")
        loaded_audio_waveform = whisperx.load_audio(args.input_audio) # Returns a NumPy array, resampled to 16kHz
        audio_duration_seconds = len(loaded_audio_waveform) / 16000.0 # whisperx.audio.SAMPLE_RATE is 16000
        logger.info(f"Audio duration: {timedelta(seconds=int(audio_duration_seconds))}.")
    except Exception as e:
        logger.error(f"Failed to load audio from '{args.input_audio}': {e}", exc_info=True)
        if loaded_audio_waveform is not None: del loaded_audio_waveform # Clean up if partially loaded
        sys.exit(1)

    transcription_result = None
    try:
        logger.info(f"Loading MLX ASR model '{args.model_name}' and transcribing...")
        transcription_result = mlx_whisper.transcribe(
            audio=args.input_audio, # MLX Whisper takes file path
            path_or_hf_repo=args.model_name,
            language=args.language if args.language and args.language.lower() != "auto" else None,
            word_timestamps=True, # Essential for diarization speaker assignment
            verbose=args.print_progress,
            initial_prompt=args.initial_prompt
        )
        logger.info("MLX Transcription complete.")
    except Exception as e:
        logger.error(f"MLX Transcription failed: {e}", exc_info=True)
        if loaded_audio_waveform is not None: del loaded_audio_waveform
        sys.exit(1)
    finally:
        gc.collect() # Collect any intermediate transcription objects

    logger.info("Word timestamps obtained from MLX. Separate alignment step (WhisperX align_model) is not used with MLX.")

    result_with_speakers = transcription_result
    diarize_model_obj = None

    if args.diarize:
        try:
            logger.info("Loading diarization pipeline (via whisperx)...")
            # Token already checked, reuse token_for_diarization
            
            # Determine diarization device
            requested_device = args.diarization_device
            actual_device = "cpu" # Default
            mps_available = False
            
            if requested_device == "mps":
                try:
                    if 'torch' not in sys.modules: # Check if torch was successfully imported
                         logger.warning("'torch' library not found. Cannot use MPS. Falling back to CPU for diarization.")
                    elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and hasattr(torch.backends.mps, 'is_available'):
                        mps_available = torch.backends.mps.is_available()
                        logger.info(f"PyTorch MPS availability check: {mps_available}")
                        if mps_available:
                            actual_device = "mps"
                        else:
                            logger.warning("MPS requested but PyTorch reports not available. Falling back to CPU for diarization.")
                    else: # Older PyTorch or incomplete MPS interface
                        logger.warning("This version of PyTorch may not fully support MPS or is missing attributes for check. Using CPU for diarization.")
                except Exception as e_mps: # Other errors during check
                    logger.warning(f"Error checking MPS availability: {e_mps}. Falling back to CPU for diarization.")
            
            logger.info(f"Initializing diarization with device: {actual_device}")
            diarize_model_obj = DiarizationPipeline(use_auth_token=token_for_diarization, device=actual_device)

            logger.info("Performing speaker diarization... This may take a very long time for lengthy audio.")
            
            diarization_complete_event = threading.Event()
            diarization_result_store = {"segments": None, "error": None}
            start_diarization_time = time.time()

            def diarize_worker():
                try:
                    # Use the pre-loaded audio waveform
                    segments = diarize_model_obj(loaded_audio_waveform,
                                                 min_speakers=args.min_speakers,
                                                 max_speakers=args.max_speakers)
                    diarization_result_store["segments"] = segments
                except Exception as e:
                    diarization_result_store["error"] = e
                finally:
                    diarization_complete_event.set()

            diarization_thread = threading.Thread(target=diarize_worker, daemon=True)
            diarization_thread.start()

            heartbeat_interval = 30 # seconds
            while not diarization_complete_event.wait(timeout=heartbeat_interval):
                elapsed_time_diar = time.time() - start_diarization_time
                logger.info(f"Diarization still in progress... (Elapsed: {timedelta(seconds=int(elapsed_time_diar))}) (Device: {actual_device})") # Using heartbeat logging

            diarization_thread.join() # Wait for the thread to finish

            if diarization_result_store["error"]:
                error_msg = f"Error during diarization thread using {actual_device} device: {diarization_result_store['error']}"
                logger.error(error_msg, exc_info=diarization_result_store['error'])
                if actual_device == "mps":
                    logger.error("Processing on MPS failed for diarization. Try running with '--diarization_device cpu'")
                raise diarization_result_store["error"]

            diarize_segments = diarization_result_store["segments"]
            
            if loaded_audio_waveform is not None:
                del loaded_audio_waveform # Audio waveform no longer needed after diarization
                loaded_audio_waveform = None # Mark as deleted
                gc.collect()
                logger.debug("Cleaned up pre-loaded audio waveform after diarization processing.")

            if diarize_segments is not None and not diarize_segments.empty:
                logger.info("Assigning speakers to words (using whisperx utility)...")
                if transcription_result and "segments" in transcription_result and transcription_result["segments"]:
                    result_with_speakers = assign_word_speakers(diarize_segments, transcription_result)
                    logger.info("Word-speaker assignment complete.")
                else:
                    logger.warning("Transcription segments are missing or empty. Cannot assign word speakers.")
            else:
                logger.warning("Diarization produced no segments or an empty result. Output will not have word-level speaker labels.")
        except Exception as e:
            logger.error(f"Diarization or speaker assignment failed: {e}", exc_info=True)
            # Check actual_device if it was set, otherwise it defaults to suggesting CPU
            current_diar_device = locals().get('actual_device', 'mps') # Safely get actual_device
            if current_diar_device == "mps":
                 logger.error("Processing on MPS failed. Try running with '--diarization_device cpu'")
        finally:
            if diarize_model_obj:
                del diarize_model_obj
                gc.collect()
                logger.debug("Cleaned up diarization model.")
    else:
        logger.info("Diarization skipped.")
        # If diarization was skipped, the pre-loaded audio waveform might still be in memory
        if loaded_audio_waveform is not None:
            del loaded_audio_waveform
            loaded_audio_waveform = None
            gc.collect()
            logger.debug("Cleaned up pre-loaded audio waveform (diarization was skipped).")


    base_output_filename = os.path.splitext(os.path.basename(args.input_audio))[0]
    diarization_applied_successfully = False
    if args.diarize and "segments" in result_with_speakers:
        for segment in result_with_speakers.get("segments", []):
            if "words" in segment:
                for word in segment.get("words", []):
                    if "speaker" in word: diarization_applied_successfully = True; break
            if diarization_applied_successfully: break
            if "speaker" in segment: diarization_applied_successfully = True; break # Check segment level too

    output_prefix = f"{base_output_filename}{'_diarized' if diarization_applied_successfully else ''}"

    if args.save_csv:
        save_csv(result_with_speakers, os.path.join(args.output_dir, f"{output_prefix}.csv"))
    if args.save_srt:
        save_srt(result_with_speakers, os.path.join(args.output_dir, f"{output_prefix}.srt"))
    if args.save_txt:
        save_txt(result_with_speakers, os.path.join(args.output_dir, f"{output_prefix}.txt"))

    end_time_total = time.time()
    logger.info(f"Processing of '{args.input_audio}' complete in {timedelta(seconds=int(end_time_total - start_time_total))}. Outputs are in '{args.output_dir}'")

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MLX Whisper Audio Processor: Transcribes audio using MLX Whisper (for Apple Silicon) and optionally diarizes using WhisperX utilities. Requires FFMPEG for MP4/video and MLX installation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("input_audio", type=str, help="Path to input audio/video file (e.g., .mp3, .mp4). FFMPEG needed for video.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory for output files.")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME,
                        help="Name of the MLX Whisper model from Hugging Face (e.g., 'mlx-community/whisper-small-mlx') or a local path to a converted MLX model directory.")
    parser.add_argument("--language", type=str, default=DEFAULT_LANGUAGE,
                        help="Language code (e.g., 'en', 'es'). 'auto' for detection. Default: 'en'.")
    parser.add_argument("--initial_prompt", type=str, default=None, help="Optional initial prompt for ASR to improve accuracy for specific words or context.")
    parser.add_argument("--print_progress", action="store_true", default=False, help="Print transcription progress from the MLX whisper library.")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug level logging for more detailed output, including pyannote logs and attribute inspection.")

    # Diarization specific arguments
    parser.add_argument("--diarize", action="store_true", help="Enable speaker diarization (uses pyannote.audio via whisperx).")
    parser.add_argument("--hf_token", type=str, default=DEFAULT_HF_TOKEN,
                        help="Hugging Face token for diarization models. Also WHISPERX_PROCESSOR_HF_TOKEN or HF_TOKEN env vars.")
    parser.add_argument("--min_speakers", type=int, default=None, help="Min speakers for diarization.")
    parser.add_argument("--max_speakers", type=int, default=None, help="Max speakers for diarization.")
    parser.add_argument("--diarization_device", type=str, default=DEFAULT_DIARIZATION_DEVICE, choices=["cpu", "mps"],
                        help="Device for diarization ('cpu' or 'mps'). MPS is experimental and requires macOS + PyTorch with MPS support.")

    # Output format arguments
    parser.add_argument("--no_csv", action="store_false", dest="save_csv", help="Do not save CSV output.")
    parser.add_argument("--no_srt", action="store_false", dest="save_srt", help="Do not save SRT output.")
    parser.add_argument("--no_txt", action="store_false", dest="save_txt", help="Do not save TXT output.")
    parser.set_defaults(save_csv=True, save_srt=True, save_txt=True)

    args = parser.parse_args()

    # Configure logging levels based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        for handler in logging.getLogger().handlers: # Ensure all handlers respect the new level
            handler.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled by command line flag.")
        # Enable more verbose logging from underlying libraries if in debug mode
        logging.getLogger("pyannote").setLevel(logging.DEBUG)
        logging.getLogger("speechbrain").setLevel(logging.DEBUG if os.getenv("SPEECHBRAIN_DEBUG") else logging.INFO) # speechbrain can be very verbose
        logging.getLogger("whisperx").setLevel(logging.DEBUG)
    else:
        # Keep third-party libraries quieter by default
        logging.getLogger("pyannote").setLevel(logging.WARNING)
        logging.getLogger("speechbrain").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING) 
        logging.getLogger("fsspec").setLevel(logging.WARNING)
        logging.getLogger("whisperx").setLevel(logging.INFO)


    # Ensure WHISPERX_PROCESSOR_HF_TOKEN is available if HF_TOKEN is set and the former is not
    # This allows users to use the more general HF_TOKEN if they have it set.
    if os.getenv("HF_TOKEN") and not os.getenv("WHISPERX_PROCESSOR_HF_TOKEN"):
        os.environ["WHISPERX_PROCESSOR_HF_TOKEN"] = os.environ["HF_TOKEN"]
        logger.debug("Using HF_TOKEN for WHISPERX_PROCESSOR_HF_TOKEN as it was not explicitly set.")

    process_audio_mlx(args)