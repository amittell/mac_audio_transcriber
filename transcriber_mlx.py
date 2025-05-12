#!/usr/bin/env python3
# transcriber_mlx.py

import argparse
import os
import sys # Added for platform check
import logging
import pandas as pd
import gc
from datetime import timedelta
import shutil
import time # For timing the process
import threading # Added for heartbeat
from collections import OrderedDict # For type checking if needed

# Attempt to load .env file if python-dotenv is installed
try:
    from dotenv import load_dotenv
    if load_dotenv():
        pass
except ImportError:
    pass

# --- MLX Specific Imports ---
try:
    import mlx.core as mx
    import mlx_whisper
except ImportError:
    # Basic logging needed here if imports fail early
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
from pyannote.audio.core.inference import Inference # For type checking
from pyannote.audio.core.pipeline import Pipeline as PyannotePipeline # For type checking


# --- Configuration & Defaults ---
DEFAULT_MODEL_NAME_SCRIPT = "mlx-community/whisper-small-mlx"
DEFAULT_LANGUAGE_SCRIPT = "en"
DEFAULT_HF_TOKEN_SCRIPT = None
DEFAULT_MODEL_DOWNLOAD_ROOT_SCRIPT = None
DEFAULT_OUTPUT_DIR_SCRIPT = "."
DEFAULT_DIARIZATION_DEVICE_SCRIPT = "cpu"


DEFAULT_MODEL_NAME = os.getenv("WHISPERX_PROCESSOR_MODEL_NAME", DEFAULT_MODEL_NAME_SCRIPT)
DEFAULT_LANGUAGE = os.getenv("WHISPERX_PROCESSOR_LANGUAGE", DEFAULT_LANGUAGE_SCRIPT)
DEFAULT_HF_TOKEN = os.getenv("WHISPERX_PROCESSOR_HF_TOKEN", DEFAULT_HF_TOKEN_SCRIPT)
DEFAULT_MODEL_DOWNLOAD_ROOT = os.getenv("WHISPERX_PROCESSOR_MODEL_DOWNLOAD_ROOT", DEFAULT_MODEL_DOWNLOAD_ROOT_SCRIPT)
DEFAULT_OUTPUT_DIR = os.getenv("WHISPERX_PROCESSOR_OUTPUT_DIR", DEFAULT_OUTPUT_DIR_SCRIPT)
DEFAULT_DIARIZATION_DEVICE = os.getenv("WHISPERX_PROCESSOR_DIARIZATION_DEVICE", DEFAULT_DIARIZATION_DEVICE_SCRIPT)


# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("MLX_Whisper_Processor")

try:
    from dotenv import load_dotenv 
    if os.path.exists(".env") and any(True for _ in open(".env")):
        if 'dotenv_loaded' not in globals():
            logger.info("Configuration from .env file might have been loaded (if python-dotenv is installed).")
            dotenv_loaded = True
    else:
        logger.debug("No .env file found or it is empty.")
except ImportError:
    logger.debug("python-dotenv not installed, .env file will not be loaded.")


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
                text_line = f"[{segment.get('speaker', 'SPEAKER_?')}] {segment['text'].strip()}"
                srt_file.write(f"{segment_idx}\n{format_timestamp_srt(segment['start'])} --> {format_timestamp_srt(segment['end'])}\n{text_line}\n\n")
                segment_idx += 1
                continue
            if not words_in_segment: continue
            current_speaker_block = []
            for word_info in words_in_segment:
                if not all(k in word_info for k in ['start', 'word']): continue
                current_word_speaker = word_info.get("speaker", "SPEAKER_?")
                word_end = word_info.get("end", word_info["start"] + 0.5)
                if not current_speaker_block or current_speaker_block[-1]["speaker"] == current_word_speaker:
                    current_speaker_block.append({"text": word_info["word"].strip(), "start": word_info["start"], "end": word_end, "speaker": current_word_speaker})
                else:
                    if current_speaker_block:
                        block_text = " ".join(w["text"] for w in current_speaker_block)
                        srt_file.write(f"{segment_idx}\n{format_timestamp_srt(current_speaker_block[0]['start'])} --> {format_timestamp_srt(current_speaker_block[-1]['end'])}\n[{current_speaker_block[0]['speaker']}] {block_text}\n\n")
                        segment_idx += 1
                    current_speaker_block = [{"text": word_info["word"].strip(), "start": word_info["start"], "end": word_end, "speaker": current_word_speaker}]
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
        if not words:
            if all(k in segment for k in ['start', 'end', 'text']):
                 rows.append({"segment_id": seg_idx, "word_id": None, "speaker": seg_speaker, "start_time": segment['start'], "end_time": segment['end'], "text": segment['text'].strip()})
            continue
        for word_idx, word in enumerate(words):
            if not all(k in word for k in ['start', 'end', 'word']): continue
            rows.append({"segment_id": seg_idx, "word_id": word_idx, "speaker": word.get('speaker', seg_speaker), "start_time": word['start'], "end_time": word['end'], "text": word['word'].strip()})
    if rows: pd.DataFrame(rows).to_csv(output_filename, index=False, encoding='utf-8'); logger.info(f"CSV file saved to {output_filename}")
    else: logger.warning(f"No data to write to CSV for {output_filename}")

def save_txt(result_with_speakers, output_filename):
    output_filename = get_unique_filename(output_filename)
    with open(output_filename, 'w', encoding='utf-8') as txt_file:
        if "segments" not in result_with_speakers:
            logger.warning("No segments found to create TXT file.")
            return
        current_speaker = None
        for segment in result_with_speakers["segments"]:
            words = segment.get("words", [])
            if not words and 'text' in segment:
                speaker = segment.get('speaker', 'SPEAKER_?')
                if speaker != current_speaker: txt_file.write(f"\n\n[{speaker}]:\n" if current_speaker else f"[{speaker}]:\n"); current_speaker = speaker
                txt_file.write(segment['text'].strip() + " ")
                continue
            if not words: continue
            if current_speaker is not None and any(word.get('speaker', 'SPEAKER_?') != current_speaker for word in words if 'speaker' in word):
                 txt_file.write("\n")
            for word_info in words:
                if 'word' not in word_info: continue
                speaker = word_info.get('speaker', 'SPEAKER_?')
                if speaker != current_speaker: txt_file.write(f"\n\n[{speaker}]:\n" if current_speaker else f"[{speaker}]:\n"); current_speaker = speaker
                txt_file.write(word_info['word'].strip() + " ")
        txt_file.write("\n")
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

    try:
        temp_audio_for_duration = whisperx.load_audio(args.input_audio)
        audio_duration_seconds = len(temp_audio_for_duration) / 16000.0
        del temp_audio_for_duration
        gc.collect()
        logger.info(f"Audio duration: {timedelta(seconds=int(audio_duration_seconds))}.")
    except Exception as e:
        logger.error(f"Failed to load audio for duration check: {e}", exc_info=True)

    transcription_result = None
    try:
        logger.info(f"Loading MLX ASR model '{args.model_name}' and transcribing...")
        transcription_result = mlx_whisper.transcribe(
            audio=args.input_audio,
            path_or_hf_repo=args.model_name,
            language=args.language if args.language and args.language.lower() != "auto" else None,
            word_timestamps=True,
            verbose=args.print_progress,
            initial_prompt=args.initial_prompt
        )
        logger.info("MLX Transcription complete.")
    except Exception as e:
        logger.error(f"MLX Transcription failed: {e}", exc_info=True); sys.exit(1)
    finally:
        gc.collect()

    logger.info("Word timestamps obtained from MLX. Separate alignment step is not used with MLX.")

    result_with_speakers = transcription_result
    if args.diarize:
        diarize_model_obj = None
        try:
            logger.info("Loading diarization pipeline (via whisperx)...")
            token_for_diarization = args.hf_token or os.environ.get("WHISPERX_PROCESSOR_HF_TOKEN") or os.environ.get("HF_TOKEN")
            audio_for_diarization = whisperx.load_audio(args.input_audio) 

            # Determine diarization device
            requested_device = args.diarization_device
            actual_device = "cpu" # Default
            mps_available = False
            if requested_device == "mps":
                if sys.platform == "darwin":
                    try:
                        # Ensure torch is imported before checking
                        import torch 
                        mps_available = torch.backends.mps.is_available() and torch.backends.mps.is_built()
                        if mps_available:
                            actual_device = "mps"
                            logger.info("MPS device is available and selected for diarization.")
                        else:
                            logger.warning("MPS device requested, but it's not available on this system (torch.backends.mps.is_available()=False). Falling back to CPU.")
                    except ImportError:
                         logger.warning("PyTorch 'torch' library not found. Cannot check for MPS device. Using CPU.")
                    except AttributeError:
                         logger.warning("PyTorch version might be too old for MPS check (torch.backends.mps). Using CPU.")
                else: # MPS only on macOS
                    logger.warning("MPS device requested, but it's only available on macOS. Falling back to CPU.")
            
            if actual_device == "cpu":
                 logger.info("Using CPU for diarization.")

            # Create DiarizationPipeline with the determined device
            diarize_model_obj = DiarizationPipeline(use_auth_token=token_for_diarization, device=actual_device)


            if args.diarization_progress:
                try:
                    # Note: Rich test happens in __main__ block now
                    logger.info("Rich library found. Attempting to enable internal pyannote.audio progress hooks.")
                    
                    pyannote_pipeline_instance = diarize_model_obj.model
                    if not isinstance(pyannote_pipeline_instance, PyannotePipeline):
                         logger.warning(f"diarize_model_obj.model is not a Pyannote Pipeline instance (type: {type(pyannote_pipeline_instance)}). Skipping progress hook setup.")
                    else:
                        logger.debug(f"Type of diarize_model_obj.model (pyannote_pipeline_instance): {type(pyannote_pipeline_instance)}")
                        
                        components_to_modify = []
                        
                        # Check standard storage locations for Inference objects in pyannote.audio.Pipeline
                        dict_attributes_to_check = ["_models", "_inferences"]
                        
                        for dict_attr_name in dict_attributes_to_check:
                            if hasattr(pyannote_pipeline_instance, dict_attr_name):
                                model_dict = getattr(pyannote_pipeline_instance, dict_attr_name)
                                if isinstance(model_dict, (dict, OrderedDict)):
                                    if args.debug: logger.debug(f"  Iterating values in '{dict_attr_name}' (type: {type(model_dict).__name__}):")
                                    for model_name, model_value in model_dict.items():
                                        if args.debug: logger.debug(f"    Checking item '{model_name}' (type: {type(model_value).__name__})")
                                        if isinstance(model_value, Inference):
                                            if model_value not in components_to_modify:
                                                components_to_modify.append(model_value)
                                                logger.info(f"Identified Inference instance '{model_name}' from '{dict_attr_name}' for progress hook.")
                                        elif isinstance(model_value, PyannotePipeline) and args.debug: 
                                            logger.debug(f"    Item '{model_name}' is a sub-pipeline. Deeper inspection not yet implemented here.")
                                else:
                                    if args.debug: logger.debug(f"  Attribute '{dict_attr_name}' is not a dict (type: {type(model_dict).__name__}).")
                            elif args.debug:
                                logger.debug(f"  Attribute '{dict_attr_name}' not found on pipeline instance.")

                        # Fallback check: Iterate direct attributes (including single underscore)
                        if not components_to_modify: 
                            if args.debug:
                                logger.debug(f"Primary dictionary search yielded no Inference objects. Iterating direct attributes of {type(pyannote_pipeline_instance).__name__} as fallback:")
                            for attr_name in dir(pyannote_pipeline_instance):
                                if attr_name.startswith('__') or attr_name in dict_attributes_to_check: 
                                    continue
                                try:
                                    attr_value = getattr(pyannote_pipeline_instance, attr_name)
                                    if isinstance(attr_value, Inference): 
                                        if attr_value not in components_to_modify:
                                            components_to_modify.append(attr_value)
                                            logger.info(f"Identified Inference instance (direct attribute) at '.{attr_name}' for progress hook.")
                                    elif args.debug: 
                                         if not callable(attr_value) and not isinstance(attr_value, (int, float, str, bool, list, dict, tuple, set, type(None))):
                                            logger.debug(f"    Fallback check: '.{attr_name}' (type: {type(attr_value).__name__}) - not Inference.")
                                except Exception:
                                    pass 

                        if not components_to_modify:
                             logger.warning("Could not identify any pyannote.audio Inference components to attach progress hook. Detailed progress might not appear.")
                        else:
                            for component_to_set_hook_on in components_to_modify:
                                try:
                                    component_to_set_hook_on.hook_ = True 
                                    logger.info(f"Set hook_=True for pyannote Inference component: {type(component_to_set_hook_on).__name__}")
                                except Exception as e_hook_set:
                                    logger.warning(f"Failed to set hook_ for component {type(component_to_set_hook_on).__name__}: {e_hook_set}")

                except ImportError:
                    logger.info("Rich library not installed. Detailed diarization progress bar may not be available. Consider `pip install rich`.")
                except Exception as e_progress_setup:
                    logger.warning(f"Could not set up detailed diarization progress: {e_progress_setup}", exc_info=True)

            logger.info("Performing speaker diarization... This may take a very long time for lengthy audio.")

            diarization_complete_event = threading.Event()
            diarization_result_store = {"segments": None, "error": None}
            start_diarization_time = time.time()

            def diarize_worker():
                try:
                    segments = diarize_model_obj(audio_for_diarization,
                                             min_speakers=args.min_speakers,
                                             max_speakers=args.max_speakers)
                    diarization_result_store["segments"] = segments
                except Exception as e:
                    diarization_result_store["error"] = e
                finally:
                    diarization_complete_event.set()

            diarization_thread = threading.Thread(target=diarize_worker, daemon=True)
            diarization_thread.start()

            heartbeat_interval = 30
            while not diarization_complete_event.wait(timeout=heartbeat_interval):
                elapsed_time_diar = time.time() - start_diarization_time
                logger.info(f"Diarization still in progress... (Elapsed: {timedelta(seconds=int(elapsed_time_diar))}) (Device: {actual_device})") 

            diarization_thread.join()

            if diarization_result_store["error"]:
                if actual_device == "mps":
                     logger.error(f"Error during diarization thread using MPS device: {diarization_result_store['error']}", exc_info=diarization_result_store['error'])
                     logger.error("Processing on MPS failed. Try running with '--diarization_device cpu'")
                else:
                     logger.error(f"Error during diarization thread using CPU device: {diarization_result_store['error']}", exc_info=diarization_result_store['error'])
                raise diarization_result_store["error"]

            diarize_segments = diarization_result_store["segments"]
            del audio_for_diarization
            gc.collect()

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
            if 'actual_device' in locals() and actual_device == "mps": # Check if actual_device was defined
                 logger.error("Processing on MPS failed. Try running with '--diarization_device cpu'")
        finally:
            if diarize_model_obj:
                del diarize_model_obj
                gc.collect()
    else:
        logger.info("Diarization skipped.")

    base_output_filename = os.path.splitext(os.path.basename(args.input_audio))[0]
    diarization_applied_successfully = False
    if args.diarize and "segments" in result_with_speakers:
        for segment in result_with_speakers.get("segments", []):
            if "words" in segment:
                for word in segment.get("words", []):
                    if "speaker" in word: diarization_applied_successfully = True; break
            if diarization_applied_successfully: break
            if "speaker" in segment: diarization_applied_successfully = True; break

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
    parser.add_argument("--diarization_progress", action="store_true", default=False,
                        help="Attempt to enable detailed pyannote.audio progress bar using 'rich' library (install 'rich' separately).")

    # Output format arguments
    parser.add_argument("--no_csv", action="store_false", dest="save_csv", help="Do not save CSV output.")
    parser.add_argument("--no_srt", action="store_false", dest="save_srt", help="Do not save SRT output.")
    parser.add_argument("--no_txt", action="store_false", dest="save_txt", help="Do not save TXT output.")
    parser.set_defaults(save_csv=True, save_srt=True, save_txt=True)

    args = parser.parse_args()

    # Configure logging levels based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled by command line flag.")
        logging.getLogger("pyannote").setLevel(logging.DEBUG)
        logging.getLogger("speechbrain").setLevel(logging.WARNING if not os.getenv("SPEECHBRAIN_DEBUG") else logging.DEBUG)
    else:
        # Keep third-party libraries quieter by default
        logging.getLogger("pyannote").setLevel(logging.WARNING)
        logging.getLogger("speechbrain").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING) 
        logging.getLogger("fsspec").setLevel(logging.WARNING)


    # --- Rich Test Added Here ---
    # This runs early, only if --diarization_progress is set
    if args.diarization_progress: 
        try:
            import rich
            # Use logger.info for start/end markers for consistency
            logger.info("--- Performing Rich Library Test ---")
            # Direct rich.print() call to test terminal rendering capabilities
            rich.print("[bold blue]Rich test:[/bold blue] If you see this in [italic green]color[/italic green] and [bold]bold[/bold], basic rich rendering is working.")
            logger.info("--- Rich Library Test End ---")
        except ImportError:
            logger.info("Rich library not installed, skipping rich test (and progress bar). Consider `pip install rich`.")
        except Exception as e_rich_test:
            logger.warning(f"Rich library test failed: {e_rich_test}")
    # --- End Rich Test ---


    # Ensure WHISPERX_PROCESSOR_HF_TOKEN is available if HF_TOKEN is set and the former is not
    if os.getenv("HF_TOKEN") and not os.getenv("WHISPERX_PROCESSOR_HF_TOKEN"):
        os.environ["WHISPERX_PROCESSOR_HF_TOKEN"] = os.environ["HF_TOKEN"]

    process_audio_mlx(args)