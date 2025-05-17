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
import json # For action items JSON handling
import datetime # For timestamp in the report header

# Attempt to import LLM handler for local model usage
try:
    from llm_handler import (load_llm_model_and_tokenizer, invoke_llm_mlx, format_gemma_chat_prompt,
                             INPUT_TOO_LONG_ERROR_INDICATOR, PROMPT_TOKENIZATION_FAILED_INDICATOR, 
                             GENERATION_OOM_ERROR_INDICATOR)
    llm_handler_imported = True
except ImportError:
    llm_handler_imported = False

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
import re
import whisperx
from whisperx.diarize import DiarizationPipeline, assign_word_speakers


# --- Configuration & Defaults ---
DEFAULT_MODEL_NAME_SCRIPT = "mlx-community/whisper-small-mlx"
DEFAULT_LANGUAGE_SCRIPT = "en"
DEFAULT_HF_TOKEN_SCRIPT = None
# DEFAULT_MODEL_DOWNLOAD_ROOT_SCRIPT = None # Removed as not used by MLX or this diarization setup
DEFAULT_OUTPUT_DIR_SCRIPT = "."
DEFAULT_DIARIZATION_DEVICE_SCRIPT = "mps"

# --- LLM System Prompts and Configuration ---
CORRECTION_SYSTEM_PROMPT = """You are an expert linguistic editor specializing in refining machine-generated meeting transcripts for maximum accuracy and readability.
Your objective is to meticulously correct the provided transcript.

Follow these instructions precisely:
1.  **Correct Errors:** Identify and rectify all grammatical errors, spelling mistakes, punctuation inaccuracies, and incorrect capitalization.
2.  **Improve Readability:** Enhance sentence structure for clarity. Combine overly fragmented sentences where appropriate and break down run-on sentences to ensure smooth, natural language flow.
3.  **Maintain Speaker Labels:** The transcript uses speaker labels in the format [SPEAKER_XX] (e.g., [SPEAKER_00], [SPEAKER_01]). These labels are critical and MUST be preserved exactly as they appear before each utterance. Do NOT alter, add, or remove any speaker labels.
4.  **Handle Disfluencies:** Remove common verbal disfluencies (e.g., "um," "uh," "er," "like," "you know," "so," "well" when used as fillers) UNLESS their removal changes the speaker's intended meaning or conveys a significant hesitation that is contextually important. Strive for a clean, professional, and natural-sounding dialogue.
5.  **Preserve Content Integrity:** Your role is to correct and refine, NOT to add new information, summarize, or alter the factual content of the transcript. The meaning and intent of the original speakers must be fully preserved.
6.  **Output Format:** Return ONLY the fully corrected transcript. The output must start directly with the first speaker label or text and end with the last piece of dialogue. Do not include any introductory phrases, concluding remarks, or any text other than the corrected transcript itself."""

SUMMARY_SYSTEM_PROMPT = """You are an AI assistant programmed to be an expert summarizer of meeting transcripts.
Your task is to generate a concise and objective summary of the provided meeting transcript.

The summary must accurately reflect the content of the transcript and should focus on the following key elements:
* **Main Topics of Discussion:** Clearly identify the primary subjects that were discussed.
* **Key Decisions Made:** List any explicit decisions, agreements, or resolutions that were reached during the meeting.
* **Significant Outcomes/Conclusions:** Detail any important results, conclusions, or takeaways from the discussions.
* **Action Items (Brief Mention, if any):** If action items are explicitly mentioned as outcomes, briefly note their existence without detailing them (a separate process will extract detailed action items).

Guidelines for the summary:
1.  **Objectivity:** The summary must be strictly factual and based ONLY on the information present in the transcript. Do not include personal opinions, interpretations, or any information not explicitly stated.
2.  **Conciseness:** Be brief and to the point. Avoid unnecessary jargon or overly detailed explanations. The length should be appropriate for a high-level overview.
3.  **Clarity:** Use clear and straightforward language.
4.  **Output Format: CRITICAL: Return ONLY the summary text. Your entire response must be the summary itself. Do NOT include any introductory phrases (e.g., "Here is the summary:"), concluding remarks, or any text other than the summary itself.**
"""

ACTION_ITEMS_SYSTEM_PROMPT = """You are an AI specializing in meticulous action item extraction from meeting transcripts.
Your sole responsibility is to identify and structure all explicit tasks, commitments, or actions that require follow-up.

For each identified action item, you MUST provide:
1.  **Task (string):** A clear, concise description of what needs to be done. This should be a direct reflection of the commitment made in the transcript.
2.  **Assignee (string or list of strings):** The speaker label(s) (e.g., "[SPEAKER_00]", "[SPEAKER_01]") of the person(s) explicitly assigned or taking responsibility for the task.
    * If no one is explicitly assigned but context strongly implies an assignee (e.g., someone volunteers), use that speaker label.
    * If multiple individuals are clearly assigned to the *same specific task*, provide their speaker labels as a JSON list of strings (e.g., ["[SPEAKER_00]", "[SPEAKER_03]"]).
    * If no assignee can be determined from the transcript, use the string "Unassigned".
3.  **Deadline (string):** Any mentioned due date, timeframe (e.g., "End of Week", "Next Monday"), or temporal marker for completion.
    * If no deadline or timeframe is explicitly mentioned, use the string "Not specified".

Output Format:
* **CRITICAL: Your entire response MUST be a single, valid JSON list of objects and nothing else.**
* Do NOT include any introductory text, concluding text, markdown formatting (like ```json), or any explanations.
* Each object in the list represents a single action item and MUST contain exactly three keys: "task", "assignee", and "deadline".
* The values for these keys must be strings, except for "assignee" which can be a string or a list of strings as specified above.
* **If no action items are found in the transcript, you MUST return an empty JSON list: `[]` AND NOTHING ELSE.**
* If the same core task is mentioned multiple times, consolidate it into a single action item.

Example of the required JSON structure:
[
  {"task": "Prepare quarterly financial report", "assignee": "[SPEAKER_01]", "deadline": "Next Friday"},
  {"task": "Send follow-up email to Project Alpha team", "assignee": "[SPEAKER_02]", "deadline": "EOD Today"}
]
"""

DEFAULT_LLM_MODEL_ID = "mlx-community/gemma-3-4b-it-qat-4bit"

MODEL_CONTEXT_WINDOWS = {
    "mlx-community/gemma-3-4b-it-qat-4bit": 128000,
    "mlx-community/Phi-3-mini-4k-instruct-4bit": 4096,
    "mlx-community/Mistral-7B-Instruct-v0.2-4bit": 32000,
    "mlx-community/Mistral-7B-Instruct-v0.3-4bit": 32000, # Assuming same as v0.2, user should verify
    "mlx-community/Llama-3.1-8B-Instruct-4bit": 128000,
    # User can add more models here: "huggingface_model_id": context_window_integer
}
LLM_PROMPT_RESERVE_TOKENS = 256 # Tokens reserved for system prompt, query structure, and safety margin.
                                # This is an estimate. For very tight context windows, precise prompt token counting might be needed.

DEFAULT_CORRECTION_MAX_TOKENS = 32000 # Default new tokens for correction if --llm_max_tokens_correction is 0.
                                      # This should be enough for the LLM to regenerate a fairly long transcript.


DEFAULT_MODEL_NAME = os.getenv("WHISPERX_PROCESSOR_MODEL_NAME", DEFAULT_MODEL_NAME_SCRIPT)
DEFAULT_LANGUAGE = os.getenv("WHISPERX_PROCESSOR_LANGUAGE", DEFAULT_LANGUAGE_SCRIPT)
DEFAULT_HF_TOKEN = os.getenv("WHISPERX_PROCESSOR_HF_TOKEN", DEFAULT_HF_TOKEN_SCRIPT)
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
            return False

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
    return output_filename  # Return the actual filename used

def chunk_text_by_tokens(text: str, tokenizer, max_tokens_per_chunk: int, overlap_tokens: int = 50) -> list[str]:
    """
    Splits text into chunks with a maximum token count per chunk, including overlap.
    Ensures that overlap_tokens is less than max_tokens_per_chunk.
    """
    if not text or not tokenizer:
        logger.warning("chunk_text_by_tokens: Empty text or no tokenizer provided.")
        return [text] if text else [] # Return original text if it's just empty or tokenizer missing

    # Safety check for negative or zero values
    if max_tokens_per_chunk <= 0:
        logger.warning(f"Invalid max_tokens_per_chunk ({max_tokens_per_chunk}). Using default of 2048.")
        max_tokens_per_chunk = 2048

    if max_tokens_per_chunk <= overlap_tokens:
        logger.warning(f"chunk_text_by_tokens: max_tokens_per_chunk ({max_tokens_per_chunk}) "
                       f"must be greater than overlap_tokens ({overlap_tokens}). Using smaller overlap.")
        overlap_tokens = max(0, max_tokens_per_chunk // 10) # Use 10% of chunk size as overlap

    all_token_ids = []
    try:
        all_token_ids = tokenizer.encode(text)
    except Exception as e:
        logger.error(f"Failed to tokenize text for chunking: {e}. Returning text unchunked.", exc_info=True)
        return [text]

    if not all_token_ids:
        logger.warning("Text produced no tokens for chunking.")
        return []
        
    chunks_text = []
    current_pos_idx = 0
    text_len_tokens = len(all_token_ids)

    while current_pos_idx < text_len_tokens:
        end_pos_idx = min(current_pos_idx + max_tokens_per_chunk, text_len_tokens)
        chunk_token_ids_segment = all_token_ids[current_pos_idx:end_pos_idx]
        
        if not chunk_token_ids_segment: # Should not happen if loop condition is correct
            break

        try:
            # Ensure tokenizer.decode can handle a list of token IDs
            chunk_text_segment = tokenizer.decode(chunk_token_ids_segment) 
            if chunk_text_segment and chunk_text_segment.strip():
                chunks_text.append(chunk_text_segment)
            elif not chunk_text_segment and chunk_token_ids_segment:
                logger.warning(f"Failed to decode token chunk segment (tokens: {chunk_token_ids_segment[:10]}...) into text. Skipping this chunk segment.")
        except Exception as e:
            logger.warning(f"Error decoding token chunk segment: {e}. Tokens: {chunk_token_ids_segment[:10]}... Skipping this chunk segment.")

        if end_pos_idx == text_len_tokens: # Reached the end of the text
            break
        
        # Advance current_pos, ensuring we don't step back if overlap is large relative to chunk size
        step = max(1, max_tokens_per_chunk - overlap_tokens)
        old_pos_idx = current_pos_idx
        current_pos_idx += step

        # Safety break if current_pos_idx doesn't advance, to prevent infinite loop
        if current_pos_idx <= old_pos_idx and text_len_tokens > 0: # Corrected logic
            logger.error("Chunking position did not advance. Breaking to prevent infinite loop.")
            break
    
    return [c for c in chunks_text if c.strip()] # Final filter for any empty strings

def format_action_items_for_report(action_items_data, report_format="txt") -> str:
    """
    Formats action items for inclusion in the report.
    'action_items_data' can be a list of dicts (parsed JSON) or a string (error/raw output).
    'report_format' can be 'txt' or 'md'.
    """
    if not action_items_data:
        return "No action items were extracted or an error occurred during extraction.\n"

    lines = []
    if isinstance(action_items_data, list) and all(isinstance(item, dict) for item in action_items_data):
        if not action_items_data: # Empty list from LLM
            return "No action items identified.\n" if report_format == "txt" else "*No action items identified.*\n"
        
        if report_format == "md":
            lines.append("### Action Items\n")
            for i, item in enumerate(action_items_data):
                task = item.get("task", "N/A")
                assignee = item.get("assignee", "Unassigned")
                deadline = item.get("deadline", "Not specified")
                lines.append(f"- **Task {i+1}:** {task}")
                lines.append(f"  - **Assignee:** {assignee}")
                lines.append(f"  - **Deadline:** {deadline}")
            lines.append("\n") # Add a newline for spacing in markdown
        else: # txt format
            lines.append("Action Items:\n")
            for i, item in enumerate(action_items_data):
                task = item.get("task", "N/A")
                assignee = item.get("assignee", "Unassigned")
                deadline = item.get("deadline", "Not specified")
                lines.append(f"  - Task {i+1}: {task}")
                lines.append(f"    Assignee: {assignee}")
                lines.append(f"    Deadline: {deadline}\n")
    elif isinstance(action_items_data, str): # Error string or raw output
        if report_format == "md":
            lines.append("### Action Items (Extraction Note)\n")
            lines.append("Could not parse action items as structured data. Raw output from LLM:\n")
            lines.append("```text")
            lines.append(action_items_data)
            lines.append("```\n")
        else: # txt format
            lines.append("Action Items (Extraction Note):\n")
            lines.append("Could not parse action items as structured data. Raw output from LLM:\n")
            lines.append(action_items_data + "\n")
    else:
        return "Invalid format for action items data received.\n"
        
    return "\n".join(lines)

def generate_llm_report_content(args, llm_results: dict, base_transcript_for_report: str) -> str:
    """
    Generates the full content for the combined LLM report.
    'args' are the command-line arguments.
    'llm_results' is a dictionary containing 'summary_text', 'action_items_data', 
                    'corrected_text_available' (boolean), as well as status fields like
                    'correction_status', 'summary_status', 'action_items_status', and
                    'model_id_used'.
    'base_transcript_for_report' is the full transcript text to include (either original or corrected).
    """
    report_parts = []
    fmt = args.llm_report_format
    now = datetime.datetime.now()

    # --- Construct Header ---
    header_lines = []
    model_id_display = llm_results.get('model_id_used') or args.llm_model_id or 'N/A (LLM tasks not run or model failed to load)'
    
    if fmt == "md":
        header_lines.append(f"# LLM Post-Processing Report")
        header_lines.append(f"**Generated on:** {now.strftime('%Y-%m-%d %H:%M:%S')}")
        header_lines.append(f"**Input Audio:** `{os.path.basename(args.input_audio)}`")
        header_lines.append(f"**LLM Model Used:** `{model_id_display}`")
        header_lines.append("\n## Options Applied:")
        if args.llm_correct: 
            status = llm_results.get('correction_status', 'Unknown')
            header_lines.append(f"- Transcript Correction: Enabled - *{status}*")
        if args.llm_summarize: 
            status = llm_results.get('summary_status', 'Unknown')
            header_lines.append(f"- Summary Generation: Enabled - *{status}*")
        if args.llm_action_items: 
            status = llm_results.get('action_items_status', 'Unknown')
            header_lines.append(f"- Action Item Extraction: Enabled - *{status}*")
        if not (args.llm_correct or args.llm_summarize or args.llm_action_items):
            header_lines.append("- No LLM post-processing tasks were selected.")
    else: # txt format
        header_lines.append("--- LLM Post-Processing Report ---")
        header_lines.append(f"Generated on: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        header_lines.append(f"Input Audio: {os.path.basename(args.input_audio)}")
        header_lines.append(f"LLM Model Used: {model_id_display}")
        header_lines.append("\nOptions Applied:")
        if args.llm_correct: 
            status = llm_results.get('correction_status', 'Unknown')
            header_lines.append(f"  - Transcript Correction: Enabled - [{status}]")
        if args.llm_summarize: 
            status = llm_results.get('summary_status', 'Unknown')
            header_lines.append(f"  - Summary Generation: Enabled - [{status}]")
        if args.llm_action_items: 
            status = llm_results.get('action_items_status', 'Unknown')
            header_lines.append(f"  - Action Item Extraction: Enabled - [{status}]")
        if not (args.llm_correct or args.llm_summarize or args.llm_action_items):
            header_lines.append("  - No LLM post-processing tasks were selected.")
    report_parts.append("\n".join(header_lines))
    report_parts.append("\n" + ("---" if fmt == "txt" else "---") + "\n") # Separator

    # --- Conditionally Add Summary ---
    if args.llm_summarize:
        if fmt == "md":
            report_parts.append("## Summary\n")
            if llm_results.get('summary_text'):
                # Add status if it's not a simple "Completed" status
                status = llm_results.get('summary_status', '')
                if status and not status.startswith('Completed'):
                    report_parts.append(f"*Status: {status}*\n\n")
                report_parts.append(llm_results['summary_text'] + "\n")
            else:
                status = llm_results.get('summary_status', 'No summary generated')
                report_parts.append(f"*{status}*\n")
        else: # txt format
            report_parts.append("Summary:\n")
            if llm_results.get('summary_text'):
                status = llm_results.get('summary_status', '')
                if status and not status.startswith('Completed'):
                    report_parts.append(f"[Status: {status}]\n\n")
                report_parts.append(llm_results['summary_text'] + "\n\n")
            else:
                status = llm_results.get('summary_status', 'No summary generated')
                report_parts.append(f"[{status}]\n\n")
        report_parts.append(("\n---" if fmt == "txt" else "\n---\n"))

    # --- Conditionally Add Action Items ---
    if args.llm_action_items:
        if llm_results.get('action_items_data') is not None: # Check for presence, could be empty list or error string
            status = llm_results.get('action_items_status', '')
            if isinstance(llm_results['action_items_data'], str) and llm_results['action_items_data'].startswith('Error:'):
                # If it's an error string like "Error: Transcript too long..."
                if fmt == "md":
                    report_parts.append("## Action Items\n")
                    report_parts.append(f"*{llm_results['action_items_data']}*\n")
                else: # txt format
                    report_parts.append("Action Items:\n")
                    report_parts.append(f"{llm_results['action_items_data']}\n\n")
            else:
                # Normal action items data (JSON parsed or raw text)
                formatted_items = format_action_items_for_report(llm_results['action_items_data'], report_format=fmt)
                # Add status note if needed
                if status and not status.startswith('Completed (JSON parsed)'):
                    if fmt == "md":
                        formatted_items = f"*Note: {status}*\n\n" + formatted_items
                    else:
                        formatted_items = f"Note: {status}\n\n" + formatted_items
                report_parts.append(formatted_items)
        else:
            # No action items data
            status = llm_results.get('action_items_status', 'No action items generated')
            if fmt == "md":
                report_parts.append("## Action Items\n")
                report_parts.append(f"*{status}*\n")
            else:
                report_parts.append("Action Items:\n")
                report_parts.append(f"[{status}]\n\n")
        report_parts.append(("\n---" if fmt == "txt" else "\n---\n"))

    # --- Add Full Transcript ---
    # The 'base_transcript_for_report' will be the corrected one if correction was run,
    # otherwise it's the original base transcript loaded for LLM processing.
    if fmt == "md":
        transcript_header = "## Full Transcript"
        if llm_results.get('corrected_text_available'):
            transcript_header += " (Corrected by LLM)"
        report_parts.append(transcript_header + "\n")
        # For potentially long transcripts, consider if Markdown needs special handling,
        # but usually, it's fine as a large block of text. Could wrap in ```text ... ``` if desired.
        report_parts.append(base_transcript_for_report + "\n")
    else: # txt format
        transcript_header = "Full Transcript"
        if llm_results.get('corrected_text_available'):
            transcript_header += " (Corrected by LLM)"
        report_parts.append(transcript_header + ":\n")
        report_parts.append(base_transcript_for_report + "\n")
        
    return "\n".join(report_parts)

# --- Main Processing Function ---
def process_audio_mlx(args):
    start_time_total = time.time()
    
    # Initialize variables
    actual_txt_path = None
    
    # Initialize llm_results_for_report at the start
    llm_results_for_report = {
        'summary_text': None,
        'summary_status': "Not run",
        'action_items_data': None,
        'action_items_status': "Not run",
        'correction_status': "Not run",
        'corrected_text_available': False,
        'model_id_used': None
    }
    
    any_llm_task_enabled = args.llm_correct or args.llm_summarize or args.llm_action_items
    if any_llm_task_enabled and not llm_handler_imported:
        logger.error("LLM post-processing requested (--llm_correct, --llm_summarize, or --llm_action_items), "
                     "but llm_handler.py module could not be imported.")
        logger.error("Ensure llm_handler.py is in the same directory and dependencies (mlx-lm) are installed.")
        sys.exit(1)

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

    # --- LLM Initialization and Setup (MOVED EARLIER) ---
    llm_model = None
    llm_tokenizer = None
    current_model_context_window = 0
    effective_llm_model_id = args.llm_model_id # User-specified ID, or default from script if None
    llm_processing_active = any_llm_task_enabled # Set based on CLI flags

    if llm_processing_active: # Only proceed if any LLM task is enabled
        logger.info("--- Preparing for LLM Post-Processing ---")
        
        if not effective_llm_model_id: # If user did not specify AND args.llm_model_id was None initially
            effective_llm_model_id = DEFAULT_LLM_MODEL_ID # Use the script's default
            logger.info(f"No --llm_model_id provided. Using default LLM model: {effective_llm_model_id}")
        else:
            logger.info(f"User specified or defaulted LLM model: {effective_llm_model_id}")

        if not effective_llm_model_id: # Final check if still no model ID
            logger.error("LLM processing enabled, but no model ID could be determined (neither specified nor a script default). Skipping LLM tasks.")
            llm_processing_active = False # Disable LLM tasks
        else:
            if not llm_handler_imported: # Should have been caught earlier, but double-check
                logger.error("llm_handler.py not imported. Cannot load LLM. Skipping LLM tasks.")
                llm_processing_active = False
            else:
                logger.info(f"Attempting to load LLM model: {effective_llm_model_id}")
                llm_model, llm_tokenizer = load_llm_model_and_tokenizer(effective_llm_model_id)
                
                if not llm_model or not llm_tokenizer:
                    logger.warning(f"Failed to load LLM model '{effective_llm_model_id}'. LLM-dependent tasks will be skipped.")
                    llm_processing_active = False # Disable LLM tasks if loading failed
                else:
                    logger.info(f"LLM model '{effective_llm_model_id}' loaded successfully.")
                    current_model_context_window = MODEL_CONTEXT_WINDOWS.get(effective_llm_model_id)
                    if not current_model_context_window:
                        if '128k' in effective_llm_model_id.lower(): current_model_context_window = 128000
                        elif '32k' in effective_llm_model_id.lower(): current_model_context_window = 32000
                        elif '16k' in effective_llm_model_id.lower(): current_model_context_window = 16384
                        elif '8k' in effective_llm_model_id.lower(): current_model_context_window = 8192
                        elif '4k' in effective_llm_model_id.lower(): current_model_context_window = 4096
                        else:
                            default_fallback_context = 4096
                            logger.warning(
                                f"Context window for '{effective_llm_model_id}' not found in known models or inferable from name. "
                                f"Using a default fallback of {default_fallback_context} tokens. "
                                f"This may lead to errors if incorrect. Consider adding the model to MODEL_CONTEXT_WINDOWS in the script."
                            )
                            current_model_context_window = default_fallback_context
                    logger.info(f"Using context window for '{effective_llm_model_id}': {current_model_context_window} tokens.")
                    llm_results_for_report['model_id_used'] = effective_llm_model_id
    else:
        logger.debug("No LLM tasks enabled by user. Skipping all LLM post-processing.")
    # --- End of LLM Initialization and Setup ---

    loaded_audio_waveform = None
    try:
        logger.info(f"Loading audio file '{args.input_audio}' for duration check and potential diarization...")
        loaded_audio_waveform = whisperx.load_audio(args.input_audio) # Returns a NumPy array, resampled to 16kHz
        audio_duration_seconds = len(loaded_audio_waveform) / 16000.0 # whisperx.audio.SAMPLE_RATE is 16000
        logger.info(f"Audio duration: {timedelta(seconds=int(audio_duration_seconds))}.")
    except Exception as e:
        logger.error(f"Failed to load audio from '{args.input_audio}': {e}", exc_info=True)
        if loaded_audio_waveform is not None: del loaded_audio_waveform 
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
    actual_device = "cpu" # Default for diarization

    if args.diarize:
        try:
            logger.info("Loading diarization pipeline (via whisperx)...")
            # Token already checked, reuse token_for_diarization
            
            # Determine diarization device
            requested_device = args.diarization_device
            # actual_device already defaulted to "cpu"
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
            current_diar_device = actual_device # Use the determined actual_device
            if current_diar_device == "mps":
                 logger.error("Processing on MPS failed. Try running with '--diarization_device cpu'")
        finally:
            if diarize_model_obj:
                del diarize_model_obj
                gc.collect()
                logger.debug("Cleaned up diarization model.")
    else:
        logger.info("Diarization skipped.")
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
    
    # --- Save files to disk FIRST, before any LLM processing ---
    if args.save_csv:
        save_csv(result_with_speakers, os.path.join(args.output_dir, f"{output_prefix}.csv"))
    if args.save_srt:
        save_srt(result_with_speakers, os.path.join(args.output_dir, f"{output_prefix}.srt"))
    if args.save_txt:
        actual_txt_path = save_txt(result_with_speakers, os.path.join(args.output_dir, f"{output_prefix}.txt"))
        if actual_txt_path: # Check if save_txt returned a valid path
             logger.info(f"Saved transcript to: {actual_txt_path} (this will be used for LLM processing if enabled)")
        else:
            logger.warning(f"TXT file not saved (or path not returned from save_txt). LLM processing might be affected if it relies on this file.")
            actual_txt_path = os.path.join(args.output_dir, f"{output_prefix}.txt") # Fallback to expected path

    # --- LLM Post-Processing Section ---
    if llm_processing_active and llm_model and llm_tokenizer: # Ensure model was loaded
        logger.info("--- Starting LLM Post-Processing for Combined Report ---")
        
        base_txt_transcript_path = actual_txt_path
        
        if not base_txt_transcript_path or not os.path.exists(base_txt_transcript_path):
            logger.warning(f"Base TXT transcript not found at '{base_txt_transcript_path}'. LLM tasks might fail or use incorrect input.")
            # Attempt to find a fallback
            fallback_path = os.path.join(args.output_dir, f"{output_prefix}.txt")
            if os.path.exists(fallback_path):
                base_txt_transcript_path = fallback_path
                logger.info(f"Using fallback transcript path: {base_txt_transcript_path}")
            else:
                logger.error(f"Cannot find transcript file. Searched: {actual_txt_path}, {fallback_path}. Aborting LLM tasks.")
                llm_processing_active = False # Cannot proceed

        transcript_for_llm_and_report = ""
        if llm_processing_active and os.path.exists(base_txt_transcript_path):
            try:
                with open(base_txt_transcript_path, 'r', encoding='utf-8') as f:
                    transcript_for_llm_and_report = f.read()
                logger.info(f"Loaded base transcript for LLM processing (length: {len(transcript_for_llm_and_report)} chars).")
            except Exception as e:
                logger.error(f"Failed to read base transcript {base_txt_transcript_path}: {e}. Skipping LLM tasks.")
                transcript_for_llm_and_report = "" 
                llm_processing_active = False # Cannot proceed
        elif llm_processing_active: # Path didn't exist after checks
             logger.error(f"Base TXT transcript not found at {base_txt_transcript_path}. Cannot perform LLM tasks.")
             llm_processing_active = False


        if llm_processing_active and transcript_for_llm_and_report:
            # 1. Transcript Correction
            llm_results_for_report['correction_status'] = "Not run" 
            if args.llm_correct:
                logger.info("Task: LLM Transcript Correction")
                user_query_correct = f"Please correct the following meeting transcript:\n\n---\n{transcript_for_llm_and_report}\n---"
                correction_prompt = format_gemma_chat_prompt(user_query_correct, system_prompt=CORRECTION_SYSTEM_PROMPT)
                
                max_c_tokens = args.llm_max_tokens_correction if args.llm_max_tokens_correction > 0 else DEFAULT_CORRECTION_MAX_TOKENS
                if args.llm_max_tokens_correction == 0: # Using default
                    logger.info(f"Using default max new tokens for correction: {max_c_tokens}.")
                
                corrected_text_output = invoke_llm_mlx(llm_model, llm_tokenizer, correction_prompt,
                                                       model_context_window=current_model_context_window,
                                                       max_tokens=max_c_tokens, temperature=0.3,
                                                       verbose_generation=args.debug)
                
                if corrected_text_output == INPUT_TOO_LONG_ERROR_INDICATOR:
                    logger.error("Transcript too long for LLM correction with the current model's context window. Skipping correction.")
                    llm_results_for_report['correction_status'] = "Skipped: Transcript too long for model context."
                elif corrected_text_output == PROMPT_TOKENIZATION_FAILED_INDICATOR:
                    logger.error("Prompt tokenization failed for correction. Skipping correction.")
                    llm_results_for_report['correction_status'] = "Failed: Prompt tokenization error."
                elif corrected_text_output == GENERATION_OOM_ERROR_INDICATOR:
                    logger.error("Out of memory during LLM correction. Skipping correction.")
                    llm_results_for_report['correction_status'] = "Failed: Generation out of memory."
                elif corrected_text_output: 
                    transcript_for_llm_and_report = corrected_text_output
                    llm_results_for_report['corrected_text_available'] = True
                    llm_results_for_report['correction_status'] = "Completed."
                    logger.info("Transcript correction by LLM successful.")
                else: 
                    logger.warning("LLM correction did not return content or encountered an unspecified error. Subsequent tasks will use the uncorrected or previously loaded transcript.")
                    llm_results_for_report['correction_status'] = "Failed or no content returned."
            elif args.llm_correct and not transcript_for_llm_and_report: # Should be caught by outer if
                llm_results_for_report['correction_status'] = "Skipped: No transcript content."

            # 2. Summary Generation
            llm_results_for_report['summary_status'] = "Not run"
            if args.llm_summarize: # transcript_for_llm_and_report is already checked by parent if
                logger.info("Task: LLM Summary Generation")
                
                effective_context_for_content = current_model_context_window - LLM_PROMPT_RESERVE_TOKENS
                LLM_PROMPT_RESERVE_TOKENS_ADJUSTED = LLM_PROMPT_RESERVE_TOKENS # Keep track of what was used
                if effective_context_for_content <= 0:
                    logger.warning(f"LLM_PROMPT_RESERVE_TOKENS ({LLM_PROMPT_RESERVE_TOKENS}) is too large for model context window ({current_model_context_window}). Reducing reserve tokens.")
                    reduced_reserve = max(128, current_model_context_window // 10)
                    logger.info(f"Adjusted reserve tokens from {LLM_PROMPT_RESERVE_TOKENS} to {reduced_reserve}")
                    effective_context_for_content = current_model_context_window - reduced_reserve
                    LLM_PROMPT_RESERVE_TOKENS_ADJUSTED = reduced_reserve

                summary_text_output = None
                transcript_tokens = 0
                try:
                    transcript_tokens = len(llm_tokenizer.encode(transcript_for_llm_and_report))
                    logger.info(f"Full transcript token count for summary: {transcript_tokens}")
                except Exception as e:
                    logger.error(f"Failed to tokenize transcript for summarization length check: {e}. Using character-based estimation.", exc_info=True)
                    transcript_tokens = len(transcript_for_llm_and_report) // 4 
                    logger.warning(f"Using estimated token count for summary planning: ~{transcript_tokens}")

                if transcript_tokens <= effective_context_for_content:
                    logger.info("Transcript fits in context window for single-pass summarization.")
                    user_query_summary = f"Please summarize the following meeting transcript:\n\n---\n{transcript_for_llm_and_report}\n---"
                    summary_prompt = format_gemma_chat_prompt(user_query_summary, system_prompt=SUMMARY_SYSTEM_PROMPT)
                    summary_text_output = invoke_llm_mlx(llm_model, llm_tokenizer, summary_prompt,
                                                         model_context_window=current_model_context_window,
                                                         max_tokens=args.llm_max_tokens_summary, temperature=0.7,
                                                         verbose_generation=args.debug)
                    if summary_text_output == INPUT_TOO_LONG_ERROR_INDICATOR:
                        logger.error("Summarization failed: Input unexpectedly too long despite pre-check. This indicates an issue with token estimation or reserve tokens.")
                        llm_results_for_report['summary_status'] = "Failed: Input too long (unexpected)."
                        summary_text_output = None
                    elif summary_text_output in [PROMPT_TOKENIZATION_FAILED_INDICATOR, GENERATION_OOM_ERROR_INDICATOR] or not summary_text_output:
                        llm_results_for_report['summary_status'] = f"Failed: {summary_text_output if summary_text_output else 'No content returned'}."
                        summary_text_output = None
                    else:
                        llm_results_for_report['summary_status'] = "Completed (single pass)."
                else: 
                    logger.info(f"Transcript token count ({transcript_tokens}) exceeds effective context for single-pass summary ({effective_context_for_content}). Attempting chunked summarization.")
                    
                    chunk_content_token_limit = int(effective_context_for_content / 2) 
                    chunk_content_token_limit = max(256, chunk_content_token_limit) 
                    overlap = int(chunk_content_token_limit * 0.15) 
                    overlap = max(50, overlap) 

                    logger.info(f"Using chunk content token limit for summarization: {chunk_content_token_limit}, overlap: {overlap}")

                    transcript_chunks = chunk_text_by_tokens(transcript_for_llm_and_report, llm_tokenizer,
                                                             max_tokens_per_chunk=chunk_content_token_limit,
                                                             overlap_tokens=overlap)
                    if not transcript_chunks:
                        logger.error("Failed to chunk transcript for summarization, or no chunks produced.")
                        llm_results_for_report['summary_status'] = "Failed: Chunking error or no chunks."
                    else:
                        logger.info(f"Split transcript into {len(transcript_chunks)} chunks for summarization.")
                        chunk_summaries = []
                        all_chunks_processed_successfully = True
                        for i, chunk_content in enumerate(transcript_chunks):
                            logger.info(f"Summarizing chunk {i+1}/{len(transcript_chunks)} (length: {len(chunk_content)} chars)")
                            user_query_chunk_summary = f"This is part of a longer meeting. Please concisely summarize ONLY this segment of the transcript:\n\n---\n{chunk_content}\n---"
                            chunk_summary_prompt = format_gemma_chat_prompt(user_query_chunk_summary, system_prompt=SUMMARY_SYSTEM_PROMPT)
                            
                            max_tokens_for_chunk_summary = min(args.llm_max_tokens_summary // 2, 
                                                               int(len(llm_tokenizer.encode(chunk_content)) * 0.7) + 100) 
                            max_tokens_for_chunk_summary = max(150, max_tokens_for_chunk_summary) 
                            max_tokens_for_chunk_summary = min(max_tokens_for_chunk_summary, 2048)

                            chunk_summary_text = invoke_llm_mlx(llm_model, llm_tokenizer, chunk_summary_prompt,
                                                                model_context_window=current_model_context_window,
                                                                max_tokens=max_tokens_for_chunk_summary, temperature=0.6,
                                                                verbose_generation=args.debug)
                            if chunk_summary_text and chunk_summary_text not in [INPUT_TOO_LONG_ERROR_INDICATOR, PROMPT_TOKENIZATION_FAILED_INDICATOR, GENERATION_OOM_ERROR_INDICATOR]:
                                chunk_summaries.append(chunk_summary_text)
                                logger.debug(f"Chunk {i+1} summary (first 100 chars): {chunk_summary_text[:100]}")
                            else:
                                logger.warning(f"Failed to summarize chunk {i+1}. Reason: {chunk_summary_text if chunk_summary_text else 'No content'}. Skipping this chunk's summary.")
                                all_chunks_processed_successfully = False 
                        
                        if chunk_summaries:
                            combined_chunk_summaries = "\n\n---\nNext Segment Summary:\n---\n\n".join(chunk_summaries)
                            logger.info(f"Generated {len(chunk_summaries)} chunk summaries. Now generating final summary from combined chunk summaries (total chars: {len(combined_chunk_summaries)}).")
                            
                            combined_summaries_tokens = 0
                            try:
                                combined_summaries_tokens = len(llm_tokenizer.encode(combined_chunk_summaries))
                            except Exception as e:
                                logger.error(f"Failed to tokenize combined chunk summaries for final pass length check: {e}. Using concatenated summaries.", exc_info=True)
                                summary_text_output = combined_chunk_summaries
                                llm_results_for_report['summary_status'] = "Completed (concatenated chunk summaries due to final pass tokenization error)."
                            else:
                                if combined_summaries_tokens <= effective_context_for_content:
                                    user_query_final_summary = (
                                        "The following are summaries of sequential parts of a longer meeting. "
                                        "Please synthesize them into one final, coherent, and concise overall meeting summary. "
                                        "Ensure the final summary flows well and captures the most important points from all segments.\n\n"
                                        "---\nCombined Segment Summaries:\n---\n"
                                        f"{combined_chunk_summaries}\n---"
                                    )
                                    final_summary_prompt = format_gemma_chat_prompt(user_query_final_summary, system_prompt=SUMMARY_SYSTEM_PROMPT)
                                    summary_text_output = invoke_llm_mlx(llm_model, llm_tokenizer, final_summary_prompt,
                                                                         model_context_window=current_model_context_window,
                                                                         max_tokens=args.llm_max_tokens_summary, temperature=0.7,
                                                                         verbose_generation=args.debug)
                                    if summary_text_output and summary_text_output not in [INPUT_TOO_LONG_ERROR_INDICATOR, PROMPT_TOKENIZATION_FAILED_INDICATOR, GENERATION_OOM_ERROR_INDICATOR]:
                                        llm_results_for_report['summary_status'] = "Completed (chunked with final pass)."
                                    else:
                                        logger.warning(f"Final summarization pass failed or returned no content. Reason: {summary_text_output}. Using concatenated chunk summaries as fallback.")
                                        summary_text_output = combined_chunk_summaries # Fallback
                                        llm_results_for_report['summary_status'] = f"Completed (concatenated chunk summaries; final pass failed: {summary_text_output if summary_text_output else 'No content'})."
                                else:
                                    logger.warning(f"Combined chunk summaries (tokens: {combined_summaries_tokens}) are too long for a final summarization pass (effective context: {effective_context_for_content}). Using concatenated chunk summaries.")
                                    summary_text_output = combined_chunk_summaries
                                    llm_results_for_report['summary_status'] = "Completed (concatenated chunk summaries; too long for final pass)."
                        elif all_chunks_processed_successfully: 
                            logger.warning("All chunk summaries were empty. No final summary can be generated.")
                            llm_results_for_report['summary_status'] = "Failed: All chunk summaries were empty."
                        else: 
                            logger.warning("No chunk summaries were successfully generated due to errors.")
                            llm_results_for_report['summary_status'] = "Failed: No chunk summaries generated due to errors."
                
                if summary_text_output:
                    llm_results_for_report['summary_text'] = summary_text_output
                    logger.info(f"Summary generation by LLM finished. Status: {llm_results_for_report['summary_status']}")
                elif llm_results_for_report['summary_status'] == "Not run": # Check if it was never really processed
                    logger.warning("LLM summarization did not produce output and status not set; marking as failed.")
                    llm_results_for_report['summary_status'] = "Failed or no content returned."

            elif args.llm_summarize and not transcript_for_llm_and_report:
                llm_results_for_report['summary_status'] = "Skipped: No transcript content."


            # 3. Action Item Extraction
            llm_results_for_report['action_items_status'] = "Not run"
            if args.llm_action_items:
                logger.info("Task: LLM Action Item Extraction")
                user_query_action = f"Please extract action items from the following meeting transcript:\n\n---\n{transcript_for_llm_and_report}\n---"
                action_items_prompt = format_gemma_chat_prompt(user_query_action, system_prompt=ACTION_ITEMS_SYSTEM_PROMPT)

                action_items_raw_output = invoke_llm_mlx(llm_model, llm_tokenizer, action_items_prompt,
                                                         model_context_window=current_model_context_window,
                                                         max_tokens=args.llm_max_tokens_action_items, temperature=0.5,
                                                         verbose_generation=args.debug)
                
                if action_items_raw_output == INPUT_TOO_LONG_ERROR_INDICATOR:
                    logger.error("Transcript too long for LLM action item extraction. Skipping.")
                    llm_results_for_report['action_items_status'] = "Skipped: Transcript too long for model context."
                    llm_results_for_report['action_items_data'] = "Error: Transcript too long for action item extraction."
                elif action_items_raw_output == PROMPT_TOKENIZATION_FAILED_INDICATOR:
                    logger.error("Prompt tokenization failed for action items. Skipping.")
                    llm_results_for_report['action_items_status'] = "Failed: Prompt tokenization error."
                    llm_results_for_report['action_items_data'] = "Error: Prompt tokenization failed."
                elif action_items_raw_output == GENERATION_OOM_ERROR_INDICATOR:
                    logger.error("Out of memory during LLM action item extraction. Skipping.")
                    llm_results_for_report['action_items_status'] = "Failed: Generation out of memory."
                    llm_results_for_report['action_items_data'] = "Error: Generation out of memory."
                elif action_items_raw_output:
                    try:
                        # Attempt to strip markdown code block if present
                        stripped_output = action_items_raw_output.strip()
                        if stripped_output.startswith("```json"):
                            stripped_output = stripped_output[len("```json"):].strip()
                        if stripped_output.startswith("```"): # General code block
                             stripped_output = stripped_output[3:].strip()
                        if stripped_output.endswith("```"):
                            stripped_output = stripped_output[:-3].strip()
                        
                        parsed_json_actions = json.loads(stripped_output)
                        llm_results_for_report['action_items_data'] = parsed_json_actions
                        llm_results_for_report['action_items_status'] = "Completed (JSON parsed)."
                        logger.info("Action item extraction by LLM successful and parsed as JSON.")
                    except json.JSONDecodeError:
                        logger.warning(f"Initial JSON parse failed for action items. Raw output was: {action_items_raw_output[:300]}...") # Log snippet
                        # Attempt regex extraction as fallback
                        try:
                            # re is imported at the top of the file
                            match = re.search(r'\[\s*(\{.*?\}\s*(,\s*\{.*?\})*\s*)?\]', stripped_output, re.DOTALL)
                            if match:
                                potential_json_str = match.group(0)
                                logger.debug(f"Regex found potential JSON: {potential_json_str[:300]}...")
                                parsed_json_actions = json.loads(potential_json_str)
                                llm_results_for_report['action_items_data'] = parsed_json_actions
                                llm_results_for_report['action_items_status'] = "Completed (JSON parsed with regex fallback)."
                                logger.info("Action item extraction by LLM successful using regex fallback parsing.")
                            else:
                                logger.warning("Regex did not find a JSON list structure. Storing raw output.")
                                llm_results_for_report['action_items_data'] = action_items_raw_output # Store raw string
                                llm_results_for_report['action_items_status'] = "Completed (raw output, JSON parse and regex failed)."
                        except json.JSONDecodeError as e_re_json:
                            logger.warning(f"JSON parsing after regex extraction failed: {e_re_json}. Storing raw output.")
                            llm_results_for_report['action_items_data'] = action_items_raw_output # Store raw string
                            llm_results_for_report['action_items_status'] = "Completed (raw output, regex found but JSON invalid)."
                        except Exception as e_re: # Catch other regex/parsing errors
                            logger.error(f"Error during regex-based JSON extraction: {e_re}. Storing raw output.")
                            llm_results_for_report['action_items_data'] = action_items_raw_output
                            llm_results_for_report['action_items_status'] = "Completed (raw output, error in regex processing)."
                        llm_results_for_report['action_items_data'] = action_items_raw_output # Store raw string
                        llm_results_for_report['action_items_status'] = "Completed (raw output, JSON parse failed)."
                else: 
                    logger.warning("LLM action item extraction did not return content or encountered an unspecified error.")
                    llm_results_for_report['action_items_status'] = "Failed or no content returned."
            elif args.llm_action_items and not transcript_for_llm_and_report:
                llm_results_for_report['action_items_status'] = "Skipped: No transcript content."
            
            # --- Generate and Save the Combined Report ---
            # This check is important: only generate report if there was a transcript to process initially.
            # transcript_for_llm_and_report might be the original or corrected one.
            # llm_results_for_report['corrected_text_available'] indicates if correction was successful.
            # The base_transcript_for_report for generate_llm_report_content should be the final version.
            if transcript_for_llm_and_report: # Check if we actually have content for the report's transcript section
                report_content_str = generate_llm_report_content(args, llm_results_for_report, transcript_for_llm_and_report)
                report_filename_suffix = f"_llm_report.{args.llm_report_format}"
                # Ensure report goes to the correct LLM output directory
                report_full_filename = get_unique_filename(os.path.join(args.actual_llm_output_dir, f"{output_prefix}{report_filename_suffix}"))
                try:
                    with open(report_full_filename, 'w', encoding='utf-8') as f:
                        f.write(report_content_str)
                    logger.info(f"Combined LLM report saved to: {report_full_filename}")
                except IOError as e:
                    logger.error(f"Failed to save combined LLM report: {e}")
            else: # This case would typically mean the initial transcript_for_llm_and_report was empty or reading failed
                 logger.info("Skipping combined LLM report generation as there was no base transcript content to process.")

        elif llm_processing_active and not transcript_for_llm_and_report:
             logger.warning("LLM processing was active, but no transcript content was available (e.g., file read failed or empty). Skipping LLM tasks and report.")

        logger.info(f"--- LLM Post-Processing for Combined Report Finished ---")
    
    elif llm_processing_active and (not llm_model or not llm_tokenizer): # LLM tasks enabled but model didn't load
        logger.warning("LLM tasks were requested, but the LLM model failed to load. Skipping all LLM post-processing.")
    
    # Clean up LLM model when done with all processing
    if llm_model is not None or llm_tokenizer is not None:
        logger.info("Cleaning up LLM model resources...")
        del llm_model, llm_tokenizer
        llm_model, llm_tokenizer = None, None # Ensure they are None
        gc.collect()
        logger.debug("LLM resources released.")
    
    end_time_total = time.time()
    output_dirs_msg = f"'{args.output_dir}'"
    if any_llm_task_enabled and args.actual_llm_output_dir != args.output_dir:
        output_dirs_msg = f"'{args.output_dir}' (transcript) and '{args.actual_llm_output_dir}' (LLM outputs)"
    
    logger.info(f"Processing of '{args.input_audio}' complete in {timedelta(seconds=int(end_time_total - start_time_total))}. Outputs are in {output_dirs_msg}")

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

    # LLM Post-processing arguments
    parser.add_argument("--llm_correct", action="store_true", help="Enable LLM-based transcript correction. Requires --llm_model_id or default.")
    parser.add_argument("--llm_summarize", action="store_true", help="Enable LLM-based summary generation. Requires --llm_model_id or default.")
    parser.add_argument("--llm_action_items", action="store_true", help="Enable LLM-based action item extraction. Requires --llm_model_id or default.")
    parser.add_argument("--llm_model_id", type=str, default=None, 
                        help=f"Hugging Face model ID for the MLX-compatible local LLM. "
                             f"If LLM tasks are enabled and this is not specified, "
                             f"defaults to: {DEFAULT_LLM_MODEL_ID}. "
                             f"Example: 'mlx-community/Phi-3-mini-4k-instruct-4bit'.")
    parser.add_argument("--llm_output_dir", type=str, default=None,
                        help="Directory for LLM-generated output files. Defaults to --output_dir if not specified.")
    parser.add_argument("--llm_max_tokens_summary", type=int, default=500,
                        help="Maximum number of tokens for the LLM to generate for summaries.")
    parser.add_argument("--llm_max_tokens_correction", type=int, default=0, # 0 means use default in script
                        help="Maximum number of new tokens for LLM correction. If 0, a script default is used (e.g., based on input or a large fixed value).")
    parser.add_argument("--llm_max_tokens_action_items", type=int, default=1000,
                        help="Maximum number of tokens for LLM action item extraction.")
    parser.add_argument("--llm_report_format", type=str, default="txt", choices=["txt", "md"],
                        help="Format for the combined LLM post-processing report file ('txt' or 'md'). Default: 'txt'.")

    args = parser.parse_args()

    # LLM Argument Validation and Setup
    args.actual_llm_output_dir = args.llm_output_dir if args.llm_output_dir else args.output_dir
    if args.actual_llm_output_dir != args.output_dir: # Ensure LLM output dir exists if different
        os.makedirs(args.actual_llm_output_dir, exist_ok=True)
    
    # Configure logging levels based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        for handler in logging.getLogger().handlers: # Ensure all handlers respect the new level
            handler.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled by command line flag.")
        # Enable more verbose logging from underlying libraries if in debug mode
        logging.getLogger("pyannote").setLevel(logging.DEBUG)
        logging.getLogger("speechbrain").setLevel(logging.DEBUG if os.getenv("SPEECHBRAIN_DEBUG") else logging.INFO) 
        logging.getLogger("whisperx").setLevel(logging.DEBUG)
    else:
        # Keep third-party libraries quieter by default
        logging.getLogger("pyannote").setLevel(logging.WARNING)
        logging.getLogger("speechbrain").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING) 
        logging.getLogger("fsspec").setLevel(logging.WARNING)
        logging.getLogger("whisperx").setLevel(logging.INFO)


    # Ensure WHISPERX_PROCESSOR_HF_TOKEN is available if HF_TOKEN is set and the former is not
    if os.getenv("HF_TOKEN") and not os.getenv("WHISPERX_PROCESSOR_HF_TOKEN"):
        os.environ["WHISPERX_PROCESSOR_HF_TOKEN"] = os.environ["HF_TOKEN"]
        logger.debug("Using HF_TOKEN for WHISPERX_PROCESSOR_HF_TOKEN as it was not explicitly set.")

    process_audio_mlx(args)