# MLX Whisper Transcriber with Diarization and LLM Post-Processing

This Python script, `transcriber_mlx.py`, provides fast audio transcription using MLX Whisper models optimized for Apple Silicon, with optional speaker diarization capabilities leveraging WhisperX and `pyannote.audio`. It supports various input audio/video formats (thanks to FFMPEG) and outputs transcripts in SRT, CSV, and TXT formats. Additionally, it offers optional LLM-based post-processing to correct transcripts, generate summaries, and extract action items using local Gemma models via MLX. Includes optional GPU acceleration for diarization on Apple Silicon via PyTorch MPS.

## Features

* **Fast Transcription:** Utilizes `mlx-whisper` for efficient transcription on Apple Silicon (M1, M2, M3 series chips).
* **Speaker Diarization:** Identifies different speakers in the audio and assigns speaker labels to transcript segments (using WhisperX, which relies on `pyannote.audio`).
* **Word-Level Timestamps:** Provides timestamps for individual words in the transcript.
* **LLM Post-Processing:** Optional post-processing of transcripts using local LLMs (Gemma/MLX):
    * **Transcript Correction:** Enhances transcript quality by fixing grammar, punctuation, and readability while preserving speaker labels.
    * **Summary Generation:** Creates concise meeting summaries that highlight main topics, decisions, and outcomes.
    * **Action Item Extraction:** Identifies and structures action items with task descriptions, assignees, and deadlines.
    * **Combined Report:** Consolidates all LLM outputs into a single, well-formatted report file.
* **Multiple Output Formats:**
    * `.srt`: SubRip subtitle format, including speaker labels.
    * `.csv`: Comma-separated values with columns for segment ID, word ID, speaker, start time, end time, and text.
    * `.txt`: Plain text transcript, with speaker changes indicated.
    * `.txt` or `.md`: Combined LLM post-processing report (includes summary, action items, and transcript).
* **Configurable:**
    * Choose different MLX Whisper models.
    * Specify audio language (or auto-detect).
    * Set Hugging Face token for diarization models.
    * Control minimum/maximum number of speakers for diarization.
    * Choose CPU or MPS (Apple Silicon GPU) for diarization (**Experimental**).
* **Broad Format Support:** Processes a wide range of audio and video file formats if FFMPEG is installed.
* **Environment Variable Configuration:** Supports configuration via a `.env` file or environment variables for common settings.
* **Heartbeat Logging:** Provides "still alive" messages during long diarization processes.

## Dependencies

### System Dependencies:

* **FFMPEG:** Required for processing most audio/video files (especially MP4, MOV, etc.) to extract the audio stream.
    * On macOS: `brew install ffmpeg`
    * On Debian/Ubuntu: `sudo apt update && sudo apt install ffmpeg`

### Python Dependencies:

* Python 3.8+
* See `requirements.txt` file. Key dependencies include:
    * `mlx`
    * `mlx-whisper`
    * `whisperx` (installs `pyannote.audio`, `torch`, etc.)
    * `pandas`
    * `python-dotenv` (optional)
    * `torch`
    * `mlx-lm` (for LLM post-processing)
* **Installation:** You can typically install the Python packages using pip:
    ```bash
    # Install dependencies from requirements file (recommended)
    pip install -r requirements.txt

    # Or manually install key components:
    # pip install mlx mlx-whisper pandas python-dotenv torch mlx-lm
    # pip install git+https://github.com/m-bain/whisperX.git@main#egg=whisperx
    ```
* **Environment:** Ensure you are in an environment where `mlx` can leverage your Apple Silicon's GPU/ANE and PyTorch (for MPS) is correctly installed for your macOS version.

## Setup

1.  **Install FFMPEG:** Follow the instructions under System Dependencies for your operating system.
2.  **Install Python Dependencies:** Use pip with the `requirements.txt` file or install manually as shown above.
3.  **Hugging Face Token & Model Agreements (for Diarization):**
    * Speaker diarization models (`pyannote.audio`) require authentication with Hugging Face.
    * **Create Account:** Sign up at [Hugging Face](https://huggingface.co/).
    * **Accept User Conditions:** Visit the Hugging Face model pages for `pyannote/segmentation-3.0` and `pyannote/speaker-diarization-3.1` (or the specific models used) and accept their user conditions. You must be logged in.
    * **Generate Token:** Go to your Hugging Face account settings -> Access Tokens -> New token (give it "read" permissions).
    * **Provide Token:** Make the token available to the script in one of these ways:
        * Use the `--hf_token YOUR_TOKEN` command-line argument.
        * Set the `HF_TOKEN` environment variable (e.g., `export HF_TOKEN="YOUR_TOKEN"`).
        * Set `WHISPERX_PROCESSOR_HF_TOKEN="YOUR_TOKEN"` in a `.env` file in the script's directory.

## Configuration

The script can be configured using a hierarchy (command-line overrides environment variables, which override `.env` file):

1.  **Command-Line Arguments:** See `--help` for all options.
    ```bash
    python transcriber_mlx.py --help
    ```
2.  **Environment Variables:**
    * `WHISPERX_PROCESSOR_MODEL_NAME`: Default Whisper model (default: `mlx-community/whisper-small-mlx`).
    * `WHISPERX_PROCESSOR_LANGUAGE`: Default language (default: `en`).
    * `WHISPERX_PROCESSOR_HF_TOKEN` or `HF_TOKEN`: Hugging Face token.
    * `WHISPERX_PROCESSOR_MODEL_DOWNLOAD_ROOT`: Root directory for model downloads.
    * `WHISPERX_PROCESSOR_OUTPUT_DIR`: Default output directory (default: `.`).
    * `WHISPERX_PROCESSOR_DIARIZATION_DEVICE`: Default diarization device (default: `mps`).
3.  **`.env` File:** Create a `.env` file in the script's directory (requires `python-dotenv` installed):
    ```env
    WHISPERX_PROCESSOR_MODEL_NAME="mlx-community/whisper-small-mlx"
    WHISPERX_PROCESSOR_LANGUAGE="en"
    HF_TOKEN="your_hugging_face_token_here"
    WHISPERX_PROCESSOR_OUTPUT_DIR="."
    WHISPERX_PROCESSOR_DIARIZATION_DEVICE="mps"
    ```

## Usage

The basic command structure is:

```bash
python transcriber_mlx.py <input_audio_file> [options]
```

### Examples:

* **Transcribe & Diarize using CPU:**
    ```bash
    python transcriber_mlx.py "path/to/your/video.mp4" --diarize --diarization_device cpu
    ```
    *(Assumes HF_TOKEN is set in environment or `.env`)*

* **Transcribe & Diarize using MPS (Apple Silicon GPU - Experimental):**
    ```bash
    python transcriber_mlx.py "audio.wav" --diarize --diarization_device mps
    ```

* **Transcribe with specific model & speaker count hints:**
    ```bash
    python transcriber_mlx.py "meeting.mp3" --model_name "mlx-community/whisper-medium-mlx" --diarize --diarization_device mps --min_speakers 3 --max_speakers 6
    ```

* **Transcribe only (no diarization) and save only SRT:**
    ```bash
    python transcriber_mlx.py "lecture.m4a" --output_dir ./transcripts --no_csv --no_txt
    ```

* **Transcribe and apply all LLM post-processing:**
    ```bash
    python transcriber_mlx.py "meeting.mp3" --llm_correct --llm_summarize --llm_action_items
    ```
    (Uses default LLM model automatically - no need to specify `--llm_model_id`)

* **Generate a markdown-formatted combined report:**
    ```bash
    python transcriber_mlx.py "meeting.mp3" --llm_model_id "mlx-community/Gemma-2-2B-it-mlx" --llm_summarize --llm_action_items --llm_report_format md
    ```

### Key Command-Line Arguments:

* `input_audio`: Path to the input audio/video file (required).
* `--output_dir`: Directory to save output files (default: `.`).
* `--model_name`: MLX Whisper model from Hugging Face or local path (default: `mlx-community/whisper-small-mlx`).
* `--language`: Language code (`en`, `es`, `auto`, etc.) (default: `en`).
* `--initial_prompt`: Optional text to provide as context/prompt to improve ASR accuracy for specific words or context.
* `--print_progress`: Print transcription progress from the MLX whisper library.
* `--debug`: Enable debug level logging for more detailed output, including pyannote logs and attribute inspection.
* `--diarize`: Enable speaker diarization (uses pyannote.audio via whisperx).
* `--hf_token`: Hugging Face token for diarization models (can also use environment variables).
* `--min_speakers`: Min speakers for diarization.
* `--max_speakers`: Max speakers for diarization.
* `--diarization_device`: Device for diarization: `cpu` or `mps` (default: `mps`).

* `--no_csv`, `--no_srt`, `--no_txt`: Flags to disable specific output file formats (all enabled by default).

**LLM Post-Processing Arguments:**
* `--llm_model_id`: Hugging Face model ID for the MLX-compatible local LLM. Optional - if not specified, defaults to `mlx-community/gemma-3-4b-it-qat-4bit`.
* `--llm_correct`: Enable LLM-based transcript correction.
* `--llm_summarize`: Enable LLM-based summary generation.
* `--llm_action_items`: Enable LLM-based action item extraction.
* `--llm_output_dir`: Directory for LLM-generated output files (defaults to `--output_dir` if not specified).
* `--llm_max_tokens_summary`: Maximum tokens for LLM summary generation (default: 500).
* `--llm_max_tokens_correction`: Maximum new tokens for LLM correction (default: 8000).
* `--llm_max_tokens_action_items`: Maximum tokens for action item extraction (default: 1000).
* `--llm_report_format`: Format for the LLM report file. Options: 'txt' or 'md' (default: 'md').
* `--llm_max_tokens_action_items`: Maximum tokens for LLM action item extraction (default: 1000).
* `--llm_report_format`: Format for the combined LLM post-processing report ('txt' or 'md') (default: 'txt').

*(Run `python transcriber_mlx.py --help` for the full list and default values.)*

## Output Files

For an input file like `my_audio.mp4`, the script will generate (by default):

* `my_audio.srt` or `my_audio_diarized.srt`: Subtitle file with speaker labels if diarized.
* `my_audio.csv` or `my_audio_diarized.csv`: Detailed CSV with word timings and speaker labels.
* `my_audio.txt` or `my_audio_diarized.txt`: Plain text transcript with speaker labels.

When LLM post-processing is enabled, the following file(s) will be generated:

* `my_audio_llm_report.txt` or `my_audio_llm_report.md`: Combined LLM report containing summary (if enabled), action items (if enabled), and the full transcript (corrected if enabled). The format depends on the `--llm_report_format` option.

The `_diarized` suffix is added if diarization was successfully applied. Existing files are not overwritten; a counter (`_1`, `_2`, etc.) is appended to the new filename.

## LLM Post-Processing

This tool leverages local MLX-compatible LLMs to enhance your transcripts through several post-processing tasks:

### Features

- **Transcript Correction:** Enhances transcript quality by fixing grammar, punctuation, and readability while preserving speaker labels.
- **Summary Generation:** Creates concise summaries of meeting content, with advanced chunking for long transcripts.
- **Action Item Extraction:** Identifies and formats action items from the meeting in a structured list.
- **Combined Report:** Consolidates all LLM outputs into a single, well-formatted report file.

### Default LLM Model

The system now comes with a default LLM model that will be automatically used when LLM features are enabled (no need to specify `--llm_model_id`):

- **Default Model:** `mlx-community/gemma-3-4b-it-qat-4bit`
- **Context Length:** 128k tokens
- **Description:** A 4-bit quantized version of Google's Gemma 3 4B Instruct model, optimized for MLX. This model offers an excellent balance of quality and performance, with a large context window (128k tokens) that can handle most transcripts without chunking.
- **Download Size:** ~2.5GB

### Alternative Recommended LLMs

While the default model works well for most cases, you can specify an alternative model using the `--llm_model_id` argument:

| Model | Context Window | Size | Quality | Notes |
|-------|---------------|------|---------|-------|
| `mlx-community/gemma-3-4b-it-qat-4bit` | 128k tokens | ~2.5GB | Very Good | Default model - balanced quality and performance |
| `mlx-community/Phi-3-mini-4k-instruct-4bit` | 4k tokens | ~1.8GB | Good | Smaller context but faster, good for shorter transcripts |
| `mlx-community/Llama-3.1-8B-Instruct-4bit` | 128k tokens | ~4.5GB | Excellent | Higher quality but larger download and more RAM needed |
| `mlx-community/Mistral-7B-Instruct-v0.3-4bit` | 32k tokens | ~4GB | Very Good | Good alternative with medium context window |

All models use 4-bit quantization to optimize for Apple Silicon memory use.

### Choosing an LLM

- For **shorter transcripts** (under 30 minutes): `Phi-3-mini-4k-instruct-4bit` is a good lightweight option
- For **general use**: The default `gemma-3-4b-it-qat-4bit` works well for most transcripts
- For **best quality** on important meetings: `Llama-3.1-8B-Instruct-4bit` offers higher quality output
- For **systems with limited RAM** (8GB Mac): Consider the smaller models

### Getting Started with Local LLMs

1. **No configuration needed for default model** - just use `--llm_correct`, `--llm_summarize` or `--llm_action_items`
2. **First run will download the model** (one-time setup) - this may take a few minutes depending on your connection
3. Models are stored in the Hugging Face cache directory (typically `~/.cache/huggingface/` on macOS)

### Handling Long Transcripts

The system now has advanced chunking for long transcripts:

- **Transcript correction** will be skipped if it exceeds the model's context window
- **Summary generation** uses a map-reduce approach:
  1. Long transcripts are split into overlapping chunks based on token count
  2. Each chunk is summarized independently
  3. The chunk summaries are combined and summarized again (if they fit the context window)
  4. If chunk summaries are too large for a final pass, they are concatenated as the final output
- **Action item extraction** will be skipped if the transcript exceeds the model's context window

The LLM report clearly indicates the processing status of each task, including details about chunking or skipping.

## Troubleshooting & Notes

* **Long Processing Times (CPU Diarization):** Diarization on the CPU can be very slow for long files. Use `--diarization_device mps` on Apple Silicon for significant speedups.
* **MPS Issues:** Using `--diarization_device mps` is experimental. If it fails or produces errors, fall back to `--diarization_device cpu`. Ensure your PyTorch installation supports MPS.
* **FFMPEG is Crucial:** Ensure FFMPEG is correctly installed and in your system's PATH for processing video files or many audio formats.
* **Memory Usage:** Larger Whisper models and long audio files consume significant RAM. Diarization also requires substantial memory.
* **LLM Context Window:** Long transcripts may exceed some models' context windows. The system will automatically chunk or skip tasks as needed, but choosing a model with a larger context (like the default `gemma-3-4b-it-qat-4bit`) is recommended for longer recordings.
* **First LLM Run:** The first time you use LLM features, the model will be downloaded, which may take several minutes depending on your connection speed.
* **Hugging Face Token/Permissions:** Diarization will fail without a valid Hugging Face token *and* if you haven't accepted the terms for the specific `pyannote.audio` models on the Hugging Face website.
* **Diarization Accuracy:** Automated diarization accuracy varies. Using `--min_speakers` and `--max_speakers` can help. Manual correction may be needed for high accuracy (use subtitle editors like Aegisub, etc.).
* **Heartbeat Logging:** During diarization, messages like "Diarization still in progress..." will appear every 30 seconds to indicate the process hasn't hung, especially useful for long CPU runs.

