# MLX Whisper Transcriber with Diarization

This Python script, `transcriber_mlx.py`, provides fast audio transcription using MLX Whisper models optimized for Apple Silicon, with optional speaker diarization capabilities leveraging WhisperX and `pyannote.audio`. It supports various input audio/video formats (thanks to FFMPEG) and outputs transcripts in SRT, CSV, and TXT formats. Includes optional GPU acceleration for diarization on Apple Silicon via PyTorch MPS.

## Features

* **Fast Transcription:** Utilizes `mlx-whisper` for efficient transcription on Apple Silicon (M1, M2, M3 series chips).
* **Speaker Diarization:** Identifies different speakers in the audio and assigns speaker labels to transcript segments (using WhisperX, which relies on `pyannote.audio`).
* **Word-Level Timestamps:** Provides timestamps for individual words in the transcript.
* **Multiple Output Formats:**
    * `.srt`: SubRip subtitle format, including speaker labels.
    * `.csv`: Comma-separated values with columns for segment ID, word ID, speaker, start time, end time, and text.
    * `.txt`: Plain text transcript, with speaker changes indicated.
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
* **Installation:** You can typically install the Python packages using pip:
    ```bash
    # Install dependencies from requirements file (recommended)
    pip install -r requirements.txt

    # Or manually install key components:
    # pip install mlx mlx-whisper pandas python-dotenv torch
    # pip install git+[https://github.com/m-bain/whisperX.git@main#egg=whisperx](https://github.com/m-bain/whisperX.git@main#egg=whisperx)
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
    * `WHISPERX_PROCESSOR_MODEL_NAME`: Default Whisper model (e.g., `mlx-community/whisper-medium-mlx`).
    * `WHISPERX_PROCESSOR_LANGUAGE`: Default language (e.g., `en`).
    * `WHISPERX_PROCESSOR_HF_TOKEN` or `HF_TOKEN`: Hugging Face token.
    * `WHISPERX_PROCESSOR_OUTPUT_DIR`: Default output directory (e.g., `./transcripts`).
    * `WHISPERX_PROCESSOR_DIARIZATION_DEVICE`: Default diarization device (`cpu` or `mps`).
3.  **`.env` File:** Create a `.env` file in the script's directory (requires `python-dotenv` installed):
    ```env
    WHISPERX_PROCESSOR_MODEL_NAME="mlx-community/whisper-large-mlx"
    WHISPERX_PROCESSOR_LANGUAGE="en"
    HF_TOKEN="your_hugging_face_token_here"
    WHISPERX_PROCESSOR_OUTPUT_DIR="./my_transcripts"
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

### Key Command-Line Arguments:

* `input_audio`: Path to the input audio/video file (required).
* `--output_dir`: Directory to save output files.
* `--model_name`: MLX Whisper model (HF ID or local path).
* `--language`: Language code (`en`, `es`, `auto`, etc.).
* `--diarize`: Enable speaker diarization.
* `--hf_token`: Your Hugging Face token (required for diarization if not set elsewhere).
* `--min_speakers`, `--max_speakers`: Optionally guide the number of speakers for diarization.
* `--diarization_device`: Choose `cpu` or `mps` (default: `cpu`). MPS is experimental.
* `--no_csv`, `--no_srt`, `--no_txt`: Flags to disable specific output file formats.
* `--debug`: Enable verbose debug logging.
* `--initial_prompt`: Optional text to provide as context/prompt to the transcription model.
* `--print_progress`: Show transcription progress if supported by `mlx-whisper`.

*(Run `python transcriber_mlx.py --help` for the full list and default values.)*

## Output Files

For an input file like `my_audio.mp4`, the script will generate (by default):

* `my_audio.srt` or `my_audio_diarized.srt`: Subtitle file with speaker labels if diarized.
* `my_audio.csv` or `my_audio_diarized.csv`: Detailed CSV with word timings and speaker labels.
* `my_audio.txt` or `my_audio_diarized.txt`: Plain text transcript with speaker labels.

The `_diarized` suffix is added if diarization was successfully applied. Existing files are not overwritten; a counter (`_1`, `_2`, etc.) is appended to the new filename.

## Troubleshooting & Notes

* **Long Processing Times (CPU Diarization):** Diarization on the CPU can be very slow for long files. Use `--diarization_device mps` on Apple Silicon for significant speedups.
* **MPS Issues:** Using `--diarization_device mps` is experimental. If it fails or produces errors, fall back to `--diarization_device cpu`. Ensure your PyTorch installation supports MPS.
* **FFMPEG is Crucial:** Ensure FFMPEG is correctly installed and in your system's PATH for processing video files or many audio formats.
* **Memory Usage:** Larger Whisper models and long audio files consume significant RAM. Diarization also requires substantial memory.
* **Hugging Face Token/Permissions:** Diarization will fail without a valid Hugging Face token *and* if you haven't accepted the terms for the specific `pyannote.audio` models on the Hugging Face website.
* **Diarization Accuracy:** Automated diarization accuracy varies. Using `--min_speakers` and `--max_speakers` can help. Manual correction may be needed for high accuracy (use subtitle editors like Aegisub, etc.).
* **Heartbeat Log:** During diarization, messages like "Diarization still in progress..." will appear every 30 seconds to indicate the process hasn't hung, especially useful for long CPU runs.
```
