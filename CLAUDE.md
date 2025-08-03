# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a multilingual ASR (Automatic Speech Recognition) project that provides real-time speech-to-text transcription using different approaches and models. The project contains multiple transcription scripts that demonstrate progressive improvements in architecture and capabilities.

## Key Commands

### Development Environment Setup

```bash
# Install dependencies using uv (Python package manager)
uv sync

# Install dependencies manually if uv is not available
pip install -r requirements.txt  # if exists, or install from pyproject.toml
```

### Running the Scripts

```bash
# Run the most advanced transcription script with CLI options
python scripts/even_better_whisper_transcribe.py
python scripts/even_better_whisper_transcribe.py --debug
python scripts/even_better_whisper_transcribe.py --output my_transcript.srt

# Run the basic version
python scripts/better_whisper_transcribe.py

# Run the NeMo-based transcription
python scripts/nemo_parakeet_tdt_ctc_transcribe.py

# Run the windowing approach
python scripts/windowing_transcribe.py

# Extract text from SRT files (uses typer CLI framework)
uv run python extract_text_from_srt.py <srt_file>  # Outputs to <srt_file>.txt by default
uv run python extract_text_from_srt.py --help  # Show help
uv run python extract_text_from_srt.py <srt_file> --output <output_file>  # Save to custom file
```

### Linting and Code Quality

```bash
# Run linting with ruff (configured in pyproject.toml)
ruff check .
ruff format .

# The project uses extensive ruff configuration with many rule sets enabled
# Check pyproject.toml for full linting configuration
```

### Testing

```bash
# No specific test framework is configured in this project
# Tests would need to be added if implementing new features
```

## Architecture Overview

### Core Components

1. **Transcription Scripts** (`scripts/`):

   - `even_better_whisper_transcribe.py`: Most advanced implementation with dual audio input (speaker + microphone), CLI interface, debug logging, and robust error handling
   - `better_whisper_transcribe.py`: Improved version with VAD, LUFS normalization, and class-based architecture
   - `nemo_parakeet_tdt_ctc_transcribe.py`: NeMo-based ASR implementation using Parakeet TDT CTC model
   - `windowing_transcribe.py`: Segmented approach using fixed-time windows

2. **Utility Scripts**:
   - `extract_text_from_srt.py`: Extracts only the transcribed text from SRT subtitle files using typer CLI framework

3. **Key Classes and Modules**:
   - `TranscriptionEngine`: Handles audio normalization and Whisper model inference
   - `AudioSegmentBuffer`: Manages audio frame buffering and timing
   - `DebugManager` & `AudioStatsLogger`: Debug and monitoring utilities (even_better version)

### Processing Pipeline

1. **Audio Capture**: Multi-threaded audio capture from speaker loopback and/or microphone
2. **Audio Mixing**: Combines multiple audio sources with level matching and normalization
3. **VAD Processing**: Voice Activity Detection to segment speech from silence
4. **Audio Preprocessing**: LUFS normalization and peak limiting
5. **ASR Inference**: faster-whisper or NeMo model transcription
6. **Output Generation**: Live console display and SRT file generation

### Threading Architecture

The most advanced implementation (`even_better_whisper_transcribe.py`) uses:

- **SpeakerCaptureThread**: Captures loopback audio from speakers
- **MicrophoneCaptureThread**: Captures audio from physical microphone
- **AudioMixerThread**: Combines and normalizes the two audio streams
- **TranscriptionWorkflowThread**: Handles VAD, buffering, and transcription

## Key Dependencies

- `faster-whisper>=1.1.1`: Main ASR engine
- `nemo-toolkit[asr]>=2.3`: Alternative ASR engine for NeMo-based scripts
- `torch>=2.7.0`: Deep learning framework
- `soundcard>=0.4.4`: Audio capture from system devices
- `webrtcvad>=2.0.10`: Voice Activity Detection
- `pyloudnorm>=0.1.1`: Audio loudness normalization
- `typer>=0.15.3`: CLI framework (for even_better version and utility scripts)
- `rich>=14.0.0`: Enhanced logging and console output

## Configuration Constants

Key configuration is centralized in constants at the top of each script:

- `SAMPLE_RATE_HZ = 16000`: Audio sampling rate
- `FRAME_DURATION_MS = 10`: Audio frame duration
- `VAD_AGGRESSIVENESS_LEVEL = 3`: Voice activity detection sensitivity
- `TARGET_LUFS = -14.0`: Target loudness level for normalization
- `LIVE_MIN_TRANSCRIPTION_DURATION_SEC = 1.0`: Minimum duration for live transcription
- `FILE_MIN_TRANSCRIPTION_DURATION_SEC = 1.5`: Minimum duration for file output

## Audio Device Handling

The project automatically detects and uses available audio devices:

- **Speaker Loopback**: Captures system audio output (what you hear)
- **Physical Microphone**: Captures microphone input
- **Fallback Logic**: Can operate with either source if the other is unavailable

## Error Handling Patterns

- **Graceful Degradation**: Scripts continue operation even if one audio source fails
- **Resource Cleanup**: Proper COM initialization/cleanup for Windows audio
- **Signal Handling**: SIGINT/SIGTERM handlers for graceful shutdown
- **Thread Management**: Timeout-based thread joining with proper cleanup

## Development Guidelines

1. **Audio Processing**: Always normalize audio using LUFS + peak limiting before transcription
2. **Threading**: Use queue-based communication between threads with proper timeout handling
3. **Error Resilience**: Implement consecutive error counting and graceful degradation
4. **Platform Compatibility**: Handle Windows-specific COM initialization for audio capture
5. **Configuration**: Use constants for all timing and threshold values
6. **Logging**: Use rich logging with appropriate levels and context
7. **CLI Interface**: Use typer for command-line interfaces with proper help and argument handling

## File Structure Notes

- `scripts/`: Contains all transcription implementations
- `doc/agent/`: Contains detailed analysis and review documentation
- `transcript.srt`: Default output file for SRT subtitles
- `debug_test.srt`: Debug output file

## Common Issues and Solutions

1. **CUDA Not Available**: Scripts will fall back to CPU processing with warnings
2. **Audio Device Not Found**: Check Windows audio settings and device availability
3. **COM Initialization Errors**: Ensure proper Windows COM setup in audio threads
4. **Memory Issues**: Monitor GPU memory usage during long transcription sessions
5. **Thread Synchronization**: Use proper stop events and timeout-based queue operations

## Performance Considerations

- GPU acceleration is preferred but not required
- Audio processing is CPU-intensive, especially with multiple streams
- Memory usage scales with buffer sizes and model size
- Real-time performance depends on GPU capability and audio complexity
