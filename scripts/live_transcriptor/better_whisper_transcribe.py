"""Realtime ASR with faster-whisper (v3) — live console and SRT output.

Live console driven by VAD + timing (natural or forced).
SRT file output on VAD-detected silence or forced boundaries.
"""

import ctypes
import logging
import os
import queue
import signal
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:

    class MicrophoneProtocol(Protocol):
        name: str
        isloopback: bool

        def recorder(self, samplerate: int, blocksize: int): ...


import numpy as np
import pyloudnorm as pyln
import soundcard as sc
import torch
import typer
import webrtcvad
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment
from rich.logging import RichHandler

if hasattr(sc, "SoundcardRuntimeWarning"):
    warnings.filterwarnings(
        "ignore",
        category=sc.SoundcardRuntimeWarning,
        message="data discontinuity in recording",
    )

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Possible clipped samples in output.",
)

# ── CONFIGURATION CONSTANTS ─────────────────────────────────
# Audio processing
SAMPLE_RATE_HZ = 16_000
FRAME_DURATION_MS = 10
FRAME_SIZE_SAMPLES = int(SAMPLE_RATE_HZ * (FRAME_DURATION_MS / 1000.0))
TARGET_LUFS = -14.0

# Transcription behavior
LIVE_MIN_TRANSCRIPTION_DURATION_SEC = 1.0
FILE_MIN_TRANSCRIPTION_DURATION_SEC = 1.5

# VAD (Voice Activity Detection)
VAD_AGGRESSIVENESS_LEVEL = 3  # Range: 0 (least aggressive) to 3 (most aggressive)
VAD_LIVE_SILENCE_DURATION_SEC = 0.2
VAD_LIVE_SILENCE_FRAMES = int(
    VAD_LIVE_SILENCE_DURATION_SEC * (1000 / FRAME_DURATION_MS)
)
VAD_FILE_SILENCE_DURATION_SEC = 0.3  # Silence duration to trigger file segment commit
VAD_FILE_SILENCE_FRAMES = int(
    VAD_FILE_SILENCE_DURATION_SEC * (1000 / FRAME_DURATION_MS)
)

# Output
DEFAULT_SRT_FILE_PATH = Path("transcript.srt")

# Magic values
MINIMUM_AUDIO_AMPLITUDE_THRESHOLD = 0.7
STEREO_CHANNEL_COUNT = 2
MONO_CHANNEL_COUNT = 1
DIM_ARRAY_2D = 2

# Threading and Queues
AUDIO_FRAME_QUEUE = queue.Queue()  # Queue for PCM audio frames
STOP_EVENT = threading.Event()

# Platform detection
WINDOWS_PLATFORM = "win32"

# Windows COM initialization for audio capture thread
COINIT_APARTMENTTHREADED = 0x2


# ── LOGGING SETUP ───────────────────────────────────────────
def setup_logging() -> None:
    """Configures logging with RichHandler for better console output."""
    logging.basicConfig(
        level=logging.INFO,
        handlers=[RichHandler(rich_tracebacks=True)],
    )


# Logger setup
setup_logging()
logger = logging.getLogger(__name__)


# ── RESOURCE INITIALIZATION ────────────────────────────────
def initialize_transcription_resources(
    model_name: str,
) -> tuple[WhisperModel, webrtcvad.Vad, pyln.Meter]:
    """Loads and initializes Whisper model, VAD, and loudness meter."""
    if not torch.cuda.is_available():
        msg = "CUDA is not available. Please use a GPU."
        raise ValueError(msg)

    device_type = "cuda"

    logger.info(f"📦 Loading Whisper {model_name} model (device: {device_type})...")
    model = WhisperModel(
        model_name,
        device=device_type,
        compute_type="int8_float16"
        if device_type == "cuda"
        else "int8",  # common optimization
    )
    logger.info("🗣️ Initializing VAD...")
    vad_detector = webrtcvad.Vad(VAD_AGGRESSIVENESS_LEVEL)
    logger.info("🔊 Initializing loudness meter...")
    loudness_meter = pyln.Meter(SAMPLE_RATE_HZ)
    return model, vad_detector, loudness_meter


# ── DOMAIN CLASSES ──────────────────────────────────────────
class TranscriptionEngine:
    """Handles audio transcription and normalization."""

    def __init__(
        self,
        whisper_model: WhisperModel,
        loudness_meter: pyln.Meter,
        language: str | None = None,
    ) -> None:
        """Initialize the transcription engine."""
        self._model = whisper_model
        self._meter = loudness_meter
        self._language = language

    def _normalize_audio_lufs(self, audio_data: np.ndarray) -> np.ndarray:
        """Normalizes audio to a target LUFS and peak level."""
        if audio_data.size == 0:
            return audio_data
        current_lufs = self._meter.integrated_loudness(audio_data)
        if current_lufs in (-np.inf, np.nan) or np.isnan(
            current_lufs
        ):  # Handle silent or invalid audio
            return audio_data

        normalized_audio = pyln.normalize.loudness(
            audio_data, current_lufs, TARGET_LUFS
        )

        # Peak normalize to prevent clipping if LUFS normalization pushed peaks > 1.0
        peak_amplitude = np.max(np.abs(normalized_audio))
        if peak_amplitude > 1.0:  # Only normalize if peak exceeds 1.0
            normalized_audio = normalized_audio / peak_amplitude
        elif peak_amplitude == 0:  # Avoid division by zero for silent audio
            return np.zeros_like(normalized_audio)

        return normalized_audio

    def transcribe_audio(
        self, audio_data: np.ndarray
    ) -> list[Segment]:  # Replace Any with actual segment type if known
        """Transcribes audio data using the Whisper model."""
        if audio_data.size == 0:
            return []

        normalized_audio = self._normalize_audio_lufs(audio_data)
        if np.max(np.abs(normalized_audio)) < MINIMUM_AUDIO_AMPLITUDE_THRESHOLD:
            return []

        segments, _ = self._model.transcribe(
            audio=normalized_audio,
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=True,
            multilingual=True,
            word_timestamps=True,
            language=self._language,
        )
        return list(segments)


class AudioSegmentBuffer:
    """Buffers audio frames and manages their conversion and timing."""

    def __init__(self, frame_duration_ms: float) -> None:
        """Initialize the audio segment buffer."""
        self._frames: list[bytes] = []
        self._buffer_start_time_sec: float = 0.0
        self._frame_duration_sec: float = frame_duration_ms / 1000.0

    def append_frame(self, pcm_frame_data: bytes) -> None:
        """Appends a PCM audio frame to the buffer."""
        self._frames.append(pcm_frame_data)

    def get_duration_sec(self) -> float:
        """Returns the total duration of buffered audio in seconds."""
        return len(self._frames) * self._frame_duration_sec

    def is_empty(self) -> bool:
        """Checks if the buffer is empty."""
        return not self._frames

    def set_start_time_sec(self, current_processing_clock_sec: float) -> None:
        """Sets the start time of the current buffered segment."""
        self._buffer_start_time_sec = current_processing_clock_sec

    @property
    def start_time_sec(self) -> float:
        """Gets the start time of the current buffered segment."""
        return self._buffer_start_time_sec

    def convert_to_audio_bytes(self) -> bytes:
        """Converts buffered PCM frames to a bytes object."""
        return b"".join(self._frames)

    def convert_to_audio_array(self) -> np.ndarray:
        """Converts buffered PCM frames to a float32 numpy array."""
        if not self._frames:
            return np.array([], dtype=np.float32)
        audio_bytes = b"".join(self._frames)
        return np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    def clear(self) -> None:
        """Clears all frames from the buffer."""
        self._frames.clear()


# ── FORMATTING AND OUTPUT HELPERS ──────────────────────────
def _format_timestamp_for_srt(seconds: float) -> str:
    """Formats seconds into SRT timestamp string (HH:MM:SS,ms)."""
    milliseconds = int((seconds % 1) * 1000)
    return time.strftime("%H:%M:%S", time.gmtime(seconds)) + f",{milliseconds:03d}"


def _display_live_transcription(
    segments: list[Segment],
    buffer_start_offset_sec: float,
    program_start_time_monotonic: float,
    last_displayed_message: str,
) -> str:
    """Prints live transcription segments to the console."""
    current_message = last_displayed_message
    for segment in segments:
        text = segment.text.strip()
        if text and text != last_displayed_message:
            # Calculate timestamp relative to program start for display
            absolute_segment_start_sec = (
                program_start_time_monotonic + buffer_start_offset_sec + segment.start
            )
            display_timestamp = time.strftime(
                "%H:%M:%S", time.localtime(absolute_segment_start_sec)
            )
            typer.echo(f"[{display_timestamp}] {text}")
            current_message = text
    return current_message


def _append_segments_to_srt_file(
    srt_file_path: Path,
    segments: list[Segment],
    buffer_start_offset_sec: float,
    current_srt_index: int,
) -> int:
    """Appends transcribed segments to the SRT file."""
    with srt_file_path.open("a", encoding="utf-8") as srt_file:
        for segment in segments:
            start_time_srt = buffer_start_offset_sec + segment.start
            end_time_srt = buffer_start_offset_sec + segment.end
            text = segment.text.strip()
            if not text:  # Skip empty segments
                continue

            srt_file.write(f"{current_srt_index}\n")
            start_ts = _format_timestamp_for_srt(start_time_srt)
            end_ts = _format_timestamp_for_srt(end_time_srt)
            srt_file.write(f"{start_ts} --> {end_ts}\n")
            srt_file.write(f"{text}\n\n")
            current_srt_index += 1
    return current_srt_index


# ── AUDIO CAPTURE THREAD ───────────────────────────────────
def _convert_chunk_to_mono_float32(audio_chunk: np.ndarray) -> np.ndarray:
    """Converts an audio chunk to mono float32.

    Expected input shape: (num_frames, num_channels) or (num_frames,).
    """
    if audio_chunk.ndim == DIM_ARRAY_2D:
        num_channels = audio_chunk.shape[1]
        if num_channels == MONO_CHANNEL_COUNT:  # Mono, but 2D array
            return audio_chunk.squeeze(axis=1)
        if num_channels == STEREO_CHANNEL_COUNT:  # Stereo
            return audio_chunk.mean(axis=1)  # Mixdown by averaging
        # More than 2 channels, or 0 channels
        msg = f"Audio data has {num_channels} channels. Using only the first channel."
        logger.warning(msg)
        return audio_chunk[:, 0]
    if audio_chunk.ndim == 1:  # Already mono and 1D
        return audio_chunk

    # Unexpected dimensions
    msg = (
        f"Unexpected audio chunk dimensions: {audio_chunk.ndim} "
        f"with shape {audio_chunk.shape}. Skipping."
    )
    logger.error(msg)
    return np.array([], dtype=np.float32)  # Return empty array for safety


def capture_audio_frames(microphone_instance: "MicrophoneProtocol") -> None:
    """Captures audio from the microphone and puts PCM frames into a queue."""
    # COM initialization is crucial for soundcard in a separate thread on Windows
    try:
        if sys.platform == WINDOWS_PLATFORM:
            ctypes.windll.ole32.CoInitializeEx(None, COINIT_APARTMENTTHREADED)
        with microphone_instance.recorder(
            samplerate=SAMPLE_RATE_HZ, blocksize=FRAME_SIZE_SAMPLES
        ) as recorder:
            logger.info(f"🎙️  Audio capture started from: {microphone_instance.name}")
            while not STOP_EVENT.is_set():
                raw_chunk = recorder.record(numframes=FRAME_SIZE_SAMPLES)
                if raw_chunk.shape[0] != FRAME_SIZE_SAMPLES:
                    # logging.debug(
                    #     f"Skipping incomplete audio chunk: "
                    #     f"{raw_chunk.shape[0]} frames"
                    # )
                    continue

                # Ensure float32 for consistent processing before int16 conversion
                # for VAD
                float32_mono_chunk = _convert_chunk_to_mono_float32(raw_chunk)
                if float32_mono_chunk.size == 0:
                    continue

                # Convert float32 mono audio to int16 bytes for VAD and queueing
                # Ensure values are in [-1.0, 1.0] range before scaling
                # np.clip(float32_mono_chunk, -1.0, 1.0, out=float32_mono_chunk)
                # pcm_data_bytes = (float32_mono_chunk * 32767.0).astype(
                #     np.int16
                # ).tobytes()  # Max value is 32767 for int16

                # The original code did not clip, it scaled directly.
                # Let's assume input audio is already in a reasonable range
                # If input `raw_chunk` is float, it's typically [-1,1]
                # `soundcard` usually returns float32 in [-1,1] range.
                pcm_data_bytes = (
                    (float32_mono_chunk * 32768.0).astype(np.int16).tobytes()
                )
                AUDIO_FRAME_QUEUE.put(pcm_data_bytes)
    except Exception:
        logger.exception("Error in audio capture thread")
    finally:
        logger.info("🎙️  Audio capture stopped.")
        if sys.platform == WINDOWS_PLATFORM:
            ctypes.windll.ole32.CoUninitialize()


# ── TRANSCRIPTION WORKFLOW HELPERS ──────────────────────────
def _process_live_segment(
    pcm_frame: bytes,
    live_audio_buffer: AudioSegmentBuffer,
    processing_clock_sec: float,
    vad_silence_frame_count: int,
    transcription_engine: TranscriptionEngine,
    program_start_time_monotonic: float,
    last_console_message: str,
) -> tuple[str, bool]:
    """Processes an audio frame for live transcription output."""
    buffer_cleared = False
    if live_audio_buffer.is_empty():
        live_audio_buffer.set_start_time_sec(processing_clock_sec)
    live_audio_buffer.append_frame(pcm_frame)

    live_buffer_duration_sec = live_audio_buffer.get_duration_sec()
    if (
        live_buffer_duration_sec >= LIVE_MIN_TRANSCRIPTION_DURATION_SEC
        and vad_silence_frame_count >= VAD_LIVE_SILENCE_FRAMES
    ):
        audio_to_transcribe = live_audio_buffer.convert_to_audio_array()

        segments = transcription_engine.transcribe_audio(audio_to_transcribe)
        last_console_message = _display_live_transcription(
            segments,
            live_audio_buffer.start_time_sec,
            program_start_time_monotonic,
            last_console_message,
        )

        live_audio_buffer.clear()
        buffer_cleared = True
    return last_console_message, buffer_cleared


def _process_file_segment(
    pcm_frame: bytes,
    file_audio_buffer: AudioSegmentBuffer,
    processing_clock_sec: float,
    vad_silence_frame_count: int,
    transcription_engine: TranscriptionEngine,
    current_srt_index: int,
    srt_file_path: Path,
) -> tuple[int, bool]:
    """Processes an audio frame for SRT file output."""
    buffer_cleared = False
    if file_audio_buffer.is_empty():
        file_audio_buffer.set_start_time_sec(processing_clock_sec)
    file_audio_buffer.append_frame(pcm_frame)

    file_buffer_duration_sec = file_audio_buffer.get_duration_sec()

    if (
        file_buffer_duration_sec >= FILE_MIN_TRANSCRIPTION_DURATION_SEC
        and vad_silence_frame_count >= VAD_FILE_SILENCE_FRAMES
    ):
        audio_to_transcribe = file_audio_buffer.convert_to_audio_array()

        segments = transcription_engine.transcribe_audio(audio_to_transcribe)
        current_srt_index = _append_segments_to_srt_file(
            srt_file_path,
            segments,
            file_audio_buffer.start_time_sec,
            current_srt_index,
        )

        file_audio_buffer.clear()
        buffer_cleared = True
    return current_srt_index, buffer_cleared


def _flush_remaining_audio_to_srt(
    file_audio_buffer: AudioSegmentBuffer,
    transcription_engine: TranscriptionEngine,
    current_srt_index: int,
    srt_file_path: Path,
) -> int:
    """Flushes any remaining audio in the file buffer to the SRT file."""
    if not file_audio_buffer.is_empty():
        logger.info("⏳ Processing remaining audio for SRT file...")
        audio_to_transcribe = file_audio_buffer.convert_to_audio_array()
        segments = transcription_engine.transcribe_audio(audio_to_transcribe)
        if segments:
            current_srt_index = _append_segments_to_srt_file(
                srt_file_path,
                segments,
                file_audio_buffer.start_time_sec,
                current_srt_index,
            )
        file_audio_buffer.clear()
    return current_srt_index


# ── TRANSCRIPTION WORKFLOW THREAD ───────────────────────────
def manage_transcription_workflow(
    transcription_engine: TranscriptionEngine,
    vad_detector: webrtcvad.Vad,
    program_start_time_monotonic: float,
    srt_file_path: Path,
) -> None:
    """Manages audio buffering, VAD, transcription, and output."""
    srt_entry_index = 1
    processing_clock_sec = 0.0  # Tracks total audio processed time

    live_audio_buffer = AudioSegmentBuffer(FRAME_DURATION_MS)
    file_audio_buffer = AudioSegmentBuffer(FRAME_DURATION_MS)

    vad_silence_frame_count = 0
    last_console_message = ""

    try:
        while not (STOP_EVENT.is_set() and AUDIO_FRAME_QUEUE.empty()):
            try:
                pcm_frame = AUDIO_FRAME_QUEUE.get(
                    timeout=0.1
                )  # Short timeout to allow stop_event check
            except queue.Empty:
                if STOP_EVENT.is_set():  # If stopping and queue is empty, break
                    break
                continue  # Otherwise, continue waiting for frames

            # --- VAD ---
            is_speech = vad_detector.is_speech(pcm_frame, SAMPLE_RATE_HZ)
            if is_speech:
                vad_silence_frame_count = 0
            else:
                vad_silence_frame_count += 1

            # --- Process for Live Console Output ---
            last_console_message, live_buffer_cleared = _process_live_segment(
                pcm_frame,
                live_audio_buffer,
                processing_clock_sec,
                vad_silence_frame_count,
                transcription_engine,
                program_start_time_monotonic,
                last_console_message,
            )
            # If live buffer was cleared by VAD, reset silence count to prevent
            # immediate re-trigger on next frame if it's also silence
            if live_buffer_cleared:
                vad_silence_frame_count = 0

            # --- Process for SRT File Output ---
            srt_entry_index, file_buffer_cleared = _process_file_segment(
                pcm_frame,
                file_audio_buffer,
                processing_clock_sec,
                vad_silence_frame_count,  # VAD_FILE_SILENCE_FRAMES used by helper
                transcription_engine,
                srt_entry_index,
                srt_file_path,
            )
            # If file buffer was cleared by VAD, reset silence count similarly
            if file_buffer_cleared:
                vad_silence_frame_count = 0

            processing_clock_sec += FRAME_DURATION_MS / 1000.0

        # --- Final flush for any remaining audio in the file buffer ---
        _flush_remaining_audio_to_srt(
            file_audio_buffer, transcription_engine, srt_entry_index, srt_file_path
        )
        logger.info("✍️ Transcription workflow finished.")

    except Exception:
        logger.exception("Error in transcription workflow")


# ── MAIN APPLICATION HELPERS ────────────────────────────────
def _initialize_srt_file(srt_path: Path) -> None:
    """Clears the SRT file if it exists, and logs its path."""
    srt_path.write_text("", encoding="utf-8")
    logger.info(f"📝 SRT file will be saved to: {srt_path.resolve()}")


def _select_audio_device() -> "MicrophoneProtocol":
    """Selects the audio device, preferring loopback, then default."""
    try:
        default_speaker = sc.default_speaker()

        # Replace "Speaker" with "Loopback" in the default speaker name
        loopback_name = default_speaker.name.replace("Speaker", "Loopback")

        # Get all microphones including loopback
        all_mics = sc.all_microphones(include_loopback=True)
        loopback_mics = [m for m in all_mics if m.isloopback]

        # Search for a loopback microphone with the modified name
        matching_loopback = next(
            (mic for mic in loopback_mics if mic.name == loopback_name), None
        )

        if matching_loopback:
            microphone = matching_loopback
            logger.info(f"🎤 Found matching loopback microphone: {microphone.name}")
        else:
            msg = (
                f"Loopback for default speaker '{default_speaker.name}' "
                f"not found. Aborting."
            )
            logger.error(msg)
            raise RuntimeError(msg)
    except Exception as e:
        logger.warning(
            f"Error selecting preferred (loopback/default) mic: {e}. "
            f"Falling back to default input."
        )
        microphone = sc.default_microphone()

    logger.info(
        f"🎤 Selected audio device: {microphone.name} "
        f"(Loopback: {microphone.isloopback})"
    )
    return microphone


# ── MAIN APPLICATION LOGIC ─────────────────────────────────
def run_transcription_pipeline(
    language: str | None, model_name: str, srt_file_path: Path
) -> None:
    """Initializes and runs the transcription pipeline."""
    program_start_time_monotonic = time.monotonic()

    _initialize_srt_file(srt_file_path)

    try:
        model, vad_detector, loudness_meter = initialize_transcription_resources(
            model_name
        )
        transcription_engine = TranscriptionEngine(model, loudness_meter, language)
        microphone = _select_audio_device()

        # Start threads
        capture_thread_instance = threading.Thread(
            target=capture_audio_frames,
            args=(microphone,),
            daemon=True,
            name="AudioCaptureThread",
        )
        transcription_thread_instance = threading.Thread(
            target=manage_transcription_workflow,
            args=(
                transcription_engine,
                vad_detector,
                program_start_time_monotonic,
                srt_file_path,
            ),
            daemon=True,
            name="TranscriptionWorkflowThread",
        )

        capture_thread_instance.start()
        transcription_thread_instance.start()

        # Graceful shutdown handling
        def signal_handler(sig: int, _: object) -> None:
            logger.info(f"\n🛑 Signal {sig} received. Shutting down gracefully...")
            STOP_EVENT.set()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Keep main thread alive while worker threads are running
        while (
            capture_thread_instance.is_alive()
            and transcription_thread_instance.is_alive()
        ):
            try:
                time.sleep(0.5)  # Check periodically
            except InterruptedError:  # Handles Ctrl+C during sleep on some systems
                signal_handler(signal.SIGINT, None)
                break

        # Ensure threads have a chance to finish after stop_event is set
        logger.info("⏳ Waiting for threads to complete...")
        capture_thread_instance.join(timeout=5.0)
        transcription_thread_instance.join(
            timeout=10.0
        )  # Transcription might take longer to flush

        if capture_thread_instance.is_alive():
            logger.warning("Audio capture thread did not terminate cleanly.")
        if transcription_thread_instance.is_alive():
            logger.warning("Transcription workflow thread did not terminate cleanly.")

        logger.info(
            f"\n✅ Transcription complete. Subtitles saved to: "
            f"{srt_file_path.resolve()}"
        )

    except Exception:
        logger.exception("An unhandled error occurred in the main application")
    finally:
        logger.info("👋 Exiting application.")


# ── CLI APPLICATION ──────────────────────────────────────
app = typer.Typer(help="Realtime ASR with faster-whisper — live console and SRT output")


@app.command()
def main(
    language: str | None = typer.Option(
        None,
        "--language",
        "-l",
        help=(
            "Language code (ISO 639-1) for transcription. Common examples: "
            "en, ja, zh, es, fr, de, ko, ru, pt. If not specified, language "
            "will be auto-detected."
        ),
        metavar="LANG",
    ),
    srt_file_path: Path | None = typer.Option(
        DEFAULT_SRT_FILE_PATH,
        "--output",
        "-o",
        help="Output SRT file path",
        metavar="FILE",
    ),
    model: str = typer.Option(
        "large-v3-turbo",
        "--model",
        "-m",
        help="Whisper model name",
        metavar="MODEL",
    ),
) -> None:
    """
    Realtime ASR with faster-whisper — live console and SRT output

    This script performs real-time automatic speech recognition using the
    faster-whisper library. It captures audio from your system's loopback device
    and provides live transcription to both console and SRT subtitle file format.

    The language parameter accepts ISO 639-1 language codes. The actual supported
    languages depend on the Whisper model being used. If an unsupported language
    is specified, the transcription will fail with an error message.

    Examples:
        # Auto-detect language
        python better_whisper_transcribe.py

        # Specify Japanese language
        python better_whisper_transcribe.py --language ja

        # Specify language and output file
        python better_whisper_transcribe.py --language en --output my_transcript.srt

        # Use different model
        python better_whisper_transcribe.py --model large-v3 --language zh
    """
    # Language information
    if language is not None:
        typer.echo(f"Using language: {language}")
    else:
        typer.echo("Language will be auto-detected from audio")

    try:
        run_transcription_pipeline(language, model, srt_file_path)
    except KeyboardInterrupt:
        typer.echo("\n\n🛑 Interrupted by user. Shutting down...")
        raise typer.Exit(0) from None
    except Exception as e:
        error_msg = str(e).lower()
        if "language" in error_msg and language is not None:
            typer.echo(f"\n❌ Language error: {e}", err=True)
            typer.echo(
                f"The language code '{language}' may not be supported "
                f"by the current Whisper model.",
                err=True,
            )
            typer.echo(
                "Try using standard ISO 639-1 codes (en, ja, zh, es, fr, de, etc.) "
                "or omit --language for auto-detection.",
                err=True,
            )
        else:
            typer.echo(f"\n❌ Fatal error: {e}", err=True)
        raise typer.Exit(1) from e


if __name__ == "__main__":
    app()
