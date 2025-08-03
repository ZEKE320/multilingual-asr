"""
Realtime ASR with faster-whisper (v3) — live console and SRT output
----------------------------------------------------------------------
* Live console driven by VAD + timing (natural or forced)
* SRT file output on VAD-detected silence or forced boundaries
* Dual audio input: Speaker loopback + Microphone with proper normalization
"""

from __future__ import annotations

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
from queue import Empty

import numpy as np
import pyloudnorm as pyln
import soundcard as sc
import torch
import typer
import webrtcvad
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment
from rich.logging import RichHandler
from soundcard.mediafoundation import _Microphone, _Recorder

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

# Windows-specific: Avoid Symlink permission errors for Hugging Face Hub
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

# ── CONFIGURATION CONSTANTS ─────────────────────────────────
# Audio processing
SAMPLE_RATE_HZ = 16000
FRAME_DURATION_MS = 10
FRAME_SIZE_SAMPLES = int(SAMPLE_RATE_HZ * (FRAME_DURATION_MS / 1000.0))
TARGET_LUFS = -14.0

# Audio conversion and normalization
# Use NumPy to obtain the correct maximum value for int16 instead of hard-coding a literal.
INT16_MAX = float(np.iinfo(np.int16).max)  # 32767.0
PEAK_AMPLITUDE_LIMIT = 1.0  # Maximum peak amplitude before clipping
# VAD processing expects at least this peak level (fraction of full scale)
PRE_VAD_TARGET_PEAK = 0.3  # If signal peak < this, boost to this level
SILENCE_THRESHOLD = 0.1  # Audio below this level considered silence
MIN_LUFS_AUDIO_DURATION_SEC = 0.5  # Minimum audio length for LUFS normalization

# Audio mixing
SPEAKER_MIX_RATIO = 0.5  # 50% speaker, 50% microphone when both available
MICROPHONE_MIX_RATIO = 0.5

# Error handling and timing
MAX_CONSECUTIVE_ERRORS = 10  # Max errors before assuming device failure
AUDIO_QUEUE_TIMEOUT_SEC = 0.1  # Timeout for audio queue operations
MIXER_SLEEP_TIME_SEC = 0.001  # Sleep time when no audio frames available
THREAD_JOIN_TIMEOUT_SEC = 5.0  # Timeout for capture/mixer threads
TRANSCRIPTION_THREAD_TIMEOUT_SEC = 10.0  # Timeout for transcription thread
MAIN_LOOP_SLEEP_SEC = 0.5  # Main thread sleep interval

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
SRT_FILE_PATH = Path("transcript.srt")

# Threading and Queues
audio_mixer_queue = queue.Queue()  # Queue for mixed audio frames
speaker_audio_queue = queue.Queue()  # Queue for speaker (loopback) audio
microphone_audio_queue = queue.Queue()  # Queue for microphone audio
stop_event = threading.Event()

# Windows COM a_initialization_for audio capture thread
COINIT_APARTMENTTHREADED = 0x2


logging.basicConfig(
    handlers=[RichHandler(rich_tracebacks=True)],
)

logger = logging.getLogger(__name__)


class DebugManager:
    """Centralizes debug mode state for auxiliary helpers.

    The current implementation is intentionally lightweight: it only exposes
    the *is_debug_mode* flag used by :class:`AudioStatsLogger`. If in the
    future we want to add FPS counters or Rich-based UIs, those extensions
    can live here without touching the rest of the pipeline.
    """

    def __init__(self, enabled: bool = False) -> None:  # noqa: D401 (imperative mood)
        self.is_debug_mode: bool = enabled

    def enable(self) -> None:
        """Turn on debug mode at runtime."""

        self.is_debug_mode = True

    def disable(self) -> None:
        """Turn off debug mode at runtime."""

        self.is_debug_mode = False


# Will be initialised inside *run_transcription_pipeline* once we know the
# value of the --debug CLI flag.
DEBUG_MANAGER: DebugManager | None = None

# Stats loggers (set at runtime). All are Optional so they are no-ops when debug is False.
SPEAKER_STATS_LOGGER: AudioStatsLogger | None = None
MIC_STATS_LOGGER: AudioStatsLogger | None = None
MIX_STATS_LOGGER: AudioStatsLogger | None = None
VAD_STATS_LOGGER: AudioStatsLogger | None = None


class AudioStatsLogger:
    """A helper class to aggregate audio statistics and report them to a DebugManager."""

    def __init__(
        self, name: str, debug_manager: "DebugManager", log_interval_sec: float = 1.0
    ):
        self.name = name
        self.debug_manager = debug_manager
        self.log_interval_sec = log_interval_sec
        self.is_debug_mode = self.debug_manager.is_debug_mode

        if not self.is_debug_mode:
            return

        self.last_log_time = time.monotonic()
        self._reset_stats()

    def _reset_stats(self):
        """Resets the statistics for the next interval."""
        self.last_log_time = time.monotonic()
        self.frames_in_interval = 0
        self.max_amp_in_interval = -1.0
        self.min_amp_in_interval = 1.0
        self.sum_avg_amp_in_interval = 0.0
        self.speech_frames_in_interval = 0

    def _report(self, stats: dict):
        """Output aggregated stats via logging.debug each interval."""
        logging.debug(
            f"[{self.name}] " + ", ".join(f"{k}={v}" for k, v in stats.items())
        )
        self._reset_stats()

    def update_raw_stats(self, audio_chunk: np.ndarray):
        """Update stats for raw float32 audio chunks."""
        if not self.is_debug_mode:
            return

        self.max_amp_in_interval = max(self.max_amp_in_interval, np.max(audio_chunk))
        self.min_amp_in_interval = min(self.min_amp_in_interval, np.min(audio_chunk))
        self.sum_avg_amp_in_interval += np.mean(np.abs(audio_chunk))
        self.frames_in_interval += 1

        if time.monotonic() - self.last_log_time >= self.log_interval_sec:
            avg_amp = (
                self.sum_avg_amp_in_interval / self.frames_in_interval
                if self.frames_in_interval > 0
                else 0
            )
            self._report(
                {
                    "Max": self.max_amp_in_interval,
                    "Min": self.min_amp_in_interval,
                    "Avg|A|": avg_amp,
                    "Frames": self.frames_in_interval,
                }
            )

    def update_vad_stats(self, pcm_frame: bytes, is_speech: bool):
        """Update stats for VAD analysis (int16 PCM bytes)."""
        if not self.is_debug_mode:
            return

        audio_array = np.frombuffer(pcm_frame, dtype=np.int16)
        self.sum_avg_amp_in_interval += np.mean(np.abs(audio_array))
        self.frames_in_interval += 1
        if is_speech:
            self.speech_frames_in_interval += 1

        if time.monotonic() - self.last_log_time >= self.log_interval_sec:
            avg_amp = (
                self.sum_avg_amp_in_interval / self.frames_in_interval
                if self.frames_in_interval > 0
                else 0
            )
            speech_ratio = (
                self.speech_frames_in_interval / self.frames_in_interval
                if self.frames_in_interval > 0
                else 0
            )
            self._report(
                {
                    "Avg|A|": avg_amp,
                    "Ratio": speech_ratio,
                    "Frames": self.frames_in_interval,
                }
            )


# ── LOGGING SETUP ───────────────────────────────────────────
def setup_logging(debug: bool = False) -> None:
    """Configures logging with RichHandler for better console output.

    Args:
        debug: If True, enables debug-level logging
    """
    log_level = logging.DEBUG if debug else logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )

    # Silence overly verbose loggers unless in debug mode
    verbose_loggers = [
        "faster_whisper",
        "soundcard",
        "huggingface_hub",
        "urllib3",
        "filelock",
    ]
    for logger_name in verbose_loggers:
        logging.getLogger(logger_name).setLevel(
            logging.WARNING if not debug else logging.INFO
        )


# ── RESOURCE INITIALIZATION ────────────────────────────────
def initialize_transcription_resources() -> tuple[
    WhisperModel, webrtcvad.Vad, pyln.Meter
]:
    """Loads and initializes Whisper model, VAD, and loudness meter."""

    if not torch.cuda.is_available():
        logging.warning(
            "⚠️  CUDA is not available. Using CPU instead. Performance will be slower."
        )
        device_type = "cpu"
        compute_type = "int8"
    else:
        device_type = "cuda"
        compute_type = "int8_float16"

    model_name = "large-v3-turbo"

    logging.info(
        f"📦 Loading Whisper {model_name} model (device: {device_type}, compute: {compute_type})..."
    )

    try:
        model = WhisperModel(
            model_name,
            device=device_type,
            compute_type=compute_type,
        )
    except:
        logging.exception(f"Failed to load Whisper model")
        raise

    logging.info("🗣️ Initializing VAD...")
    vad_detector = webrtcvad.Vad(VAD_AGGRESSIVENESS_LEVEL)
    logging.info("🔊 Initializing loudness meter...")
    loudness_meter = pyln.Meter(SAMPLE_RATE_HZ)
    return model, vad_detector, loudness_meter


# ── DOMAIN CLASSES ──────────────────────────────────────────
class TranscriptionEngine:
    """Handles audio transcription and normalization."""

    def __init__(self, whisper_model: WhisperModel, loudness_meter: pyln.Meter):
        self._model = whisper_model
        self._meter = loudness_meter

    def _normalize_audio_lufs(self, audio_data: np.ndarray) -> np.ndarray:
        """Normalizes audio to a target LUFS and peak level.

        Only applies LUFS normalization if audio is long enough (>0.5 seconds).
        For shorter audio, only peak normalization is applied.
        """
        if audio_data.size == 0:
            return audio_data

        # Check if audio is long enough for LUFS measurement
        min_samples_for_lufs = int(MIN_LUFS_AUDIO_DURATION_SEC * SAMPLE_RATE_HZ)

        if audio_data.size >= min_samples_for_lufs:
            try:
                current_lufs = self._meter.integrated_loudness(audio_data)
                if current_lufs != -np.inf and not np.isnan(current_lufs):
                    normalized_audio = pyln.normalize.loudness(
                        audio_data, current_lufs, TARGET_LUFS
                    )
                else:
                    # Audio is too quiet for LUFS measurement
                    normalized_audio = audio_data
            except Exception as e:
                logging.warning(
                    f"LUFS normalization failed: {e}. Using peak normalization only."
                )
                normalized_audio = audio_data
        else:
            # Audio too short for LUFS, use original
            logging.warning(f"Audio too short for LUFS, using original.")
            normalized_audio = audio_data

        # Always apply peak normalization to prevent clipping and ensure sufficient level.
        peak_amplitude = np.max(np.abs(normalized_audio))

        if peak_amplitude == 0:  # Completely silent buffer
            return np.zeros_like(normalized_audio)

        if peak_amplitude > PEAK_AMPLITUDE_LIMIT:  # Too loud → scale down
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
        # Check if audio is too quiet (below noise floor) **after** normalization.
        peak_amplitude = np.max(np.abs(normalized_audio))
        if peak_amplitude < SILENCE_THRESHOLD:
            return []

        try:
            segments, _info = self._model.transcribe(
                audio=normalized_audio,
                beam_size=5,
                vad_filter=True,
                condition_on_previous_text=True,
                multilingual=True,
                word_timestamps=True,
            )
            return list(segments)
        except Exception as e:
            logging.error(f"Transcription failed: {e}")
            return []


class AudioSegmentBuffer:
    """Buffers audio frames and manages their conversion and timing."""

    def __init__(self, frame_duration_ms: float):
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
            print(f"[{display_timestamp}] {text}")
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
            srt_file.write(
                f"{_format_timestamp_for_srt(start_time_srt)} --> {_format_timestamp_for_srt(end_time_srt)}\n"
            )
            srt_file.write(f"{text}\n\n")
            current_srt_index += 1
    return current_srt_index


# ── AUDIO CAPTURE THREAD ───────────────────────────────────
def _convert_chunk_to_mono_float32(audio_chunk: np.ndarray) -> np.ndarray:
    """Converts an audio chunk to mono float32. Expected input shape: (num_frames, num_channels) or (num_frames,)."""
    if audio_chunk.ndim == 2:
        num_channels = audio_chunk.shape[1]
        if num_channels == 1:  # Mono, but 2D array e.g. (FRAME_SIZE, 1)
            return audio_chunk.squeeze(axis=1)
        if num_channels == 2:  # Stereo
            return audio_chunk.mean(axis=1)  # Mixdown by averaging
        # More than 2 channels, or 0 channels
        logging.warning(
            f"Audio data has {num_channels} channels. Using only the first channel."
        )
        return audio_chunk[:, 0]
    if audio_chunk.ndim == 1:  # Already mono and 1D
        return audio_chunk

    # Unexpected dimensions
    logging.error(
        f"Unexpected audio chunk dimensions: {audio_chunk.ndim} with shape {audio_chunk.shape}. Skipping."
    )
    return np.array([], dtype=np.float32)  # Return empty array for safety


def capture_audio_frames(
    microphone_instance: _Microphone,
    target_queue: queue.Queue,
    device_name: str,
) -> None:
    """Captures audio from the specified microphone/loopback and puts PCM frames into the target queue."""
    try:
        # COM initialization for Windows
        ctypes.windll.ole32.CoInitializeEx(None, COINIT_APARTMENTTHREADED)

        with microphone_instance.recorder(
            samplerate=SAMPLE_RATE_HZ, blocksize=FRAME_SIZE_SAMPLES
        ) as recorder:
            logging.info(
                f"🎙️  Audio capture started from {device_name}: {microphone_instance.name}"
            )
            _record_audio_loop(recorder, target_queue, device_name)

    except Exception:
        logging.exception(f"[{device_name}] Fatal error in audio capture thread.")
        target_queue.put(None)  # Signal failure
    finally:
        logging.info(f"🎙️  Audio capture stopped for {device_name}.")
        ctypes.windll.ole32.CoUninitialize()


def _record_audio_loop(
    recorder: _Recorder,
    target_queue: queue.Queue,
    device_name: str,
) -> None:
    """Main recording loop for audio capture."""
    consecutive_errors = 0

    while not stop_event.is_set():
        try:
            raw_chunk = recorder.record(numframes=FRAME_SIZE_SAMPLES)
            consecutive_errors = 0  # Reset on success

            if raw_chunk.shape[0] != FRAME_SIZE_SAMPLES:
                continue  # Skip incomplete frames

            # Convert to mono float32
            float32_mono_chunk = _convert_chunk_to_mono_float32(raw_chunk)
            if float32_mono_chunk.size == 0:
                continue

            # Gated pre-VAD auto-gain:
            #   • Very low-level noise (<0.02) → no gain (keeps silence silent)
            #   • Moderate speech (<PRE_VAD_TARGET_PEAK) → boost up to target
            peak = np.max(np.abs(float32_mono_chunk))
            if SILENCE_THRESHOLD < peak < PRE_VAD_TARGET_PEAK:
                float32_mono_chunk *= PRE_VAD_TARGET_PEAK / peak

            # Update appropriate stats logger based on target queue identity (safer than string compare)
            if target_queue is speaker_audio_queue and SPEAKER_STATS_LOGGER is not None:
                SPEAKER_STATS_LOGGER.update_raw_stats(float32_mono_chunk)
            elif (
                target_queue is microphone_audio_queue and MIC_STATS_LOGGER is not None
            ):
                MIC_STATS_LOGGER.update_raw_stats(float32_mono_chunk)

            # Convert to int16 bytes for VAD
            pcm_data_bytes = (float32_mono_chunk * INT16_MAX).astype(np.int16).tobytes()
            target_queue.put(pcm_data_bytes)

        except Exception as e:
            consecutive_errors += 1

            # Log appropriate error message
            if "data discontinuity" in str(e).lower() and consecutive_errors == 1:
                logging.debug(e)
            else:
                logging.warning(f"[{device_name}] Error recording audio frame: {e}")

            # Check if too many errors
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                logging.error(
                    f"[{device_name}] Too many consecutive errors. Assuming device failure."
                )
                break


# ── AUDIO MIXING THREAD ─────────────────────────────────
def _try_get_audio_frame(
    queue: queue.Queue, source_failed: bool
) -> tuple[bytes | None, bool]:
    """Attempts to get an audio frame from a queue.

    Returns:
        tuple: (frame, source_failed) - frame is None if not available or source failed
    """
    if source_failed or queue.empty():
        return None, source_failed

    try:
        frame = queue.get_nowait()
        if frame is None:  # Sentinel value indicates failure
            return None, True
        return frame, False
    except Empty:
        return None, source_failed


def _apply_level_matching(
    speaker: np.ndarray, mic: np.ndarray, max_gain: float = 4.0
) -> tuple[np.ndarray, np.ndarray]:
    """Brings two mono signals to a comparable RMS level.

    The quieter source is linearly boosted up to *max_gain* (≈ +12 dB) so that
    neither stream dominates the other when they are mixed.

    Args:
        speaker: Float-32 mono array from loopback/speaker.
        mic: Float-32 mono array from microphone.
        max_gain: Upper bound for the applied gain factor.

    Returns:
        Tuple of (speaker_adjusted, mic_adjusted).
    """

    def _rms(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0

    rms_s = _rms(speaker)
    rms_m = _rms(mic)
    eps = 1e-6

    if rms_s > rms_m + eps:
        gain_m = min(rms_s / (rms_m + eps), max_gain)
        gain_s = 1.0
    elif rms_m > rms_s + eps:
        gain_s = min(rms_m / (rms_s + eps), max_gain)
        gain_m = 1.0
    else:
        gain_s = gain_m = 1.0

    return speaker * gain_s, mic * gain_m


def _mix_frames(
    speaker_frame: bytes | None,
    microphone_frame: bytes | None,
) -> bytes | None:
    """Mixes speaker and microphone frames with proper normalization.

    Returns:
        Mixed audio frame or None if no frames available
    """
    if not speaker_frame and not microphone_frame:
        return None

    # Both frames available - mix them
    if speaker_frame and microphone_frame:
        # Convert to float32 arrays
        speaker_array = (
            np.frombuffer(speaker_frame, dtype=np.int16).astype(np.float32) / INT16_MAX
        )
        microphone_array = (
            np.frombuffer(microphone_frame, dtype=np.int16).astype(np.float32)
            / INT16_MAX
        )

        # Level-match sources before mixing so that neither dominates.
        speaker_array, microphone_array = _apply_level_matching(
            speaker_array, microphone_array
        )

        # Equal-power mix (50 % / 50 %)
        mixed_array = (speaker_array + microphone_array) * 0.5

        # Debug statistics
        if MIX_STATS_LOGGER is not None:
            MIX_STATS_LOGGER.update_raw_stats(mixed_array)

        # Convert back to int16
        return (mixed_array * INT16_MAX).astype(np.int16).tobytes()

    # Only one frame available - return it
    return speaker_frame or microphone_frame


def mix_audio_streams() -> None:
    """Mixes speaker and microphone audio streams with simple peak normalization."""
    speaker_failed = False
    microphone_failed = False
    # No logger here, VAD logger is more informative

    logging.info("🎵 Audio mixer started.")

    while not stop_event.is_set():
        # Get frames from both sources
        speaker_frame, speaker_failed = _try_get_audio_frame(
            speaker_audio_queue, speaker_failed
        )
        if speaker_failed and speaker_frame is None:
            logging.error("Speaker audio source has failed.")

        microphone_frame, microphone_failed = _try_get_audio_frame(
            microphone_audio_queue, microphone_failed
        )
        if microphone_failed and microphone_frame is None:
            logging.error("Microphone audio source has failed.")

        # Check if both sources have failed
        if speaker_failed and microphone_failed:
            logging.error("Both audio sources have failed. Stopping mixer.")
            stop_event.set()
            break

        # Mix available frames
        mixed_frame = _mix_frames(speaker_frame, microphone_frame)
        if mixed_frame:
            audio_mixer_queue.put(mixed_frame)
        else:
            # No frames available, sleep briefly to avoid busy waiting
            time.sleep(MIXER_SLEEP_TIME_SEC)

    logging.info("🎵 Audio mixer stopped.")


# ── TRANSCRIPTION WORKFLOW HELPERS ──────────────────────────
def _process_live_segment(
    pcm_frame: bytes,
    live_audio_buffer: AudioSegmentBuffer,
    processing_clock_sec: float,
    vad_silence_frame_count: int,
    transcription_engine: TranscriptionEngine,
    program_start_time_monotonic: float,
    last_console_message: str,
) -> tuple[
    str, bool
]:  # Returns updated last_console_message and whether buffer was cleared
    """Processes an audio frame for live transcription output."""
    buffer_cleared = False

    # Append frame to buffer
    if live_audio_buffer.is_empty():
        live_audio_buffer.set_start_time_sec(processing_clock_sec)
    live_audio_buffer.append_frame(pcm_frame)

    live_buffer_duration_sec = live_audio_buffer.get_duration_sec()

    if (
        live_buffer_duration_sec > 0
        and live_buffer_duration_sec >= LIVE_MIN_TRANSCRIPTION_DURATION_SEC
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
) -> tuple[int, bool]:  # Returns updated srt_index and whether buffer was cleared
    """Processes an audio frame for SRT file output."""
    buffer_cleared = False

    # Append frame to buffer
    if file_audio_buffer.is_empty():
        file_audio_buffer.set_start_time_sec(processing_clock_sec)
    file_audio_buffer.append_frame(pcm_frame)

    file_buffer_duration_sec = file_audio_buffer.get_duration_sec()

    if (
        file_buffer_duration_sec > 0
        and file_buffer_duration_sec >= FILE_MIN_TRANSCRIPTION_DURATION_SEC
        and vad_silence_frame_count >= VAD_FILE_SILENCE_FRAMES
    ):
        audio_to_transcribe = file_audio_buffer.convert_to_audio_array()

        segments = transcription_engine.transcribe_audio(audio_to_transcribe)
        current_srt_index = _append_segments_to_srt_file(
            SRT_FILE_PATH,
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
) -> int:
    """Flushes any remaining audio in the file buffer to the SRT file."""
    if not file_audio_buffer.is_empty():
        logging.info("⏳ Processing remaining audio for SRT file...")
        audio_to_transcribe = file_audio_buffer.convert_to_audio_array()
        segments = transcription_engine.transcribe_audio(audio_to_transcribe)
        if segments:
            current_srt_index = _append_segments_to_srt_file(
                SRT_FILE_PATH,
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
) -> None:
    """Manages audio buffering, VAD, transcription, and output."""
    srt_entry_index = 1
    processing_clock_sec = 0.0  # Tracks total audio processed time

    live_audio_buffer = AudioSegmentBuffer(FRAME_DURATION_MS)
    file_audio_buffer = AudioSegmentBuffer(FRAME_DURATION_MS)

    vad_silence_frame_count = 0
    last_console_message = ""

    try:
        while not (stop_event.is_set() and audio_mixer_queue.empty()):
            try:
                pcm_frame = audio_mixer_queue.get(
                    timeout=AUDIO_QUEUE_TIMEOUT_SEC
                )  # Short timeout to allow stop_event check
            except Empty:
                if stop_event.is_set():  # If stopping and queue is empty, break
                    break
                continue  # Otherwise, continue waiting for frames

            # --- VAD ---
            is_speech = vad_detector.is_speech(pcm_frame, SAMPLE_RATE_HZ)

            # Debug statistics (per-frame VAD result)
            if VAD_STATS_LOGGER is not None:
                VAD_STATS_LOGGER.update_vad_stats(pcm_frame, is_speech)

            if is_speech:
                # Reset silence counter while speech is detected
                vad_silence_frame_count = 0
            else:
                # Increment silence counter when frame is classified as non-speech
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

            # --- Process for SRT File Output ---
            srt_entry_index, file_buffer_cleared = _process_file_segment(
                pcm_frame,
                file_audio_buffer,
                processing_clock_sec,
                vad_silence_frame_count,
                transcription_engine,
                srt_entry_index,
            )

            # If a buffer was cleared, it means a transcription happened.
            # We reset the counter here to prevent the other buffer from
            # transcribing the same segment again.
            if live_buffer_cleared or file_buffer_cleared:
                vad_silence_frame_count = 0

            processing_clock_sec += FRAME_DURATION_MS / 1000.0

        # --- Final flush for any remaining audio in the file buffer ---
        # Note: We don't need to flush the live buffer as it's for display only.
        _flush_remaining_audio_to_srt(
            file_audio_buffer, transcription_engine, srt_entry_index
        )
        logging.info("✍️ Transcription workflow finished.")

    except Exception as e:
        logging.error(f"Error in transcription workflow: {e}", exc_info=True)


# ── MAIN APPLICATION HELPERS ────────────────────────────────
def _initialize_srt_file(srt_path: Path) -> None:
    """Clears the SRT file if it exists, and logs its path."""
    srt_path.write_text("", encoding="utf-8")
    logging.info(f"📝 SRT file will be saved to: {srt_path.resolve()}")


def _find_speaker_loopback() -> _Microphone | None:
    """Finds and returns the loopback device for the default speaker."""
    try:
        default_speaker = sc.default_speaker()
        if not default_speaker:
            logging.error(
                "❌ No default speaker found. Speaker audio will not be captured."
            )
            return None

        # Replace "Speaker" with "Loopback" in the default speaker name
        loopback_name = default_speaker.name.replace("Speaker", "Loopback")

        # Get all loopback devices
        all_mics = sc.all_microphones(include_loopback=True)
        loopback_mics = [m for m in all_mics if m.isloopback]

        # Find matching loopback
        matching_loopback = next(
            (mic for mic in loopback_mics if mic.name == loopback_name), None
        )

        if matching_loopback:
            logging.info(f"🔊 Found speaker loopback: {matching_loopback.name}")
            return matching_loopback

        # No matching loopback found
        logging.error(
            f"❌ Loopback for default speaker '{default_speaker.name}' not found."
        )
        logging.info("💡 Available loopback devices:")
        for mic in loopback_mics:
            logging.info(f"   - {mic.name}")
        return None

    except Exception:
        logging.exception("❌ Error selecting speaker loopback.")
        return None


def _find_physical_microphone() -> _Microphone | None:
    """Finds and returns a physical microphone device."""
    try:
        default_mic = sc.default_microphone()
        if not default_mic:
            logging.error(
                "❌ No default microphone found. Microphone audio will not be captured."
            )
            return None

        # Check if default mic is actually a loopback device
        if hasattr(default_mic, "isloopback") and default_mic.isloopback:
            logging.error(
                "❌ Default microphone is a loopback device. Looking for physical microphone..."
            )

            # Get all physical microphones
            physical_mics = sc.all_microphones(include_loopback=False)
            if not physical_mics:
                logging.error("❌ No physical microphone found.")
                return None

            # Use first available physical microphone
            microphone = physical_mics[0]
            logging.info(
                f"🎤 Selected first available physical microphone: {microphone.name}"
            )
            return microphone

        # Default mic is a physical microphone
        logging.info(f"🎤 Found default microphone: {default_mic.name}")
        return default_mic

    except Exception:
        logging.exception("❌ Error selecting microphone.")
        return None


def _select_audio_devices() -> tuple[_Microphone | None, _Microphone | None]:
    """Selects both speaker (loopback) and microphone devices.

    Returns:
        tuple: (speaker_loopback, microphone) - Either can be None if not available
    """
    speaker_loopback = _find_speaker_loopback()
    microphone = _find_physical_microphone()

    # Validate that at least one audio source is available
    if not speaker_loopback and not microphone:
        raise RuntimeError(
            "No audio input devices available. Cannot proceed with transcription."
        )

    # Log warnings for missing devices
    if not speaker_loopback:
        logging.warning(
            "⚠️  Speaker audio will not be captured. Only microphone input will be used."
        )
    if not microphone:
        logging.warning(
            "⚠️  Microphone audio will not be captured. Only speaker output will be used."
        )

    return speaker_loopback, microphone


# ── MAIN APPLICATION LOGIC ─────────────────────────────────
def run_transcription_pipeline(debug: bool = False) -> None:
    """Initializes and runs the transcription pipeline."""
    setup_logging(debug=debug)
    program_start_time_monotonic = time.monotonic()

    _initialize_srt_file(SRT_FILE_PATH)

    try:
        model, vad_detector, loudness_meter = initialize_transcription_resources()
        transcription_engine = TranscriptionEngine(model, loudness_meter)
        speaker_loopback, microphone = _select_audio_devices()

        # Prepare threads list
        threads: list[threading.Thread] = []

        # Instantiate debug manager and per-stream stats loggers BEFORE starting threads
        global DEBUG_MANAGER  # noqa: PLW0603
        DEBUG_MANAGER = DebugManager(enabled=debug)

        # Instantiate global stats loggers (they become no-ops when debug is False)
        global SPEAKER_STATS_LOGGER, MIC_STATS_LOGGER, MIX_STATS_LOGGER, VAD_STATS_LOGGER  # noqa: PLW0603
        SPEAKER_STATS_LOGGER = AudioStatsLogger("Speaker", DEBUG_MANAGER)
        MIC_STATS_LOGGER = AudioStatsLogger("Microphone", DEBUG_MANAGER)
        MIX_STATS_LOGGER = AudioStatsLogger("Mixed", DEBUG_MANAGER)
        VAD_STATS_LOGGER = AudioStatsLogger("VAD", DEBUG_MANAGER)

        # Start speaker capture thread if available
        if speaker_loopback:
            speaker_thread = threading.Thread(
                target=capture_audio_frames,
                args=(
                    speaker_loopback,
                    speaker_audio_queue,
                    "Speaker",
                ),
                daemon=True,
                name="SpeakerCaptureThread",
            )
            threads.append(speaker_thread)
            speaker_thread.start()

        # Start microphone capture thread if available
        if microphone:
            microphone_thread = threading.Thread(
                target=capture_audio_frames,
                args=(
                    microphone,
                    microphone_audio_queue,
                    "Microphone",
                ),
                daemon=True,
                name="MicrophoneCaptureThread",
            )
            threads.append(microphone_thread)
            microphone_thread.start()

        # Start audio mixer thread (uses global MIX_STATS_LOGGER)
        mixer_thread = threading.Thread(
            target=mix_audio_streams,
            daemon=True,
            name="AudioMixerThread",
        )
        threads.append(mixer_thread)
        mixer_thread.start()

        # Start transcription thread
        transcription_thread = threading.Thread(
            target=manage_transcription_workflow,
            args=(
                transcription_engine,
                vad_detector,
                program_start_time_monotonic,
            ),
            daemon=True,
            name="TranscriptionWorkflowThread",
        )
        threads.append(transcription_thread)
        transcription_thread.start()

        # Graceful shutdown handling
        def signal_handler(sig: int, _) -> None:
            logging.info(f"\n🛑 Signal {sig} received. Shutting down gracefully...")
            stop_event.set()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Keep main thread alive while worker threads are running
        while all(thread.is_alive() for thread in threads):
            try:
                time.sleep(MAIN_LOOP_SLEEP_SEC)  # Check periodically
            except InterruptedError:  # Handles Ctrl+C during sleep on some systems
                signal_handler(signal.SIGINT, None)  # type: ignore
                break

        # Ensure threads have a chance to finish after stop_event is set
        logging.info("⏳ Waiting for threads to complete...")

        for thread in threads:
            thread_name = thread.name
            if "Capture" in thread_name or "Mixer" in thread_name:
                thread.join(timeout=THREAD_JOIN_TIMEOUT_SEC)
            else:  # Transcription thread might need more time
                thread.join(timeout=TRANSCRIPTION_THREAD_TIMEOUT_SEC)

            if thread.is_alive():
                logging.warning(f"{thread_name} did not terminate cleanly.")

        logging.info(
            f"\n✅ Transcription complete. Subtitles saved to: {SRT_FILE_PATH.resolve()}"
        )

    except RuntimeError as e:
        logging.error(f"❌ {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(
            f"An unhandled error occurred in the main application: {e}", exc_info=True
        )
        sys.exit(1)

    finally:
        logging.info("👋 Exiting application.")


# Create typer app
app = typer.Typer(help="Realtime ASR with faster-whisper — live console and SRT output")


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
    output: Path = typer.Option(
        SRT_FILE_PATH, "--output", help=f"Output SRT file path"
    ),
):
    """
    Realtime ASR with faster-whisper — live console and SRT output

    Examples:

        python even_better_whisper_transcribe.py

        python even_better_whisper_transcribe.py --debug

        python even_better_whisper_transcribe.py --output my.srt

        python even_better_whisper_transcribe.py --speaker-ratio 0.7 --mic-ratio 0.3
    """
    # Update global settings
    global SRT_FILE_PATH
    SRT_FILE_PATH = output

    try:
        run_transcription_pipeline(debug=debug)
    except KeyboardInterrupt:
        typer.echo("\n\n🛑 Interrupted by user. Shutting down...")
        raise typer.Exit(0)
    except Exception as e:
        typer.echo(f"\n❌ Fatal error: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
