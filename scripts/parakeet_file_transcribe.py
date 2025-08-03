import gc
import logging
import time
from pathlib import Path

import nemo.collections.asr as nemo_asr
import numpy as np
import scipy.signal
import torch
import typer
import webrtcvad
from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from pydub import AudioSegment
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import track

# --- Constants ---
LONG_AUDIO_THRESHOLD_MS = 600_000  # 10 minutes in milliseconds
MAX_SEGMENT_DURATION_MS = 480_000  # 8 minutes in milliseconds
VAD_SAMPLE_RATE = 16_000
VAD_FRAME_DURATION_MS = 30
VAD_AGGRESSIVENESS = 3
SUBSAMPLING_FACTOR = 8
MILLISECONDS_PER_SECOND = 1000.0  # 1 second = 1000 milliseconds
INT16_MAX = 32767


# --- Helper functions for time conversions ---
def ms_to_sec(ms: float | int) -> float:
    """Convert milliseconds to seconds."""
    return ms / MILLISECONDS_PER_SECOND


def sec_to_ms(sec: float | int) -> float:
    """Convert seconds to milliseconds."""
    return sec * MILLISECONDS_PER_SECOND


# --- Console ---
console = Console()

# --- Logger Setup ---
logging.basicConfig(
    level=logging.INFO,
    handlers=[RichHandler(rich_tracebacks=True, console=console)],
)

logger = logging.getLogger("ParakeetTranscribe")


def _split_audio_by_silence(audio: AudioSegment) -> list[tuple[AudioSegment, float]]:
    """長時間音声を無音区間で分割します。"""
    audio_length_ms = len(audio)
    if audio_length_ms <= LONG_AUDIO_THRESHOLD_MS:
        return [(audio, 0.0)]

    logger.info(f"長時間音声({ms_to_sec(audio_length_ms):.1f}秒)を分割します...")

    # 音声データ準備
    audio_data = np.array(audio.get_array_of_samples(), dtype=np.float32)
    if audio.channels > 1:
        audio_data = audio_data.mean(axis=1)

    # VAD用16kHzリサンプル
    original_sr = audio.frame_rate
    if original_sr != VAD_SAMPLE_RATE:
        resample_ratio = VAD_SAMPLE_RATE / original_sr
        resampled_length = int(len(audio_data) * resample_ratio)
        vad_audio = (
            scipy.signal.resample(audio_data, resampled_length).astype(np.int16)
            * INT16_MAX
        )
    else:
        vad_audio = (audio_data * INT16_MAX).astype(np.int16)
        resample_ratio = 1.0

    # VAD無音検出
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    frame_length = int(VAD_SAMPLE_RATE * ms_to_sec(VAD_FRAME_DURATION_MS))

    split_points_ms = []
    # フレームごとに無音検出 (終端の小さなフレームは無視)
    for i in range(0, len(vad_audio) - frame_length, frame_length):
        if not vad.is_speech(
            vad_audio[i : i + frame_length].tobytes(), VAD_SAMPLE_RATE
        ):
            time_ms = int(sec_to_ms(i / original_sr / resample_ratio))
            split_points_ms.append(time_ms)

    # 分割実行
    segments = []
    current_segment_start_ms = 0

    for split_point_ms in split_points_ms:
        segment_duration_ms = split_point_ms - current_segment_start_ms
        if segment_duration_ms >= MAX_SEGMENT_DURATION_MS:
            segments.append(
                (
                    audio[current_segment_start_ms:split_point_ms],
                    ms_to_sec(current_segment_start_ms),
                )
            )
            current_segment_start_ms = split_point_ms

    # 最終セグメント
    if current_segment_start_ms < audio_length_ms:
        segments.append(
            (audio[current_segment_start_ms:], ms_to_sec(current_segment_start_ms))
        )

    # フォールバック：時間分割
    if len(segments) <= 1:
        for segment_start_ms in range(0, audio_length_ms, MAX_SEGMENT_DURATION_MS):
            segment_end_ms = min(
                segment_start_ms + MAX_SEGMENT_DURATION_MS, audio_length_ms
            )
            segments.append(
                (
                    audio[segment_start_ms:segment_end_ms],
                    ms_to_sec(segment_start_ms),
                )
            )

    logger.info(f"{len(segments)}個のセグメントに分割完了")
    return segments


def _adjust_hypothesis_timestamps(
    hypotheses: list[Hypothesis], time_offset_seconds: float, time_stride: float
):
    """仮説のタイムスタンプを調整します。"""
    time_offset_frames = time_offset_seconds / time_stride

    for hyp in hypotheses:
        if not (hasattr(hyp, "timestamp") and hyp.timestamp["segment"]):
            logger.warning(f"タイムスタンプが見つかりません: {hyp.text}")
            continue

        for ts in hyp.timestamp["segment"]:
            ts["start_offset"] += time_offset_frames
            ts["end_offset"] += time_offset_frames


def _setup_model(
    model_name: str, device: str
) -> tuple[EncDecRNNTBPEModel | None, float]:
    """ASRモデルをロードし、タイムスタンプ計算のためのtime_strideを決定します。"""
    logger.info(f"'{model_name}' モデルをロードしています...")
    try:
        asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
        if not isinstance(asr_model, EncDecRNNTBPEModel):
            msg = f"モデルが予期しない型です: {type(asr_model).__name__}"
            logger.error(msg)
            raise ValueError(msg)

        asr_model.to(torch.bfloat16)
        asr_model.to(device)

        # タイムスタンプ計算に必要なtime_strideを取得します。
        # Parakeet (FastConformer)モデルのエンコーダは、サブサンプリング係数が8で固定されています。
        # `window_stride`は、STFT（短時間フーリエ変換）でウィンドウをスライドさせるステップサイズ（秒）です。
        # これらの積が、タイムスタンプの各オフセット単位が何秒に対応するかを決定します。
        time_stride = SUBSAMPLING_FACTOR * asr_model.cfg.preprocessor.window_stride

    except Exception as e:
        msg = "モデルのロード中にエラーが発生しました。"
        logger.exception(msg)
        raise ValueError(msg) from e

    else:
        logger.info("モデルのロードが完了しました。")
        return asr_model, time_stride


def _find_audio_files(input_dir: Path) -> list[Path]:
    """入力ディレクトリからサポートされている音声ファイルを再帰的に検索します。"""
    audio_extensions = [".wav", ".mp3", "wma", ".flac", ".m4a", ".ogg", ".aac"]
    logger.info(
        f"ディレクトリ '{input_dir}' 内の音声ファイルを再帰的に検索しています..."
    )
    audio_files = sorted(
        [p for p in input_dir.rglob("*") if p.suffix.lower() in audio_extensions]
    )
    if not audio_files:
        logger.warning(
            f"対応する拡張子 {audio_extensions} の音声ファイルが見つかりませんでした。"
        )
    else:
        logger.info(f"{len(audio_files)}個の音声ファイルを処理します。")
    return audio_files


def _run_transcription(
    asr_model: EncDecRNNTBPEModel,
    audio_path: Path,
    batch_size: int,
) -> list[Hypothesis] | None:
    """
    指定された単一の音声ファイルを前処理し、文字起こしを実行します。
    - モノラル変換
    - モデルの要求仕様に合わせたリサンプリング
    """
    logger.info(f"'{audio_path.name}' の文字起こしを開始します...")
    target_sr = asr_model.cfg.preprocessor.sample_rate

    try:
        audio = AudioSegment.from_file(audio_path)
        # 1. モノラルに変換
        if audio.channels > 1:
            audio = audio.set_channels(1)
        # 2. モデルのサンプリングレートにリサンプリング
        if audio.frame_rate != target_sr:
            audio = audio.set_frame_rate(target_sr)

        # 3. 長時間音声の分割処理
        segments = _split_audio_by_silence(audio)
        path_hypotheses = []

        for idx, (segment_audio, time_offset_seconds) in enumerate(segments):
            logger.info(f"Processing segment: {idx} / {len(segments)}")

            # NumPy配列に変換 (float32)
            audio_sample = np.array(segment_audio.get_array_of_samples()).astype(
                np.float32
            )

            # --- 動的な正規化 ---
            # sample_width (バイト数) に応じて適切な整数型の最大値で割る
            sample_width = segment_audio.sample_width
            if sample_width == 2:  # 16-bit audio
                max_val = np.iinfo(np.int16).max
            elif sample_width == 4:  # 32-bit audio
                max_val = np.iinfo(np.int32).max
            elif sample_width == 1:  # 8-bit audio
                max_val = np.iinfo(np.int8).max
            else:
                # サポート外のビット深度ではエラーを発生させ、このファイルの処理を中断する
                msg = (
                    f"Unsupported sample width for normalization: {sample_width} bytes."
                )
                logger.error(msg)
                raise ValueError(msg)

            if max_val:
                audio_sample /= max_val

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            segment_hypotheses = asr_model.transcribe(
                audio=audio_sample,
                batch_size=batch_size,
                return_hypotheses=True,
                timestamps=True,
            )

            # タイムスタンプ調整
            time_stride = SUBSAMPLING_FACTOR * asr_model.cfg.preprocessor.window_stride
            _adjust_hypothesis_timestamps(
                segment_hypotheses, time_offset_seconds, time_stride
            )

            path_hypotheses.extend(segment_hypotheses)

    except Exception as e:
        msg = f"ファイル処理中にエラーが発生しました: {audio_path.name}"
        logger.exception(msg)
        raise ValueError(msg) from e

    else:
        return path_hypotheses


def _format_timestamp_for_srt(seconds: float) -> str:
    """秒をSRT形式のタイムスタンプ文字列（HH:MM:SS,ms）に変換します。"""
    seconds = max(seconds, 0)
    milliseconds = int((seconds % 1) * 1000)
    return time.strftime("%H:%M:%S", time.gmtime(seconds)) + f",{milliseconds:03d}"


def _save_srt_file(
    hypotheses: list[Hypothesis],
    time_stride: float,
    audio_file: Path,
):
    """文字起こしの仮説からSRTファイルを生成して保存します。"""
    logger.info(f"'{audio_file.name}' のSRTファイルを保存しています...")

    try:
        srt_lines = []
        srt_index = 1
        last_end_time = 0.0

        for hyp in hypotheses:
            if not isinstance(hyp, Hypothesis):
                logger.warning(
                    f"予期しない型の仮説 ({type(hyp).__name__}) をスキップします: {audio_file.name}"
                )
                continue

            timestamp_data = hyp.timestamp
            segments = timestamp_data.get("segment")

            if segments:
                # セグメント内のタイムスタンプはソートする
                segments.sort(key=lambda s: s.get("start_offset", 0))
                for seg_data in segments:
                    text = seg_data.get("segment", "").strip()
                    start = seg_data.get("start_offset")
                    end = seg_data.get("end_offset")

                    start_time = start * time_stride
                    end_time = end * time_stride
                    start_str = _format_timestamp_for_srt(start_time)
                    end_str = _format_timestamp_for_srt(end_time)

                    srt_lines.append(str(srt_index))
                    srt_lines.append(f"{start_str} --> {end_str}")
                    srt_lines.append(text)
                    srt_lines.append("")

                    srt_index += 1
                    last_end_time = max(last_end_time, end_time)
            else:
                logger.warning("segmentが見つからないため、テキストのみを使用します。")

                text = hyp.text.strip()
                # 最後のタイムスタンプを基点に期間ゼロのタイムスタンプを割り当てる
                time = last_end_time + 0.001 if srt_lines else 0.0
                time_str = _format_timestamp_for_srt(time)

                srt_lines.append(str(srt_index))
                srt_lines.append(f"{time_str} --> {time_str}")
                srt_lines.append(text)
                srt_lines.append("")

                srt_index += 1
                last_end_time = time

        if not srt_lines:
            logger.warning(
                f"{audio_file.name} から有効な文字起こし結果を生成できませんでした。"
            )
            return

        combined_content = "\n".join(srt_lines)
        output_path = audio_file.with_suffix(".srt")
        output_path.write_text(combined_content, encoding="utf-8")
        logger.info(f"SRTファイルを保存しました: {output_path}")
    except Exception as e:
        msg = f"{audio_file.name} のSRTファイル保存中にエラーが発生しました。"
        logger.exception(msg)
        raise ValueError(msg) from e


def _transcribe_with_parakeet(input_dir: Path, model_name: str, batch_size: int):
    """ディレクトリ内のすべての音声ファイルを文字起こしするメインワークフロー。"""
    # 1. デバイスのセットアップ
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        logger.info(f"GPUが利用可能です。デバイス: {device}")
    else:
        logger.warning(
            "GPUが利用できません。CPUで実行します（処理に時間がかかります）。"
        )

    # 2. モデルのロード
    asr_model, time_stride = _setup_model(model_name, device)
    if not asr_model:
        raise typer.Exit(code=1)

    # 3. 音声ファイルの検索
    audio_files = _find_audio_files(input_dir)
    if not audio_files:
        return

    logger.info(f"audio_files: {audio_files}")

    # 4. ファイルごとに文字起こしと保存を実行
    for audio_path in track(
        audio_files, description="ファイル処理中...", console=console
    ):
        try:
            hypotheses = _run_transcription(asr_model, audio_path, batch_size)
            _save_srt_file(hypotheses, time_stride, audio_path)
        except Exception:
            logger.exception(f"ファイル処理中にエラーが発生しました: {audio_path.name}")
            continue

    logger.info("すべてのファイルの処理が完了しました。")


app = typer.Typer(
    help="ディレクトリ内の音声ファイルを再帰的に検索し、NVIDIA NeMo Parakeetモデルで文字起こししてSRTとして保存します。",
    pretty_exceptions_show_locals=False,
)


@app.command()
def main(
    input_dir: Path = typer.Argument(
        ...,
        help="音声ファイルが含まれるディレクトリのパス。",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
    ),
    model_name: str = typer.Option(
        "nvidia/parakeet-tdt-0.6b-v2",
        "--model",
        "-m",
        help="使用するNVIDIA NeMoのASRモデル名。",
    ),
    batch_size: int = typer.Option(
        16,
        "--batch-size",
        "-b",
        help="文字起こし処理のバッチサイズ。",
    ),
):
    """CLI entrypoint."""
    _transcribe_with_parakeet(
        input_dir=input_dir, model_name=model_name, batch_size=batch_size
    )


if __name__ == "__main__":
    app()
