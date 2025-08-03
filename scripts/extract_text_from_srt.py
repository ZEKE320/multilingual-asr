#!/usr/bin/env python3
"""
SRTファイルから文字起こしテキストのみを抽出するスクリプト
"""

from pathlib import Path
from typing import Optional

import typer


def extract_text_from_srt(srt_file_path: Path) -> list[str]:
    """
    SRTファイルから文字起こしテキストのみを抽出する

    Args:
        srt_file_path: SRTファイルのパス

    Returns:
        文字起こしテキストのリスト
    """
    with open(srt_file_path, "r", encoding="utf-8") as file:
        content = file.read()

    # SRTブロックを分割 (空行で区切られている)
    blocks = content.strip().split("\n\n")

    text_lines = []

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) >= 3:
            # 1行目: 字幕番号
            # 2行目: タイムスタンプ
            # 3行目以降: テキスト
            text_content = "\n".join(lines[2:])
            text_lines.append(text_content)

    return text_lines


def main(
    srt_file: Path = typer.Argument(..., help="SRTファイルのパス"),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="出力ファイルのパス (デフォルト: SRTファイル名.txt)",
    ),
):
    """
    SRTファイルから文字起こしテキストのみを抽出する
    """
    if not srt_file.exists():
        typer.echo(f"エラー: ファイル '{srt_file}' が見つかりません", err=True)
        raise typer.Exit(1)

    try:
        text_lines = extract_text_from_srt(srt_file)

        # 出力ファイル名を決定（デフォルトはSRTファイル名.txt）
        if output is None:
            output = srt_file.with_suffix(".txt")

        # テキストファイルに出力
        with open(output, "w", encoding="utf-8") as f:
            for text in text_lines:
                f.write(text + "\n")

        typer.echo(f"抽出したテキストを {output} に保存しました")

    except Exception as e:
        typer.echo(f"エラー: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
