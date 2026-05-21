"""Export LangGraph structure to terminal and artifact files."""

from __future__ import annotations

from pathlib import Path

from utils.config import get_config, resolve_path
from utils.logging import get_logger

logger = get_logger(__name__)


def export_graph_artifact(compiled_graph) -> Path:
    cfg = get_config()
    out_dir = resolve_path(cfg["paths"]["graph_artifact_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    graph = compiled_graph.get_graph()
    mermaid = graph.draw_mermaid()
    mmd_path = out_dir / "investigator_graph.mmd"
    mmd_path.write_text(mermaid, encoding="utf-8")

    ascii_art = ""
    try:
        ascii_art = graph.draw_ascii()
    except Exception:
        ascii_art = "(ascii render unavailable)"

    txt_path = out_dir / "investigator_graph.txt"
    txt_path.write_text(ascii_art + "\n\n--- mermaid ---\n\n" + mermaid, encoding="utf-8")

    print("\n=== Fraud Investigator Twin — LangGraph ===\n")
    print(ascii_art)
    print(f"\nMermaid saved to: {mmd_path}")
    print(f"Full artifact: {txt_path}\n")
    logger.info("Graph artifact written to %s", out_dir)
    return out_dir
