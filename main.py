#!/usr/bin/env python3
"""Fraud Investigator Twin — local PoC entry point."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Ensure project root on path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from agents.graph import run_investigation, seed_memory_from_file
from evaluation.harness import run_evaluation
from tools.case_loader import load_case_file
from utils.config import reload_config, resolve_path
from utils.logging import setup_logging

app = typer.Typer(help="Fraud Investigator Twin — local digital twin PoC")
console = Console()


def _investigate_case(case_path: str, output: str | None = None) -> None:
    """Core investigate logic — safe to call from Typer commands or demo()."""
    reload_config()
    setup_logging()
    seed_memory_from_file()
    raw = load_case_file(case_path)
    result = run_investigation(raw)
    _print_result(result)
    if output:
        out_path = resolve_path(output) if not Path(output).is_absolute() else Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"[green]Wrote[/green] {out_path}")


@app.command()
def investigate(
    case_path: str = typer.Argument(..., help="Path to case JSON file"),
    output: str | None = typer.Option(None, "--output", "-o", help="Write result JSON to path"),
):
    """Run full investigation pipeline on one case."""
    _investigate_case(case_path, output=output)


@app.command()
def graph():
    """Print LangGraph structure and save Mermaid/ASCII artifacts."""
    setup_logging()
    from agents.graph import build_investigator_graph

    build_investigator_graph(export_graph=True)
    console.print("[green]Graph artifact exported to data/artifacts/[/green]")


@app.command()
def seed():
    """Seed long-term memory from data/seed_cases.json."""
    setup_logging()
    n = seed_memory_from_file(force=True)
    console.print(f"[green]Seeded {n} cases[/green]")


@app.command()
def evaluate(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run evaluation harness on data/eval_cases.json."""
    setup_logging()
    metrics = run_evaluation(verbose=verbose)
    table = Table(title="Evaluation metrics")
    table.add_column("Metric")
    table.add_column("Value")
    for key in ("accuracy", "macro_f1", "false_positive_rate_fraud", "false_negative_rate_fraud",
                "explanation_quality_proxy_mean", "retrieval_relevance_mean", "consistency_rate", "cases_evaluated"):
        if key in metrics:
            table.add_row(key, str(metrics[key]))
    console.print(table)
    if verbose and metrics.get("per_case"):
        for row in metrics["per_case"]:
            console.print(row)
    out = resolve_path("data/eval_report.json")
    out.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    console.print(f"[dim]Full report: {out}[/dim]")


@app.command()
def demo():
    """Run demo case from data/cases/sample_case.json."""
    path = resolve_path("data/cases/sample_case.json")
    _investigate_case(str(path), output="data/demo_sample_case.json")


def _print_result(result) -> None:
    exp = result.explainability
    console.print(
        Panel(
            f"[bold]{result.decision.value.upper()}[/bold]  (confidence {result.confidence:.0%})\n\n"
            f"{exp.analyst_rationale}\n\n"
            f"[dim]Top reasons:[/dim] " + "; ".join(exp.top_reasons[:3]),
            title=f"Case {result.case_id}",
            border_style="blue",
        )
    )
    console.print("\n[bold]Explainability sections[/bold]")
    for k, v in exp.to_analyst_sections().items():
        console.print(f"  [cyan]{k}[/cyan]: {str(v)[:200]}..." if len(str(v)) > 200 else f"  [cyan]{k}[/cyan]: {v}")

    if result.reward_score_output:
        r = result.reward_score_output
        console.print("\n[bold magenta]Reward Alignment Scoring[/bold magenta]")
        console.print(f"  [magenta]Reward Score:[/magenta] {r.reward_score:.2f}")
        console.print(f"  [magenta]Confidence Alignment:[/magenta] {r.confidence_alignment:.2f}")
        console.print(f"  [magenta]Top Positive Signals:[/magenta] {', '.join(r.top_positive_signals)}")
        console.print(f"  [magenta]Top Negative Signals:[/magenta] {', '.join(r.top_negative_signals)}")
        console.print(f"  [magenta]Explanation:[/magenta] {r.reward_explanation}")


if __name__ == "__main__":
    app()
