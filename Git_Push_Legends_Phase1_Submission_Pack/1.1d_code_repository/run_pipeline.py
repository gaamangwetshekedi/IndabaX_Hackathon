#!/usr/bin/env python3
"""Run the complete Git Push Legends Phase 1 pipeline."""

from pathlib import Path

from src.pipeline import run


if __name__ == "__main__":
    result = run(Path(__file__).resolve().parent)
    print(f"team={result['team']}")
    print(f"selected_model={result['winner']}")
    for model, metrics in result["metrics"].items():
        print(
            f"model={model} rmse={metrics['rmse']:.4f} "
            f"mae={metrics['mae']:.4f} smape={metrics['smape']:.2f}%"
        )
    print("predictions=outputs/predictions.csv")
    print("reports=output/pdf")

