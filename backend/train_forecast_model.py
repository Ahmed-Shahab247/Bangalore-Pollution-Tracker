"""Train, seasonally evaluate, and save Bengaluru PM2.5 forecasting models."""

import argparse
from pathlib import Path

from app.forecasting import (
    GROUND_STATION_DATA_PATH,
    MODEL_PATH,
    WEATHER_DATA_PATH,
    train_forecaster,
)


def _print_evaluation(bundle):
    print(
        "horizon  holdout    model                    n      MAE      RMSE"
    )
    for horizon, result in bundle["models"].items():
        evaluation = result["evaluation"]
        metrics = evaluation["overall"]
        for model_name, scores in metrics.items():
            print(
                f"{horizon:>3} h    all-seasons {model_name:<24} "
                f"{evaluation['evaluated_examples']:>5} "
                f"{scores['mae']:>8.2f} {scores['rmse']:>8.2f}"
            )
        for season, holdout in evaluation["seasonal_holdouts"].items():
            for model_name, scores in holdout["metrics"].items():
                print(
                    f"{horizon:>3} h    {season:<12} {model_name:<24} "
                    f"{holdout['test_examples']:>5} "
                    f"{scores['mae']:>8.2f} {scores['rmse']:>8.2f}"
                )
        print(
            f"{horizon}-hour model trained on {result['sample_count']} "
            f"complete city-hour examples across {result['station_count']} stations."
        )
        if evaluation["skipped_seasons"]:
            print(f"Skipped seasons: {evaluation['skipped_seasons']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weather",
        type=Path,
        default=WEATHER_DATA_PATH,
        help="Open-Meteo historical weather CSV used only for model features.",
    )
    parser.add_argument(
        "--ground-stations",
        type=Path,
        default=GROUND_STATION_DATA_PATH,
        help="CPCB/KSPCB-attributed PM2.5 observation CSV used as targets.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        help="Output joblib model path.",
    )
    args = parser.parse_args()

    bundle = train_forecaster(
        weather_path=args.weather,
        ground_station_path=args.ground_stations,
        model_path=args.model,
    )
    _print_evaluation(bundle)
    print(f"Weather features: {args.weather}")
    print(f"Ground-station targets: {args.ground_stations}")
    print(f"Saved model: {args.model}")


if __name__ == "__main__":
    main()
