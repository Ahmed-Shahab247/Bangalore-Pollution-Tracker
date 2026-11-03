"""Train, monthly-backtest, and save Bengaluru PM2.5 forecasting models."""

import argparse
from pathlib import Path

from app.forecasting import (
    GROUND_STATION_DATA_PATH,
    MIN_STATIONS_PER_HOUR,
    MODEL_PATH,
    WEATHER_DATA_PATH,
    train_forecaster,
)


def _print_evaluation(bundle):
    print("horizon  holdout    model                    n      MAE      RMSE")
    for horizon, result in bundle["models"].items():
        evaluation = result["evaluation"]
        metrics = evaluation["overall"]
        for model_name, scores in metrics.items():
            print(
                f"{horizon:>3} h    pooled      {model_name:<24} "
                f"{evaluation['evaluated_examples']:>5} "
                f"{scores['mae']:>8.2f} {scores['rmse']:>8.2f}"
            )
        for month, holdout in evaluation["monthly_holdouts"].items():
            for model_name, scores in holdout["metrics"].items():
                print(
                    f"{horizon:>3} h    {month:<12} {model_name:<24} "
                    f"{holdout['test_examples']:>5} "
                    f"{scores['mae']:>8.2f} {scores['rmse']:>8.2f}"
                )
        print(
            f"{horizon}-hour selection: {result['selected_model']} "
            f"(MAE {result['model_metrics']['mae']:.2f}); "
            f"persistence skill={result['skill_score_vs_persistence']:.3f}; "
            f"MAE difference vs persistence 95% block-bootstrap CI="
            f"{evaluation['persistence_mae_difference_ci']['low']:.2f} to "
            f"{evaluation['persistence_mae_difference_ci']['high']:.2f}."
        )
        print(
            f"Trained on {result['sample_count']} city-hour examples with at least "
            f"the required {MIN_STATIONS_PER_HOUR} reporting stations across "
            f"{result['station_count']} stations."
        )
        if evaluation["skipped_months"]:
            print(f"Skipped months: {evaluation['skipped_months']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weather",
        type=Path,
        default=WEATHER_DATA_PATH,
        help="Historical forecast-weather CSV used only for model features.",
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
