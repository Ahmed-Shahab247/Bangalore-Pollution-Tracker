"""Train and evaluate the Open-Meteo city-level PM2.5 forecaster."""

from app.forecasting import DATA_PATH, MODEL_PATH, train_forecaster


def main():
    bundle = train_forecaster()
    for horizon, result in bundle["models"].items():
        print(
            f"{horizon}-hour forecast: {result['sample_count']} examples; "
            f"model MAE {result['model_metrics']['mae']:.2f} µg/m³; "
            f"persistence MAE {result['persistence_metrics']['mae']:.2f}; "
            f"24-hour seasonal MAE {result['seasonal_metrics']['mae']:.2f}; "
            f"beats both baselines: {result['beats_baselines']}."
        )
    print(f"Training data: {DATA_PATH}.")
    print(f"Saved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
