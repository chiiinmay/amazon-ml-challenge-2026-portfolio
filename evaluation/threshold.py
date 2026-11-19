import pandas as pd
import numpy as np


PREDICTION_PATH = r"C:\Users\User\Downloads\Telegram Desktop\prediction_scores.tsv"
GROUND_TRUTH_PATH = "dataset/train/train_ground_truth.tsv"


def f05(true_ids, predicted_ids):
    true_ids = set(true_ids)
    predicted_ids = set(predicted_ids)

    # No true matches: correct empty prediction = 1.0
    # Any predicted match = 0.0
    if not true_ids:
        return 1.0 if not predicted_ids else 0.0

    # True matches exist, but model predicted nothing
    if not predicted_ids:
        return 0.0

    true_positives = len(true_ids & predicted_ids)

    precision = true_positives / len(predicted_ids)
    recall = true_positives / len(true_ids)

    # No overlap between true and predicted matches
    if precision == 0 or recall == 0:
        return 0.0

    return (
        1.25 * precision * recall
        / (0.25 * precision + recall)
    )


def load_ground_truth(path):

    df = pd.read_csv(
        path,
        sep="\t",
        keep_default_na=False
    )

    ground_truth = {}

    for _, row in df.iterrows():

        source1_id = row["source1_entity_id"]
        matched = row["matched_entity_ids"]

        if matched.strip() == "":
            matched_ids = []
        else:
            matched_ids = [
                x.strip()
                for x in matched.split(",")
                if x.strip()
            ]

        ground_truth[source1_id] = matched_ids

    return ground_truth


def evaluate_threshold(
    predictions,
    ground_truth,
    evaluated_ids,
    threshold
):

    selected = predictions[
        predictions["match_probability"] >= threshold
    ]

    predicted = {}

    for source1_id, group in selected.groupby(
        "source1_entity_id"
    ):

        predicted[source1_id] = group[
            "candidate_entity_id"
        ].tolist()

    scores = []

    # IMPORTANT:
    # Evaluate all 5,000 dev entities,
    # including entities for which the model
    # predicts nothing.
    for source1_id in evaluated_ids:

        true_ids = ground_truth.get(
            source1_id,
            []
        )

        predicted_ids = predicted.get(
            source1_id,
            []
        )

        scores.append(
            f05(
                true_ids,
                predicted_ids
            )
        )

    return float(np.mean(scores))


if __name__ == "__main__":

    print("Loading prediction scores...")

    predictions = pd.read_csv(
        PREDICTION_PATH,
        sep="\t"
    )

    predictions["source1_entity_id"] = (
        predictions["source1_entity_id"].astype(str)
    )

    predictions["candidate_entity_id"] = (
        predictions["candidate_entity_id"].astype(str)
    )

    print(
        "Prediction rows:",
        len(predictions)
    )

    evaluated_ids = set(
        predictions["source1_entity_id"]
    )

    print(
        "Dev Source 1 entities:",
        len(evaluated_ids)
    )

    print("Loading ground truth...")

    ground_truth = load_ground_truth(
        GROUND_TRUTH_PATH
    )

    # Make sure every dev S1 has a ground-truth entry.
    missing_ground_truth = (
        evaluated_ids - set(ground_truth)
    )

    print(
        "Missing ground truth:",
        len(missing_ground_truth)
    )

    if missing_ground_truth:
        raise ValueError(
            "Some B dev entities are missing "
            "from ground truth."
        )

    print("\nThreshold sweep:\n")

    thresholds = np.arange(
        0.10,
        1.00,
        0.05
    )

    results = []

    for threshold in thresholds:

        score = evaluate_threshold(
            predictions,
            ground_truth,
            evaluated_ids,
            threshold
        )

        results.append(
            (threshold, score)
        )

        print(
            f"Threshold {threshold:.2f}"
            f" -> F0.5 = {score:.6f}"
        )

    best_threshold, best_score = max(
        results,
        key=lambda x: x[1]
    )

    print("\n==============================")
    print("BEST VALIDATION THRESHOLD")
    print("==============================")
    print(
        f"Threshold: {best_threshold:.2f}"
    )
    print(
        f"F0.5: {best_score:.6f}"
    ) 