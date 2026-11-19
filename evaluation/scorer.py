def f05_for_entity(true_ids, predicted_ids):
    true_ids = set(true_ids)
    predicted_ids = set(predicted_ids)

    # True singleton
    if not true_ids:
        return 1.0 if not predicted_ids else 0.0

    # Missed all true matches
    if not predicted_ids:
        return 0.0

    true_positives = len(true_ids & predicted_ids)

    precision = true_positives / len(predicted_ids)
    recall = true_positives / len(true_ids)

    if precision == 0 and recall == 0:
        return 0.0

    return (1.25 * precision * recall) / (0.25 * precision + recall)


def macro_f05(ground_truth, predictions):
    scores = []

    for source1_id, true_ids in ground_truth.items():
        predicted_ids = predictions.get(source1_id, [])

        score = f05_for_entity(
            true_ids,
            predicted_ids
        )

        scores.append(score)

    if not scores:
        return 0.0

    return sum(scores) / len(scores)


if __name__ == "__main__":

    # Basic tests
    print("Test 1:", f05_for_entity([], []))
    print("Test 2:", f05_for_entity([], ["S2-001"]))
    print("Test 3:", f05_for_entity(["S2-001"], ["S2-001"]))
    print("Test 4:", f05_for_entity(["S2-001"], []))
    print(
        "Test 5:",
        f05_for_entity(
            ["S2-001", "S2-002"],
            ["S2-001"]
        )
    ) 