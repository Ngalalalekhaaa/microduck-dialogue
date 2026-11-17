"""Report skill-selection quality separately from easy feedback responses."""

from collections import Counter, defaultdict

import numpy as np

from .toy_protocol import DECISIONS
from .toy_selector_labels import EOS, FIELDS, MOTION, canonical_plan, encode_targets, goal_label


def ratio(correct, count):
    return dict(correct=int(correct), support=int(count), accuracy=float(correct / count) if count else None)


def decision_metrics(expected, predicted):
    matrix = np.zeros((len(DECISIONS), len(DECISIONS)), dtype=np.int64)
    for truth, guess in zip(expected, predicted, strict=True):
        matrix[DECISIONS.index(truth), DECISIONS.index(guess)] += 1
    supports, guesses = matrix.sum(1), matrix.sum(0)
    per_class = {}
    for index, name in enumerate(DECISIONS):
        correct, support = int(matrix[index, index]), int(supports[index])
        per_class[name] = dict(
            support=support,
            predicted=int(guesses[index]),
            recall=correct / support if support else None,
            f1=2 * correct / int(supports[index] + guesses[index]) if support else None,
        )
    present = [row["f1"] for row in per_class.values() if row["support"]]
    return dict(
        accuracy=float(matrix.trace() / matrix.sum()) if matrix.sum() else None,
        macro_f1_present=float(np.mean(present)) if present else None,
        per_class=per_class,
        confusion_matrix=matrix.tolist(),
        class_order=list(DECISIONS),
    )


def evaluate(rows, predictions, train_plans, *, raw_tokens=None):
    if len(rows) != len(predictions):
        raise ValueError("Predictions and source rows must align exactly.")
    targets = [row["target"] for row in rows]
    answer = {}
    for scope in ("all", "user", "feedback"):
        indices = [i for i, row in enumerate(rows) if scope == "all" or row["request"]["event"]["kind"] == scope]
        answer[f"decision_{scope}"] = decision_metrics(
            [targets[i]["decision"] for i in indices], [predictions[i]["decision"] for i in indices]
        )
    action_indices = [i for i, target in enumerate(targets) if target["decision"] in MOTION]
    exact = [
        target["decision"] == prediction["decision"]
        and canonical_plan(target["goals"]) == canonical_plan(prediction["goals"])
        for target, prediction in zip(targets, predictions, strict=True)
    ]
    answer["semantic_exact_all"] = ratio(sum(exact), len(rows))
    answer["action_plan_exact"] = ratio(sum(exact[i] for i in action_indices), len(action_indices))
    by_length, by_novelty, by_family = defaultdict(list), defaultdict(list), defaultdict(list)
    fields = defaultdict(list)
    for index in action_indices:
        expected, predicted = targets[index]["goals"], predictions[index]["goals"]
        by_length[str(len(expected))].append(exact[index])
        novelty = "seen_plan" if canonical_plan(expected) in train_plans else "unseen_plan"
        by_novelty[novelty].append(exact[index])
        by_family[rows[index]["split_group"]].append(exact[index])
        for position, goal in enumerate(expected):
            token, attributes = goal_label(goal)
            if position < len(predicted):
                guessed_token, guessed_attributes = goal_label(predicted[position])
            else:
                guessed_token, guessed_attributes = None, {}
            fields["semantic_token"].append(token == guessed_token)
            fields["goal_exact"].append(token == guessed_token and attributes == guessed_attributes)
            for name in FIELDS:
                if attributes[name] != -100:
                    fields[name].append(attributes[name] == guessed_attributes.get(name))
    answer["action_by_length"] = {
        str(length): ratio(sum(by_length[str(length)]), len(by_length[str(length)])) for length in range(1, 7)
    }
    answer["action_by_plan_novelty"] = {
        key: ratio(sum(by_novelty[key]), len(by_novelty[key])) for key in ("seen_plan", "unseen_plan")
    }
    answer["action_by_family"] = {key: ratio(sum(values), len(values)) for key, values in by_family.items()}
    answer["action_fields"] = {key: ratio(sum(values), len(values)) for key, values in fields.items()}
    answer["decoded_action_length"] = ratio(
        sum(len(targets[i]["goals"]) == len(predictions[i]["goals"]) for i in action_indices), len(action_indices)
    )
    nonmotion = [i for i, target in enumerate(targets) if target["decision"] not in MOTION]
    stop = [i for i, target in enumerate(targets) if target["decision"] == "cancel"]
    answer["false_start"] = ratio(sum(predictions[i]["decision"] in MOTION for i in nonmotion), len(nonmotion))
    answer["false_start"]["count"] = answer["false_start"].pop("correct")
    answer["false_start"]["rate"] = answer["false_start"].pop("accuracy")
    answer["missed_cancel"] = ratio(sum(predictions[i]["decision"] != "cancel" for i in stop), len(stop))
    answer["missed_cancel"]["count"] = answer["missed_cancel"].pop("correct")
    answer["missed_cancel"]["rate"] = answer["missed_cancel"].pop("accuracy")
    for key, indices, predicate in (
        ("false_start_user", nonmotion, lambda index: predictions[index]["decision"] in MOTION),
        ("missed_cancel_user", stop, lambda index: predictions[index]["decision"] != "cancel"),
    ):
        user_indices = [index for index in indices if rows[index]["request"]["event"]["kind"] == "user"]
        count = sum(predicate(index) for index in user_indices)
        answer[key] = dict(
            count=count, support=len(user_indices), rate=count / len(user_indices) if user_indices else None
        )
    answer["predicted_decisions"] = dict(Counter(prediction["decision"] for prediction in predictions))
    if raw_tokens is not None:
        expected_tokens = encode_targets(rows)["token"]
        guessed = np.asarray(raw_tokens)
        motion_mask = np.zeros_like(expected_tokens, dtype=bool)
        motion_mask[action_indices] = True
        active = (expected_tokens != -100) & motion_mask
        eos = (expected_tokens == EOS) & active
        answer["raw_eos_at_expected_end"] = ratio(int((guessed[eos] == EOS).sum()), int(eos.sum()))
        answer["raw_token_by_position"] = {
            str(position): ratio(
                int((guessed[:, position][mask] == expected_tokens[:, position][mask]).sum()), int(mask.sum())
            )
            for position in range(expected_tokens.shape[1])
            if (mask := active[:, position]).any()
        }
    return answer


def majority_user_baseline(rows):
    """Choose the most frequent complete semantic response using training users only."""
    counts, examples = Counter(), {}
    for row in rows:
        if row["request"]["event"]["kind"] != "user":
            continue
        target = row["target"]
        key = (target["decision"], canonical_plan(target["goals"]))
        counts[key] += 1
        examples[key] = dict(decision=target["decision"], goals=target["goals"])
    if not counts:
        raise ValueError("The majority-user baseline requires training user examples.")
    winner = counts.most_common(1)[0][0]
    return dict(prediction=examples[winner], training_support=counts[winner], training_user_rows=sum(counts.values()))
