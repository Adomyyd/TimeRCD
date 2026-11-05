import pandas as pd

from experiments.stage1.analyze import bootstrap_mean


META = ["case", "series", "source", "has_P"]


def paired_statistics(frame, dimensions, values, selectors):
    keys = META + ["score_kind"] + dimensions
    if frame.duplicated(keys + ["selector"]).any():
        raise ValueError("Duplicate selector observations")
    rows = []
    for identity, group in frame.groupby(keys, sort=True, dropna=False):
        if set(group.selector) != set(selectors):
            raise ValueError("Incomplete selectors")
        indexed = group.set_index("selector")
        row = dict(zip(keys, identity))
        for value in values:
            observations = indexed[value]
            if observations.isna().any() and not observations.isna().all():
                raise ValueError("Selector-dependent missingness")
            label = observations.loc["label"]
            random = observations.drop("label").mean()
            row[f"label_{value}"] = label
            row[f"random_{value}"] = random
            row[f"label_minus_random_{value}"] = label - random
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(frame, dimensions, values, config):
    rows = []
    for subset, chosen in (("all", frame), ("with_P", frame[frame.has_P]),
                            ("without_P", frame[~frame.has_P])):
        for identity, group in chosen.groupby(["score_kind"] + dimensions, sort=True, dropna=False):
            if group.case.duplicated().any():
                raise ValueError("Bootstrap units must be unique sequences")
            for value in values:
                present = group[value].dropna().to_numpy(dtype=float)
                result = bootstrap_mean(present, config["bootstrap_seed"], config["bootstrap_repetitions"]) if len(present) else {
                    "mean": None, "ci95_low": None, "ci95_high": None}
                rows.append({"subset": subset, **dict(zip(["score_kind"] + dimensions, identity)),
                             "statistic": value, "cases": len(group), "present_cases": len(present), **result})
    return pd.DataFrame(rows)


def source_sensitivity(frame, dimensions, values):
    rows = []
    for subset, chosen in (("all", frame), ("with_P", frame[frame.has_P])):
        for source in sorted(chosen.source.unique()):
            for mode, group in (("source", chosen[chosen.source == source]),
                                ("leave_one_source_out", chosen[chosen.source != source])):
                for identity, block in group.groupby(["score_kind"] + dimensions, sort=True, dropna=False):
                    for value in values:
                        present = block[value].dropna()
                        rows.append({"subset": subset, "mode": mode, "source": source,
                                     **dict(zip(["score_kind"] + dimensions, identity)), "statistic": value,
                                     "cases": len(block), "present_cases": len(present),
                                     "mean": float(present.mean()) if len(present) else None})
    return pd.DataFrame(rows)
