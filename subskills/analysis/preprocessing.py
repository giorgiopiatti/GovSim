import os

import numpy as np
import pandas as pd

from .utils import generate_colors


def columns_non_relevant(df):
    columns_with_variability = []
    for column in df.columns:
        if any(isinstance(x, list) for x in df[column]):
            try:
                # Convert each list to a tuple for comparison since tuples are hashable
                # If the list contains dictionaries, convert them to sorted tuples of tuples (key-value pairs)
                unique_tuples = set(
                    tuple(
                        tuple(sorted(d.items())) if isinstance(d, dict) else d
                        for d in x
                    )
                    for x in df[column]
                )
                if len(unique_tuples) == 1:
                    columns_with_variability.append(column)
            except TypeError as e:
                # Handle other potential TypeError if encountered
                # print(f"Error processing column {column}: {e}")
                columns_with_variability.append(column)
        else:
            if df[column].nunique() == 1:
                columns_with_variability.append(column)
    return columns_with_variability


import json

import pandas as pd

from .utils import NAMES


def prepare_run_data(base_path_run, run_name):
    acc = []
    for name in NAMES:
        f = f"{base_path_run}/{run_name}/{name}.json"
        if not os.path.exists(f):
            continue
        try:
            d = json.load(open(f))
        except json.JSONDecodeError:
            continue
        d = (
            pd.json_normalize(d)
            .rename(
                columns={
                    "instances": f"{name}_instances",
                    "score_mean": f"{name}_score_mean",
                    "score_std": f"{name}_score_std",
                    "score_ci_lower": f"{name}_score_ci_lower",
                    "score_ci_upper": f"{name}_score_ci_upper",
                }
            )
            .drop(
                columns=[
                    "name",
                ]
            )
        )
        d = d.round(2)
        d[name] = (
            d[f"{name}_score_mean"].astype(str)
            # + "±"
            # + d[f"{name}_score_std"].astype(str)
        )
        acc.append(d)
    if len(acc) == 0:
        return None

    concatenated_df = pd.concat(acc, axis=1)

    # Compute average score across NAMES
    concatenated_df["agg_score"] = (
        concatenated_df.filter(like="_score_mean").mean(axis=1).round(2)
    )
    final_df = concatenated_df.loc[:, ~concatenated_df.columns.duplicated()]
    return final_df


def get_data(subset_name):
    # subskills_check_v0.3/
    base_path_run = os.path.join(
        os.path.dirname(__file__), "..", "results", subset_name
    )
    runs_names = os.listdir(base_path_run)
    acc = []
    runs = {}
    for n in runs_names:
        d = prepare_run_data(base_path_run, n)
        if d is None:
            continue
        d["group_name"] = n
        d["id"] = n
        runs[n] = d

        acc.append(d)
    summary_df = pd.concat(acc)
    non_relevant = columns_non_relevant(summary_df)
    if len(summary_df) <= 1:
        non_relevant = []

    for c in summary_df.columns:
        if c.endswith("_std"):
            non_relevant.append(c)
        elif c.endswith("_mean"):
            non_relevant.append(c)
        elif c.endswith("_instances"):
            non_relevant.append(c)
        # elif c.endswith("_ci_lower"):
        #     non_relevant.append(c)
        # elif c.endswith("_ci_upper"):
        #     non_relevant.append(c)
    non_relevant = [c for c in non_relevant if c not in NAMES]
    non_relevant = list(set(non_relevant))
    summary_df = summary_df.drop(columns=non_relevant)
    colours = generate_colors(len(summary_df))
    np.random.seed(42)
    np.random.shuffle(colours)

    summary_df["colour_0"] = [c[0] for c in colours]
    summary_df["colour_1"] = [c[1] for c in colours]

    # sort by agg_score
    # summary_df = summary_df.sort_values(by="agg_score", ascending=False)
    # # take only the best across config.llm.path
    # summary_df = summary_df.drop_duplicates(subset=["config.llm.path"], keep="first")

    return {
        "summary": summary_df,
        "runs": runs,
    }
