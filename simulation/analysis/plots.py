import traceback

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import statsmodels.stats.api as sms
from plotly.subplots import make_subplots

from utils.charts import prepare_table


# find first round where either we have Nan or 0, for each column except x and round
def compute_survival_months_stats(df):
    acc = []
    collapsed = []
    for col in df.columns:
        if col not in ["x", "round"]:
            # Check for the first occurrence of 0 or NaN
            first_invalid_index = df[df[col].isna() | (df[col] < 5)].index.min()

            if pd.notna(first_invalid_index):
                # Print the round number of the first occurrence
                survival_months = df.loc[first_invalid_index, "round"] + 1
                collapsed.append(True)
            else:
                survival_months = 12
                collapsed.append(False)
            acc.append(survival_months)

    mean = np.mean(acc)
    std = np.std(acc)
    ci = sms.DescrStatsW(acc).tconfint_mean(0.05) if len(acc) > 1 else [0, 0]

    # compute percentage of collapse until N
    t = np.array(list(zip(acc, collapsed)))
    percentage_collapse = [
        pd.DataFrame(
            {
                "month": [i],
                "percentage_collapsed": [((t[:, 0] <= i) & (t[:, 1] == 1)).mean()],
            },
            index=[i],
        )
        for i in range(1, 13)
    ]

    # max survival months
    max_survival_months = np.max(acc)

    return mean, ci[1] - mean, pd.concat(percentage_collapse), std, max_survival_months


import math


def get_figures_group(
    summary_group_df,
    resource_in_pool,
    id_to_display_name,
    selected_groups=None,
    only_beginning=False,
):
    fig_num_resource = go.Figure()
    fig_kaplan_meier = go.Figure()

    fig_percentage_collapse = go.Figure()

    if selected_groups is None:
        selected_groups = summary_group_df["id"].values

    average_collapsed_time_stats = []

    for group_i, group in enumerate(selected_groups):
        try:
            ALPHA = 0.32
            CONFIDENCE = 0.68
            data = resource_in_pool[group]

            d = summary_group_df.loc[group]
            colors = (d["colour_0"], d["colour_1"])

            m, c, perc_collapsed, std, _ = compute_survival_months_stats(data)
            s = {
                "name": (
                    group
                    if id_to_display_name[group] == ""
                    else id_to_display_name[group]
                ),
                "survival_months_mean": m,
                "survival_months_std": std,
                "survival_months_ci": c,
                "color_0": colors[0],
                "color_1": colors[1],
            }

            s["percentage_collapsed_1"] = (
                perc_collapsed["percentage_collapsed"].iloc[1]
                if len(perc_collapsed) > 1
                else 0
            )
            s["percentage_collapsed_12"] = (
                perc_collapsed["percentage_collapsed"].iloc[12]
                if len(perc_collapsed) > 12
                else 0
            )

            fig_percentage_collapse.add_trace(
                go.Scatter(
                    x=perc_collapsed["month"],
                    y=perc_collapsed["percentage_collapsed"],
                    name=(
                        group
                        if id_to_display_name[group] == ""
                        else id_to_display_name[group]
                    ),
                    mode="lines",
                    marker=dict(color=colors[0]),
                    customdata=np.repeat(group, len(perc_collapsed)),
                    hovertemplate="Month: %{x}<br>Percentage collapsed: %{y}",
                    hoverinfo="all",
                )
            )

            data = data.fillna(0)

            if only_beginning:
                # Extend the data to 12 months, starting from the first month which is not present.
                start = data["round"].max()

                required_entries = (12 - (start + 1)) * 2 + 2
                if required_entries > 2:
                    # Generating the repeated sequence of months
                    extended_rounds = np.repeat(np.arange(start + 1, 13), 2)
                    # Ensure that we do not exceed the required number of entries in case the last month needs only one entry
                    extended_rounds = extended_rounds[:required_entries]

                    # Creating the DataFrame for the missing entries
                    missing_months_df = pd.DataFrame(
                        {
                            "round": extended_rounds,
                            "x": extended_rounds,
                            **{
                                col: np.nan
                                for col in data.columns
                                if col != "x" and col != "round"
                            },
                        }
                    )

                    # Append the missing months to the original DataFrame
                    data = pd.concat([data, missing_months_df], ignore_index=True)
                    data = data.fillna(0)

            ci = data.drop(columns=["x", "round"]).apply(
                lambda x: sms.DescrStatsW(x).tconfint_mean(ALPHA), axis=1
            )
            data["resource_ci_lower"] = ci.apply(lambda x: x[0])
            data["resource_ci_upper"] = ci.apply(lambda x: x[1])
            data["resource_mean"] = data.drop(columns=["x", "round"]).apply(
                np.mean, axis=1
            )

            sub = [0]
            sub.extend(range(1, len(data["x"]), 2))
            fig_num_resource.add_trace(
                go.Scatter(
                    x=(
                        data["x"]
                        if not only_beginning
                        else data["x"].iloc[sub].apply(lambda x: math.ceil(x))
                    ),
                    y=(
                        data["resource_mean"]
                        if not only_beginning
                        else data["resource_mean"].iloc[sub]
                    ),
                    name=(
                        group
                        if id_to_display_name[group] == ""
                        else id_to_display_name[group]
                    ),
                    mode="lines",
                    marker=dict(color=colors[0]),
                    customdata=(
                        np.repeat(group, len(data))
                        if not only_beginning
                        else np.repeat(group, len(data.iloc[sub]))
                    ),
                    hovertemplate="Time: %{x}<br>Mean: %{y}",
                    hoverinfo="all",
                )
            )
            fig_num_resource.add_traces(
                [
                    go.Scatter(
                        x=(
                            data["x"]
                            if not only_beginning
                            else data["x"].iloc[sub].apply(lambda x: math.ceil(x))
                        ),
                        y=(
                            data["resource_ci_upper"]
                            if not only_beginning
                            else data["resource_ci_upper"].iloc[sub]
                        ),
                        mode="lines",
                        line_color="rgba(0,0,0,0)",
                        showlegend=False,
                        customdata=(
                            np.repeat(group, len(data))
                            if not only_beginning
                            else np.repeat(group, len(data.iloc[sub]))
                        ),
                    ),
                    go.Scatter(
                        x=(
                            data["x"]
                            if not only_beginning
                            else data["x"].iloc[sub].apply(lambda x: math.ceil(x))
                        ),
                        y=(
                            data["resource_ci_lower"]
                            if not only_beginning
                            else data["resource_ci_lower"].iloc[sub]
                        ),
                        mode="lines",
                        line_color="rgba(0,0,0,0)",
                        showlegend=False,
                        fill="tonexty",
                        fillcolor=colors[1],
                        customdata=(
                            np.repeat(group, len(data))
                            if not only_beginning
                            else np.repeat(group, len(data.iloc[sub]))
                        ),
                    ),
                ]
            )

            # Kaplan-Meier
            from lifelines import KaplanMeierFitter

            durations = []
            events = []
            for index, run in data.items():
                if index == "round" or index == "x":
                    continue
                event_time = None
                for i, r in enumerate(run):
                    if r == 0.0 or np.isnan(r):
                        event_time = data.loc[i, "round"] + 1
                        break
                events.append(0 if event_time is None else 1)
                if event_time is None:
                    event_time = data.loc[i, "round"] + 1
                durations.append(event_time)

            kmf = KaplanMeierFitter(alpha=ALPHA)

            x = data["round"].unique()
            x = np.append(x, [x[-1] + 1])
            kmf.fit(durations, event_observed=events, timeline=x)

            # Getting the survival function data
            kmf_survival = kmf.survival_function_
            kmf_confidence = kmf.confidence_interval_survival_function_

            # Create a Plotly figure
            s["KM_estimate_1"] = (
                kmf_survival["KM_estimate"].iloc[1] if len(kmf_survival) > 1 else 0
            )
            s["KM_estimate_12"] = (
                kmf_survival["KM_estimate"].iloc[12] if len(kmf_survival) > 12 else 0
            )
            # Add the Kaplan-Meier survival curve
            fig_kaplan_meier.add_trace(
                go.Scatter(
                    x=kmf_survival.index,
                    y=kmf_survival["KM_estimate"],
                    mode="lines",
                    name=(
                        group
                        if id_to_display_name[group] == ""
                        else id_to_display_name[group]
                    ),
                    line=dict(color=colors[0]),
                )
            )

            # Add confidence interval area
            fig_kaplan_meier.add_trace(
                go.Scatter(
                    x=kmf_confidence.index,
                    y=kmf_confidence[f"KM_estimate_lower_{CONFIDENCE}"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                )
            )

            fig_kaplan_meier.add_trace(
                go.Scatter(
                    x=kmf_confidence.index,
                    y=kmf_confidence[f"KM_estimate_upper_{CONFIDENCE}"],
                    mode="lines",
                    fill="tonexty",  # Fill area between the confidence interval lines
                    fillcolor=colors[1],
                    line=dict(width=0),
                    showlegend=False,
                )
            )

            average_collapsed_time_stats.append(s)
        except Exception as e:
            print(f"Error processing group {group}:")
            print(e)
            traceback.print_exc()

    leggend_info = {
        "fish": {
            "fig_num_resource_title": "Fish in lake over time",
            "fig_num_resource_yaxis_title_not_beg": "#tons of fish",
            "fig_num_resource_yaxis_title_beg": "#tons fish after fishing",
        },
        "sheep": {
            "fig_num_resource_title": "Available grass over time",
            "fig_num_resource_yaxis_title_not_beg": "#ha of grass",
            "fig_num_resource_yaxis_title_beg": "#ha of grass after grazing",
        },
        "pollution": {
            "fig_num_resource_title": f"% unpolluted water over time",
            "fig_num_resource_yaxis_title_not_beg": f"% unpolluted water",
            "fig_num_resource_yaxis_title_beg": f"% unpoll. water after production",
        },
    }
    if "fish" in selected_groups[0]:
        leggend_info = leggend_info["fish"]
    elif "sheep" in selected_groups[0]:
        leggend_info = leggend_info["sheep"]
    elif "pollution" in selected_groups[0]:
        leggend_info = leggend_info["pollution"]
    else:
        raise ValueError("Group not found")

    fig_num_resource.update_layout(
        title=leggend_info["fig_num_resource_title"],
        xaxis_title="Month",
        yaxis_title=(
            leggend_info["fig_num_resource_yaxis_title_not_beg"]
            if not only_beginning
            else leggend_info["fig_num_resource_yaxis_title_beg"]
        ),
        legend=dict(
            orientation="h",  # Horizontal orientation
            yanchor="bottom",  # Anchor legend at the bottom
            y=-1,  # Position legend below the plot area
            xanchor="left",  # Anchor legend horizontally in the center
            x=0,  # Center the legend with respect to the plot's x-axis
        ),
    )
    if not only_beginning:
        fig_num_resource.update_xaxes(range=[0, 11.8], dtick=1)
    else:
        fig_num_resource.update_xaxes(range=[0, 12.8], dtick=1)
    y_range = [0, 101]

    # Customizing y-axis ticks and gridlines
    fig_num_resource.update_yaxes(
        range=y_range,
        tickvals=[
            i for i in range(y_range[0], y_range[1] + 1, 10)
        ],  # Gridlines every 10 units
        ticktext=[
            str(i) if i % 20 == 0 else "" for i in range(y_range[0], y_range[1] + 1, 10)
        ],  # Labels every 20 units
    )
    fig_num_resource.update_layout(showlegend=False)
    # Customize layout
    fig_kaplan_meier.update_layout(
        title="Kaplan-Meier Survival Estimate",
        xaxis_title="Month",
        yaxis_title="Shared Resouce Survival Probability",
        legend=dict(
            orientation="h",  # Horizontal orientation
            yanchor="bottom",  # Anchor legend at the bottom
            y=-0.4,  # Position legend below the plot area
            xanchor="left",  # Anchor legend horizontally in the center
            x=0,  # Center the legend with respect to the plot's x-axis
        ),
    )

    fig_kaplan_meier.update_xaxes(range=[0, 12], dtick=1)  # Adjust x-axis range
    fig_kaplan_meier.update_yaxes(range=[0, 1], dtick=0.2)

    fig_kaplan_meier.update_layout(showlegend=False)

    fig_percentage_collapse.update_layout(
        title="Percentage of collapse over time",
        xaxis_title="Month",
        yaxis_title="Percentage of collapse",
        legend=dict(
            orientation="h",  # Horizontal orientation
            yanchor="bottom",  # Anchor legend at the bottom
            y=-1,  # Position legend below the plot area
            xanchor="left",  # Anchor legend horizontally in the center
            x=0,  # Center the legend with respect to the plot's x-axis
        ),
    )
    fig_percentage_collapse.update_xaxes(range=[1, 11.8], dtick=1)
    fig_percentage_collapse.update_yaxes(range=[0, 1], dtick=0.1)
    fig_percentage_collapse.update_layout(showlegend=False)

    # Average survival time
    average_collapsed_time_stats = pd.DataFrame(average_collapsed_time_stats)
    fig_average_collapsed_time = go.Figure()
    fig_average_collapsed_time.add_trace(
        go.Bar(
            x=average_collapsed_time_stats["name"],
            y=average_collapsed_time_stats["survival_months_mean"],
            error_y=dict(
                type="data",
                array=average_collapsed_time_stats["survival_months_ci"],
                visible=True,
            ),
            marker=dict(color=average_collapsed_time_stats["color_0"]),
        )
    )

    fig_average_collapsed_time.update_layout(
        title="Average survival time",
        xaxis_title="Group",
        yaxis_title="Months",
        legend=dict(
            orientation="h",  # Horizontal orientation
            yanchor="bottom",  # Anchor legend at the bottom
            y=-1,  # Position legend below the plot area
            xanchor="left",  # Anchor legend horizontally in the center
            x=0,  # Center the legend with respect to the plot's x-axis
        ),
    )
    fig_average_collapsed_time.update_layout(showlegend=False)

    latex_table = average_collapsed_time_stats[
        ["name", "KM_estimate_1", "KM_estimate_12", "survival_months_mean"]
    ].to_latex(
        columns=["name", "KM_estimate_1", "KM_estimate_12", "survival_months_mean"],
        header=[
            "LLM",
            "$P_\\text{{surv}}@1\\uparrow$ ",
            "$P_\\text{{surv}}@12\\uparrow$",
            "Avg. survival months $\\uparrow$",
        ],
        column_format="l|m{0.15\\textwidth}m{0.15\\textwidth} m{0.25\\textwidth}",
        index=False,
        float_format="%.2f",
        caption="\\red{todo}",
        label="table:todo",
    )

    return (
        fig_num_resource,
        fig_kaplan_meier,
        fig_percentage_collapse,
        fig_average_collapsed_time,
        latex_table,
        average_collapsed_time_stats,
    )


from .utils import generate_colors


def get_figures_single_run(
    preprocessing_data,
    group_name,
    generate_colors=generate_colors,
    max_y_axis=100,
    run_names=None,
):
    summary_df = preprocessing_data["summary_df"]
    run_data = preprocessing_data["run_data"]
    resource_in_pool = preprocessing_data["resource_in_pool"][group_name]
    fig_num_resource = go.Figure()

    fig_resource_by_persona = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[
            *[f"Persona {i}" for i in range(5)],
            "Fish limit - conversation",
        ],
    )

    fig_single_resource_by_persona = {f"persona_{p}": go.Figure() for p in range(5)}
    fig_single_resource_by_persona["resource_limit"] = go.Figure()

    persona_to_position = [
        {"row": 1, "col": 1},
        {"row": 1, "col": 2},
        {"row": 1, "col": 3},
        {"row": 2, "col": 1},
        {"row": 2, "col": 2},
        {"row": 2, "col": 3},
    ]

    if run_names is None:
        run_names = summary_df[summary_df["group"] == group_name]["name"]

    colors = generate_colors(len(run_names))
    for i, run in enumerate(run_names):
        df = run_data[run]

        fig_num_resource.add_trace(
            go.Scatter(
                x=resource_in_pool["x"],
                y=resource_in_pool[run],
                name=run,
                mode="lines",
                marker=dict(color=colors[i][0]),
                customdata=[(run, int(np.floor(i))) for i in resource_in_pool["x"]],
            )
        )
        for p in range(5):
            persona_id = f"persona_{p}"
            # Filter the DataFrame for the current persona and action 'fish'
            d = df[
                df.apply(
                    lambda x: x["agent_id"] == persona_id
                    and x["action"] == "harvesting",
                    axis=1,
                )
            ]

            # Add a trace for each persona
            fig_resource_by_persona.add_trace(
                go.Scatter(
                    x=d["round"],
                    y=d["resource_collected"],
                    name=run,
                    mode=("lines" if len(d["resource_collected"]) > 1 else "markers"),
                    marker=dict(color=colors[i][0]),
                    customdata=[(run, f"persona_{p}")] * len(d),
                    showlegend=(p == 0),
                ),
                **persona_to_position[p],
            )

            fig_single_resource_by_persona[persona_id].add_trace(
                go.Scatter(
                    x=d["round"],
                    y=d["resource_collected"],
                    name=run,
                    mode=("lines" if len(d["resource_collected"]) > 1 else "markers"),
                    marker=dict(color=colors[i][0]),
                    customdata=[(run, f"persona_{p}")] * len(d),
                    showlegend=True,
                )
            )

            fig_resource_by_persona.update_yaxes(
                range=[0, max_y_axis],
                dtick=20 if max_y_axis > 50 else 10,
                **persona_to_position[p],
            )

            fig_single_resource_by_persona[persona_id].update_yaxes(
                range=[0, max_y_axis], dtick=20 if max_y_axis > 50 else 10
            )

        df_filtered = df[df["action"] == "utterance"]
        df_grouped = df_filtered.groupby("round").first()
        df_grouped.fillna(-1, inplace=True)

        # fig_resource_by_persona.add_trace(
        fig_resource_by_persona.add_trace(
            go.Scatter(
                x=df_grouped.index,
                y=df_grouped["resource_limit"],
                name=run,
                mode=("lines" if len(df_grouped["resource_limit"]) > 1 else "markers"),
                marker=dict(color=colors[i][0]),
                customdata=[(run, "resource_limit")] * len(d),
                showlegend=False,
            ),
            **persona_to_position[-1],
        )

        fig_single_resource_by_persona["resource_limit"].add_trace(
            go.Scatter(
                x=df_grouped.index,
                y=df_grouped["resource_limit"],
                name=run,
                mode=("lines" if len(df_grouped["resource_limit"]) > 1 else "markers"),
                marker=dict(color=colors[i][0]),
                customdata=[(run, "resource_limit")] * len(d),
                showlegend=True,
            )
        )
        fig_single_resource_by_persona["resource_limit"].update_yaxes(
            range=[-5, 20], dtick=5
        )

        fig_resource_by_persona.update_yaxes(
            range=[-5, 20], dtick=5, **persona_to_position[-1]
        )

    fig_num_resource.update_layout(
        title="Fish in lake over time",
        xaxis_title="Month",
        yaxis_title="#tons of fish",
        legend=dict(
            orientation="h",  # Horizontal orientation
            yanchor="bottom",  # Anchor legend at the bottom
            y=-0.4,  # Position legend below the plot area
            xanchor="left",  # Anchor legend horizontally in the center
            x=0,  # Center the legend with respect to the plot's x-axis
        ),
        # Ensure there's enough space at the bottom of the plot for the legend
        margin=dict(b=100),
    )
    fig_num_resource.update_xaxes(range=[0, 15], dtick=1)
    fig_num_resource.update_yaxes(range=[-1, 102], dtick=20)

    fig_resource_by_persona.update_layout(
        title=f"Fish caught by each persona",
        yaxis_title="#tons fish",
        legend=dict(
            orientation="h",  # Horizontal orientation
            yanchor="bottom",  # Anchor legend at the bottom
            y=-0.4,  # Position legend below the plot area
            xanchor="left",  # Anchor legend horizontally in the center
            x=0,  # Center the legend with respect to the plot's x-axis
        ),
        # Ensure there's enough space at the bottom of the plot for the legend
        margin=dict(b=100),
    )

    for p in range(5):
        fig_single_resource_by_persona[f"persona_{p}"].update_layout(
            title=f"Fish caught by persona {p}",
            xaxis_title="Month",
            yaxis_title="#tons fish",
            legend=dict(
                orientation="h",  # Horizontal orientation
                yanchor="bottom",  # Anchor legend at the bottom
                y=-0.4,  # Position legend below the plot area
                xanchor="left",  # Anchor legend horizontally in the center
                x=0,  # Center the legend with respect to the plot's x-axis
            ),
            # Ensure there's enough space at the bottom of the plot for the legend
            margin=dict(b=100),
        )
        fig_single_resource_by_persona[f"persona_{p}"].update_xaxes(
            range=[0, 10], dtick=1
        )

    fig_resource_by_persona.update_xaxes(range=[0, 10], dtick=1)

    return fig_num_resource, fig_resource_by_persona, fig_single_resource_by_persona


def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    array = array.flatten()  # all values are treated equally, arrays must be 1d
    # remove nans
    array = array[~np.isnan(array)]
    if np.amin(array) < 0:
        array -= np.amin(array)  # values cannot be negative
    # array += 0.0000001 #values cannot be 0
    array = np.sort(array)  # values must be sorted
    index = np.arange(1, array.shape[0] + 1)  # index per array element
    n = array.shape[0]  # number of array elements
    g = (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))  # Gini coefficient
    return g


def get_gini_graph(
    preprocessing_data,
    group_name,
    generate_colors=generate_colors,
    max_y_axis=100,
    run_names=None,
):
    summary_df = preprocessing_data["summary_df"]
    run_data = preprocessing_data["run_data"]
    resource_in_pool = preprocessing_data["resource_in_pool"][group_name]

    if run_names is None:
        run_names = summary_df[summary_df["group"] == group_name]["name"]

    fig = go.Figure()
    colors = generate_colors(len(run_names))
    acc = []
    for i, run in enumerate(run_names):
        df = run_data[run]

        df = df[df["agent_id"] != "framework"]
        g = df[df["action"] == "harvesting"].groupby("round")
        g_gini = g["resource_collected"].apply(lambda x: gini(x.values)).round(2)

        fig.add_trace(
            go.Scatter(
                x=g_gini.index + 1,
                y=g_gini,
                name=run,
                mode="lines",
                marker=dict(color=colors[i][0]),
            )
        )
        acc.append(g_gini)
    acc = pd.concat(acc, axis=1)

    fig.update_layout(
        title="Gini coefficient of fish caught over time",
        xaxis_title="Round",
        yaxis_title="Gini coefficient",
        showlegend=True,
    )
    fig.update_yaxes(range=[0, 1], dtick=0.2)
    fig.update_xaxes(range=[1, 15], dtick=1)
    return fig, acc


def get_payoffs(
    preprocessing_data,
    group_name,
    run_names=None,
):
    summary_df = preprocessing_data["summary_df"]
    run_data = preprocessing_data["run_data"]
    resource_in_pool = preprocessing_data["resource_in_pool"][group_name]

    if run_names is None:
        run_names = summary_df[summary_df["group"] == group_name]["name"]

    acc = []
    for i, run in enumerate(run_names):
        df = run_data[run]
        df = df[df["agent_id"] != "framework"]
        sum_payoff_agent = (
            df[df["round"] >= 0].groupby("agent_id")["resource_collected"].sum()
        )
        acc.append(sum_payoff_agent)

    acc = pd.concat(acc, axis=1)
    return acc.mean().mean(), acc.mean().std(), acc


def get_payoffs_from_shock(
    preprocessing_data,
    group_name,
    run_names=None,
    shock_round=3,
):
    summary_df = preprocessing_data["summary_df"]
    run_data = preprocessing_data["run_data"]
    resource_in_pool = preprocessing_data["resource_in_pool"][group_name]

    if run_names is None:
        run_names = summary_df[summary_df["group"] == group_name]["name"]

    acc = []
    for i, run in enumerate(run_names):
        df = run_data[run]
        df = df[df["agent_id"] != "framework"]
        sum_payoff_agent = (
            df[df["round"] >= shock_round]
            .groupby("agent_id")["resource_collected"]
            .sum()
        )
        acc.append(sum_payoff_agent)

    acc = pd.concat(acc, axis=1)
    return acc.mean().mean(), acc.mean().std(), acc


def get_gini_groups(
    preprocessing_data, summary_group_df, resource_in_pool, id_to_display_name
):
    fig = go.Figure()

    selected_groups = summary_group_df["id"].values
    summary_df = preprocessing_data["summary_df"]
    average_collapsed_time_stats = []

    run_data = preprocessing_data["run_data"]

    for group_i, group in enumerate(selected_groups):

        data = resource_in_pool[group]

        d = summary_group_df.loc[group]
        colors = (d["colour_0"], d["colour_1"])

        run_names = summary_df[summary_df["group"] == group]["name"]

        acc = []
        for i, run in enumerate(run_names):
            df = run_data[run]

            df = df[df["agent_id"] != "framework"]
            g = df[df["action"] == "harvesting"].groupby("round")
            g_gini = g["resource_collected"].apply(lambda x: gini(x.values)).round(2)

            acc.append(g_gini)
        acc = pd.concat(acc, axis=1)

        fig.add_trace(
            go.Scatter(
                x=acc.index + 1,
                y=acc.mean(axis=1),
                name=(
                    group
                    if id_to_display_name[group] == ""
                    else id_to_display_name[group]
                ),
                mode="lines",
                marker=dict(color=colors[0]),
            )
        )

        fig.update_layout(
            title="Gini coefficient of fish caught over time",
            xaxis_title="Round",
            yaxis_title="Gini coefficient",
            showlegend=True,
        )
        fig.update_yaxes(range=[0, 1], dtick=0.2)
        fig.update_xaxes(range=[1, 12], dtick=1)
    return fig
