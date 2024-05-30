import traceback

import dash
import dash_mantine_components as dmc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import statsmodels.stats.api as sms
from dash import Input, Output, State, dash_table, dcc, html
from plotly.subplots import make_subplots

from utils.charts import get_LLM_order, get_pretty_name_llm

from .utils import NAMES, create_table, generate_colors


def get_bar_subskills(summary_group_df, id_to_display_name, selected_groups=None):

    if selected_groups is None:
        selected_groups = summary_group_df["id"].values

    fig = go.Figure()
    labels = {
        "multiple_math_consequence_after_using_same_amount": "A) Math basic reasoning",
        "multiple_sim_consequence_after_using_same_amount": "C) Sim basic reasoning",
        "multiple_math_shrinking_limit": "B) Math shrinking limit",
        "multiple_math_shrinking_limit_assumption": "B') Math shrinking limit assumption",
        "multiple_sim_shrinking_limit": "D) Sim shrinking limit",
        "multiple_sim_shrinking_limit_assumption": "D') Sim shrinking limit assumption",
        "multiple_sim_universalization_consume_grass": "F) Sim grass sustainable (Universalization)",
        "multiple_sim_consume_grass_standard_persona": "E) Sim grass sustainable",
    }

    # order selected group based on get_LLM_order which is an array of ordered display_name columns
    selected_groups = sorted(
        selected_groups,
        key=lambda x: get_LLM_order().index(id_to_display_name[x]),
    )
    for idx, score_type in enumerate(NAMES):
        scores = []
        names = []
        for group_i, group in enumerate(selected_groups):
            group_name = (
                group if id_to_display_name[group] == "" else id_to_display_name[group]
            )
            names.append(group_name)
            group_data = summary_group_df[summary_group_df["id"] == group]
            scores.append(float(group_data[score_type].iloc[0]))
        # Add the accumulated scores as a histogram trace
        fig.add_trace(go.Bar(x=names, y=scores, name=labels[score_type]))

    fig.update_layout(barmode="group")
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=12))
    fig.update_yaxes(title_text="Score", range=[0, 1.01], dtick=0.1, showgrid=True)
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="center", x=0.5)
    )
    return fig
