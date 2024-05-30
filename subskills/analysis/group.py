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

from .app import app, global_store
from .plots import get_bar_subskills
from .utils import NAMES, create_table, generate_colors

# Run selection
run_selection = html.Div(
    id="run_selection",
    children=[
        html.H4("Group selection"),
        dash_table.DataTable(
            id="datatable-groups",
            filter_action="native",
            sort_action="native",
            sort_mode="single",
            row_selectable="multi",
            selected_columns=[],
            selected_rows=[],
            page_action="native",
            page_current=0,
            page_size=8,
            markdown_options={"html": True},
            style_table={
                "width": "100%",
                "overflowX": "auto",
                "height": "600px",
            },  # Fixed width
            # Optional: Style the cells to ensure they don't expand too much
            style_cell={
                "width": "100px",
                "maxWidth": "100px",
                "overflow": "hidden",
                "textOverflow": "ellipsis",
                "whiteSpace": "normal",
            },
            editable=True,
        ),
    ],
)

selected_group_info = html.Div(
    [
        dcc.Graph(id="scores-histogram"),
    ],
    style={"height": "350px"},
)

group = html.Div(
    children=[
        dmc.Stack(
            children=[
                selected_group_info,
                run_selection,
                dmc.Button("Export", n_clicks=0, id="export-group-data"),
            ]
        )
    ],
    style={
        "margin-left": "1%",
        "margin-right": "1%",
        "display": "inline-block",
        "vertical-align": "top",
        "width": "100%",
    },
)


@app.callback(
    Output("url", "pathname"),
    [Input("datatable-groups", "active_cell")],
    [State("url", "pathname")],
)
def display_click_data(active_cell, current_path):
    if active_cell:
        print(active_cell)
        if active_cell["column_id"] == "display_name":
            return dash.no_update
        else:
            return f"{current_path}/details/{active_cell['row_id']}"
    return current_path


@app.callback(
    [
        Output("datatable-groups", "columns"),
        Output("datatable-groups", "data"),
        Output("datatable-groups", "style_data_conditional"),
    ],
    Input("runs-group", "data"),
)
def run_selection_display(subset_name):
    preprocessing_data = global_store(subset_name)
    summary_group_df = preprocessing_data["summary"]
    summary_group_df["display_name"] = summary_group_df["config.llm.path"].apply(
        get_pretty_name_llm
    )

    conditions = [
        {
            "if": {"column_id": "group_name"},
            "backgroundColor": "var(--bs-white)",  # Set a default or fallback color
            "width": "20px",
            "borderRadius": "50%",  # Creates the circle shape,
            "color": "transparent",
        }
        for color in summary_group_df["colour_0"]
    ] + [
        {
            "if": {
                "filter_query": f"{{id}} = '{d['id']}'",
                "column_id": "group_name",
            },
            "color": "black",
            "backgroundColor": d["colour_0"],
        }
        for _, d in summary_group_df.iterrows()
    ]

    columns = [
        {
            "name": "group_name",
            "id": "group_name",
            "presentation": "markdown",
        },
        {
            "name": "display_name",
            "id": "display_name",
            "presentation": "markdown",
        },
        *[
            {"name": i, "id": i, "selectable": True}
            for i in summary_group_df.columns
            if i != "id"
            and i != "colour_1"
            and i != "colour_0"
            and i != "group_name"
            and i not in NAMES
            and i != "agg_score"
            and i != "display_name"
        ],
        *[{"name": i, "id": i, "selectable": True} for i in NAMES],
        {"name": "agg_score", "id": "agg_score", "selectable": True},
    ]
    data = summary_group_df.to_dict("records")
    style_data_conditional = conditions
    return columns, data, style_data_conditional


@app.callback(
    Output("datatable-groups", "selected_rows"),
    Input("datatable-groups", "derived_virtual_indices"),
)
def update_runs_display(row_ids):
    return row_ids


import plotly.io as pio

pio.templates.default = "plotly_white"

import copy
import os


def prepare_fig_for_export(fig):

    fig.update_layout(
        title="",
        font_family="Times New Roman",
        font_size=12,
        title_font_size=12,
        margin_l=0,
        margin_t=0,
        margin_b=5,
        margin_r=0,
        width=800,
        height=300,
    )
    return fig


@app.callback(
    Output("scores-histogram", "figure"),
    [
        Input("runs-group", "data"),
        Input("datatable-groups", "derived_virtual_selected_row_ids"),
        Input("datatable-groups", "data"),
        Input("export-group-data", "n_clicks"),
    ],
)
def update_statistics(subset_name, selected_groups, table_data, export_clicks):
    export = False
    ctx = dash.callback_context
    if not ctx.triggered:
        # In case the callback wasn't triggered by the inputs listed (shouldn't happen in this setup)
        return dash.no_update
    else:
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if export_clicks > 0 and trigger_id == "export-group-data":
        export = True
    id_to_display_name = {d["id"]: d["display_name"] for d in table_data}

    preprocessing_data = global_store(subset_name)
    summary_group_df = preprocessing_data["summary"]

    fig = get_bar_subskills(summary_group_df, id_to_display_name, selected_groups)

    if export:
        base_path = os.path.dirname(os.path.realpath(__file__))
        current_time = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_path = os.path.join(base_path, "..", "exports", current_time)
        os.makedirs(base_path, exist_ok=True)

        fig_export = copy.deepcopy(fig)
        fig_export.update_layout(title="Subskill evaluation", showlegend=True)
        fig_export = prepare_fig_for_export(fig_export)
        fig_export.write_image(os.path.join(base_path, "subskill_evaluation.pdf"))

    return fig
