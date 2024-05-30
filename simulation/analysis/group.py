import copy
import os
import traceback

import dash
import dash_mantine_components as dmc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import statsmodels.stats.api as sms
from dash import Input, Output, State, dash_table, dcc, html
from plotly import io as pio
from plotly.subplots import make_subplots

from utils.charts import get_LLM_order, get_pretty_name_llm

from .app import app, global_store
from .utils import create_table, generate_colors, prepare_fig_for_export

pio.kaleido.scope.mathjax = None

# Run selection
run_selection = html.Div(
    id="run_selection",
    children=[
        html.H4("Group selection"),
        dash_table.DataTable(
            id="datatable-groups",
            filter_action="native",
            sort_action="native",
            sort_mode="multi",
            row_selectable="multi",
            selected_columns=[],
            selected_rows=[],
            page_action="native",
            page_current=0,
            page_size=20,
            markdown_options={"html": True},
            style_table={
                "width": "100%",
                "overflowX": "auto",
                "height": "900px",
            },  # Fixed width
            # Optional: Style the cells to ensure they don't expand too much
            style_cell={
                # "width": "200px",
                "maxWidth": "200px",
                "overflow": "hidden",
                "textOverflow": "ellipsis",
                "whiteSpace": "normal",
            },
            editable=True,
        ),
    ],
)


@app.callback(
    [
        Output("datatable-groups", "columns"),
        Output("datatable-groups", "data"),
        Output("datatable-groups", "style_data_conditional"),
    ],
    Input("runs-group", "data"),
)
def run_selection_display(group_name):
    preprocessing_data = global_store(group_name)
    summary_group_df = preprocessing_data["summary_group_df"]
    summary_group_df["display_name"] = summary_group_df["llm.path"].apply(
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
        *[
            {"name": i, "id": i, "selectable": True}
            for i in summary_group_df.columns
            if i != "id" and i != "colour_1" and i != "colour_0" and i != "group_name"
        ],
    ]
    data = summary_group_df.to_dict("records")
    style_data_conditional = conditions
    return columns, data, style_data_conditional


tab_group_statistics = html.Div(
    dmc.TabsPanel(
        value="group-statistics",
        children=[
            dcc.Graph(id="resource-over-time-graph-group"),
            dcc.Graph(id="kaplan-meier-resource-survival-group"),
            dcc.Graph(id="percentage-collapse-graph-group"),
            dcc.Graph(id="average-collapsed-time-graph-group"),
        ],
    )
)

selected_group_info = dmc.Stack(
    [
        dmc.Button("Export", n_clicks=0, id="export-group-data"),
        dmc.Tabs(
            [
                dmc.TabsList(
                    [
                        dmc.Tab("Group statistics", value="group-statistics"),
                        # dmc.Tab("Runs Selection", value="runs-selection"),
                    ],
                    grow=True,
                ),
                tab_group_statistics,
            ],
            value="group-statistics",
        ),
    ],
)

group = html.Div(
    children=[dmc.SimpleGrid(cols=2, children=[run_selection, selected_group_info])],
    style={
        "margin-left": "1%",
        "margin-right": "1%",
        "display": "inline-block",
        "vertical-align": "top",
        "width": "100%",
    },
)


@app.callback(
    Output("datatable-groups", "selected_rows"),
    Input("datatable-groups", "derived_virtual_indices"),
)
def update_runs_display(row_ids):
    return row_ids


from .plots import get_figures_group


@app.callback(
    [
        Output("resource-over-time-graph-group", "figure"),
        Output("kaplan-meier-resource-survival-group", "figure"),
        Output("percentage-collapse-graph-group", "figure"),
        Output("average-collapsed-time-graph-group", "figure"),
    ],
    [
        Input("runs-group", "data"),
        Input("datatable-groups", "derived_virtual_selected_row_ids"),
        Input("datatable-groups", "data"),
        Input("export-group-data", "n_clicks"),
    ],
)
def update_statistics(group_name, selected_groups, table_data, export_clicks):
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
    preprocessing_data = global_store(group_name)
    summary_group_df = preprocessing_data["summary_group_df"]
    resource_in_pool = preprocessing_data["resource_in_pool"]

    (
        fig_num_resource,
        fig_kaplan_meier,
        fig_percentage_collapse,
        fig_average_collapsed_time,
        latex_table,
        _,
    ) = get_figures_group(
        summary_group_df, resource_in_pool, id_to_display_name, selected_groups
    )

    # Export fig, if requested and copy fig such that we don't modify the original

    fig_num_resource = prepare_fig_for_export(fig_num_resource)
    fig_kaplan_meier = prepare_fig_for_export(fig_kaplan_meier)
    fig_percentage_collapse = prepare_fig_for_export(fig_percentage_collapse)
    fig_average_collapsed_time = prepare_fig_for_export(fig_average_collapsed_time)

    if export:

        fig_num_resource_export = copy.deepcopy(fig_num_resource)
        fig_kaplan_meier_export = copy.deepcopy(fig_kaplan_meier)
        fig_percentage_collapse_export = copy.deepcopy(fig_percentage_collapse)
        fig_average_collapsed_time_export = copy.deepcopy(fig_average_collapsed_time)

        # Add the group name to the title
        fig_num_resource_export.update_layout(
            title_text=f"Fish in lake over time", showlegend=True
        )
        fig_kaplan_meier_export.update_layout(
            title_text=f"Kaplan-Meier Survival Estimate", showlegend=True
        )
        fig_percentage_collapse_export.update_layout(
            title_text=f"Percentage of collapse over time", showlegend=True
        )

        fig_average_collapsed_time_export.update_layout(
            title_text=f"Average survival time", showlegend=True
        )

        base_path = os.path.dirname(os.path.realpath(__file__))

        current_time = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_path = os.path.join(base_path, "..", "..", "..", "exports", current_time)
        os.makedirs(base_path, exist_ok=True)

        fig_num_resource_export = prepare_fig_for_export(fig_num_resource_export)
        fig_kaplan_meier_export = prepare_fig_for_export(fig_kaplan_meier_export)
        fig_percentage_collapse_export = prepare_fig_for_export(
            fig_percentage_collapse_export
        )
        fig_average_collapsed_time_export = prepare_fig_for_export(
            fig_average_collapsed_time_export
        )
        fig_num_resource_export.write_image(f"{base_path}/fish_over_time.pdf")
        fig_kaplan_meier_export.write_image(f"{base_path}/kaplan_meier.pdf")
        fig_percentage_collapse_export.write_image(
            f"{base_path}/percentage_collapse.pdf"
        )
        fig_average_collapsed_time_export.write_image(
            f"{base_path}/average_collapsed_time.pdf"
        )

        # export table

        with open(f"{base_path}/table.tex", "w") as f:
            f.write(latex_table)
    return (
        fig_num_resource,
        fig_kaplan_meier,
        fig_percentage_collapse,
        fig_average_collapsed_time,
    )


# Callback to handle click events on the graph
@app.callback(
    Output("url", "pathname"),
    [Input("resource-over-time-graph-group", "clickData")],
    [State("url", "pathname")],
)
def handle_graph_click(clickData, current_path):
    if clickData:
        # Extract clicked element information (e.g., curve number)
        d = clickData["points"][0]

        return f"{current_path}/{d['customdata'].split('/')[1]}/details"
        # Add more conditions for other groups
    return current_path  # Stay on the current page if no relevant click
