import base64

import dash
import dash_mantine_components as dmc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import statsmodels.stats.api as sms
from dash import Input, Output, State, dcc, html
from plotly.subplots import make_subplots

from .app import app, global_store
from .utils import create_table, generate_colors


def get_tab(test_name):
    num_passing = 10
    num_failing = 10
    return dmc.TabsPanel(
        [
            dmc.Stack(
                [
                    dmc.Text(id=f"{test_name}-score-mean", mt=10),
                    dmc.SimpleGrid(
                        cols=2,
                        children=[
                            dmc.Stack(
                                children=[
                                    html.H5("Passing instances"),
                                    dcc.Graph(id=f"{test_name}-passing-graph"),
                                    dmc.Slider(
                                        id=f"{test_name}-passing-slider",
                                        value=0,
                                        min=0,
                                        step=1,
                                        mr=10,
                                    ),
                                    html.Div(id=f"{test_name}-passing-prompt"),
                                ]
                            ),
                            dmc.Stack(
                                children=[
                                    html.H5("Failing instances"),
                                    dcc.Graph(id=f"{test_name}-failing-graph"),
                                    dmc.Slider(
                                        id=f"{test_name}-failing-slider",
                                        marks=[
                                            {"value": i, "label": str(i)}
                                            for i in range(0, num_failing)
                                        ],
                                        value=0,
                                        min=0,
                                        max=num_failing - 1,
                                        step=1,
                                        ml=10,
                                        mr=10,
                                    ),
                                    html.Div(id=f"{test_name}-failing-prompt"),
                                ]
                            ),
                        ],
                    ),
                ]
            )
        ],
        value=test_name,
    )


from .utils import NAMES

tab_lists = dmc.TabsList([dmc.Tab(name, value=name) for name in NAMES])
tab_panels = [get_tab(name) for name in NAMES]

# callback


def make_callbacks(name):
    funcs = []

    @app.callback(
        [
            Output(f"{name}-passing-slider", "marks"),
            Output(f"{name}-passing-slider", "style"),
            Output(f"{name}-passing-slider", "max"),
            Output(f"{name}-failing-slider", "marks"),
            Output(f"{name}-failing-slider", "style"),
            Output(f"{name}-failing-slider", "max"),
            Output(f"{name}-passing-graph", "figure"),
            Output(f"{name}-failing-graph", "figure"),
        ],
        [Input("runs-group", "data"), Input("group-name", "children")],
    )
    def update_slider(subset_name, group_name):
        preprocessing_data = global_store(subset_name)
        run = preprocessing_data["runs"][group_name]

        passed_values = []
        failed_values = []
        data_plot_passing = []
        data_plot_failing = []
        for i, e in enumerate(run[f"{name}_instances"].array[0]):
            if e["passed"]:
                passed_values.append(i)
                data_plot_passing.append(
                    pd.DataFrame(
                        {"x": [e["correct_answer"]], "y": [e["answer"]]}, index=[i]
                    )
                )
            else:
                failed_values.append(i)
                data_plot_failing.append(
                    pd.DataFrame(
                        {"x": [e["correct_answer"]], "y": [e["answer"]]}, index=[i]
                    )
                )

        num_passing = len(passed_values)
        marks_passing = [{"value": i, "label": str(i)} for i in range(0, num_passing)]
        max_passing = num_passing - 1
        num_failing = len(failed_values)
        marks_failing = [{"value": i, "label": str(i)} for i in range(0, num_failing)]
        max_failing = num_failing - 1

        style_passing = {"display": "none" if num_passing == 0 else "block"}

        style_failing = {"display": "none" if num_failing == 0 else "block"}

        # plot with x-axis with correct values, and y-axis with answer
        data_plot_passing = (
            pd.concat(data_plot_passing)
            if data_plot_passing
            else pd.DataFrame({"x": [], "y": []})
        )
        data_plot_failing = (
            pd.concat(data_plot_failing)
            if data_plot_failing
            else pd.DataFrame({"x": [], "y": []})
        )

        fig_passing = px.scatter(
            data_plot_passing,
            x="x",
            y="y",
            title="Passing instances",
            labels={"x": "Correct answer", "y": "Answer"},
        )
        fig_failing = px.scatter(
            data_plot_failing,
            x="x",
            y="y",
            title="Failing instances",
            labels={"x": "Correct answer", "y": "Answer"},
        )

        return (
            marks_passing,
            style_passing,
            max_passing,
            marks_failing,
            style_failing,
            max_failing,
            fig_passing,
            fig_failing,
        )

    funcs.append(update_slider)

    scroll_to_bottom = """
    <script>
    window.onload = function() {
        window.scrollTo(0, document.body.scrollHeight);
    }
    </script>
    """

    @app.callback(
        Output(f"{name}-passing-prompt", "children"),
        [
            Input("runs-group", "data"),
            Input("group-name", "children"),
            Input(f"{name}-passing-slider", "value"),
        ],
    )
    def update_passing_prompt(subset_name, group_name, value):
        preprocessing_data = global_store(subset_name)
        run = preprocessing_data["runs"][group_name]

        passed_values = []
        failed_values = []
        for i, e in enumerate(run[f"{name}_instances"].array[0]):
            if e["passed"]:
                passed_values.append(i)
            else:
                failed_values.append(i)

        if len(passed_values) == 0:
            return "No passing instances"
        real_value = passed_values[value]
        h = run[f"{name}_instances"].array[0][real_value]

        return html.Iframe(
            sandbox="allow-scripts",
            srcDoc=("".join(h["html_prompt"]).replace("\n", "<br>")) + scroll_to_bottom,
            width="100%",
            height="600",
        )

    funcs.append(update_passing_prompt)

    @app.callback(
        Output(f"{name}-failing-prompt", "children"),
        [
            Input("runs-group", "data"),
            Input("group-name", "children"),
            Input(f"{name}-failing-slider", "value"),
        ],
    )
    def update_failing_prompt(subset_name, group_name, value):

        preprocessing_data = global_store(subset_name)
        run = preprocessing_data["runs"][group_name]

        passed_values = []
        failed_values = []
        for i, e in enumerate(run[f"{name}_instances"].array[0]):
            if e["passed"]:
                passed_values.append(i)
            else:
                failed_values.append(i)

        if len(failed_values) == 0:
            return html.Div("No failing instances")
        real_value = failed_values[value]
        h = run[f"{name}_instances"].array[0][real_value]

        return html.Iframe(
            sandbox="allow-scripts",
            srcDoc=("".join(h["html_prompt"]).replace("\n", "<br>")) + scroll_to_bottom,
            width="100%",
            height="600",
        )

    funcs.append(update_failing_prompt)

    @app.callback(
        Output(f"{name}-score-mean", "children"),
        [Input("runs-group", "data"), Input("group-name", "children")],
    )
    def update_score(subset_name, group_name):
        preprocessing_data = global_store(subset_name)
        run = preprocessing_data["runs"][group_name]

        return f"Mean score: {run[f'{name}_score_mean'].array[0]:.2f}"

    funcs.append(update_score)

    return funcs


callbacks = []
for name in NAMES:
    callbacks.append(make_callbacks(name))

details_layout = dmc.Stack(
    [
        dmc.Header(
            ml=30,
            height=100,
            children=[
                dmc.Title(children="Details", order=1),
                dmc.Text(id="group-name"),
                dmc.Text(id="group-config"),
            ],
        ),
        dmc.Tabs([tab_lists, *tab_panels], value=NAMES[0], m=20),
    ]
)


@app.callback(
    Output("group-name", "children"),
    Input("url", "pathname"),
)
def update_group_name(url):
    if not "/details/" in url:
        return dash.no_update

    # remove duplicate / in url
    url = "/".join([u for u in url.split("/") if u != ""])
    group_name = url.split("/")[2]
    return group_name


@app.callback(
    Output("group-config", "children"),
    [
        Input("runs-group", "data"),
        Input("group-name", "children"),
    ],
)
def update_group_config(subset_name, group_name):
    preprocessing_data = global_store(subset_name)
    run = preprocessing_data["runs"][group_name]

    return str(run[["config.llm.path", "config.llm.cot_prompt"]].iloc[0].to_dict())
