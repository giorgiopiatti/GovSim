import colorsys

import plotly.express as px
import plotly.io as pio


def generate_colors_paper(n_runs):
    # Access the default Plotly color sequence
    plotly_template = pio.templates["plotly"]
    default_color_sequence = plotly_template.layout["colorway"]
    colors = []
    for i in range(n_runs):
        # Use modulo to cycle through the color sequence if n_runs exceeds the number of available colors
        color_hex = default_color_sequence[i % len(default_color_sequence)]

        # Convert HEX color to RGB
        rgb = tuple(int(color_hex.lstrip("#")[j : j + 2], 16) for j in (0, 2, 4))

        # Create solid and translucent versions of the color
        rgba_solid = "rgba({}, {}, {}, 1.0)".format(rgb[0], rgb[1], rgb[2])
        rgba_translucent = "rgba({}, {}, {}, 0.15)".format(rgb[0], rgb[1], rgb[2])

        colors.append([rgba_solid, rgba_translucent])
    return colors


def generate_colors(n_runs):
    colors = []
    for i in range(n_runs):
        hue = i / n_runs  # Vary the hue value evenly across the spectrum
        saturation = 0.9  # Keep saturation high for vibrant colors
        value = 0.9  # Keep value high to avoid dark colors
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        rgba_solid = "rgba({}, {}, {}, 1.0)".format(
            int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
        )
        rgba_translucent = "rgba({}, {}, {}, 0.2)".format(
            int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
        )
        colors.append([rgba_solid, rgba_translucent])
    return colors

from dash import dcc, html


def create_table(df):
    columns, values = df.columns, df.values
    header = [html.Tr([html.Th(col) for col in columns])]
    rows = [html.Tr([html.Td(cell) for cell in row]) for row in values]
    table = [html.Thead(header), html.Tbody(rows)]
    return table


import plotly.io as pio

pio.templates.default = "plotly_white"


def prepare_fig_for_export(fig):

    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="center", x=0.5),
        showlegend=True,
    )

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
        height=200,
    )
    return fig
