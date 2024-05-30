import colorsys

NAMES = [
    "multiple_math_consequence_after_using_same_amount",
    "multiple_math_shrinking_limit",
    "multiple_sim_consequence_after_using_same_amount",
    "multiple_sim_shrinking_limit",
    "multiple_sim_act_standard",
    "multiple_sim_act_universalization",
    "multiple_math_shrinking_limit_assumption",
    "multiple_sim_shrinking_limit_assumption",
]


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
