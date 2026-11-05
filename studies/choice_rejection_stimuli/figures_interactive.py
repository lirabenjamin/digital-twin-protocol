"""Interactive Figure 4: correlations among the 100 candidate moderators across the 100 stimuli, as a Plotly heatmap
for the online report (output/moderators/moderator_cor.html). Same matrix and ordering as moderation.R (hierarchical clustering,
complete linkage, on 1 - r). No axis labels: hovering a cell shows both moderators and r.

    python3 figures_interactive.py
"""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

HERE = __import__("pathlib").Path(__file__).resolve().parent
OUT = HERE / "output" / "moderators"

cor = pd.read_csv(OUT / "moderator_cor.csv").set_index("moderator")
names = {m["id"]: m["name"] for m in json.loads((HERE / "moderators.json").read_text())}
ok = cor.index[cor.notna().all(axis=1)]
cor = cor.loc[ok, ok]
order = leaves_list(linkage(squareform(1 - cor.to_numpy(), checks=False), method="complete"))
cor = cor.iloc[order, order]
labels = [names[m] for m in cor.index]

fig = go.Figure(go.Heatmap(
    z=cor.to_numpy(), x=labels, y=labels, zmin=-1, zmax=1,
    colorscale=[[0, "#2166ac"], [0.5, "#ffffff"], [1, "#b2182b"]],
    colorbar=dict(title="r", thickness=12, len=0.6),
    hovertemplate="%{y}<br>%{x}<br>r = %{z:.2f}<extra></extra>",
))
fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, autorange="reversed", scaleanchor="x")
fig.update_layout(width=720, height=680, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white",
                  paper_bgcolor="white")
fig.write_html(OUT / "moderator_cor.html", include_plotlyjs="cdn", full_html=True, config={"displaylogo": False})
print(f"wrote {OUT / 'moderator_cor.html'} ({len(labels)} moderators)")
