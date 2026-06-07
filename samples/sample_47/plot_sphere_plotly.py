import plotly.graph_objects as go

xs, ys, zs = [], [], []

with open("./build/sample_47/sphere.txt") as f:
    for line in f:
        x, y, z = map(float, line.split())
        xs.append(x)
        ys.append(y)
        zs.append(z)

fig = go.Figure(
    data=[
        go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="markers",
            marker=dict(size=3, opacity=0.8)
        )
    ]
)

fig.update_layout(
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z",
        xaxis=dict(range=[-1, 1]),
        yaxis=dict(range=[-1, 1]),
        zaxis=dict(range=[-1, 1]),
        aspectmode="cube" # equal unit scale x/y/z
    )
)

fig.show()
