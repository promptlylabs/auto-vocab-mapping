import plotly.graph_objs as go

def parallel(df, label='metric'):
    df = df.fillna(0)
    
    dimensions_dict = [dict(range = [min(df[cname]),max(df[cname])],
                    constraintrange = [min(df[cname]),max(df[cname])],
                    label = cname, values = df[cname]) for cname in df.columns if cname != label]
    
    df['dummy'] = df.reset_index().index
    dimensions_dict.append(dict(range=[df['dummy'].min(),df['dummy'].max()],
                       tickvals = df['dummy'], ticktext = df[label],
                       label=label, values=df['dummy']))
    dimensions = list(dimensions_dict)

    fig = go.Figure(data=go.Parcoords(line = dict(color = df['dummy'], colorscale=['rgba(99,110,250,0.9)',
    'rgba(239,85,59,0.9)',
    'rgba(0,204,150,0.9)',
    'rgba(171,99,250,0.9)',
    'rgba(255,161,90,0.9)',
    'rgba(25,211,243,0.9)',
    'rgba(255,102,146,0.9)',
    'rgba(182,232,128,0.9)',
    'rgba(255,151,255,0.9)',
    'rgba(254,203,82,0.9)']
                    ), dimensions=dimensions))
    
    fig.update_layout(
        title=f"{label} eval",
        width=1200,
        height=500,
        font=dict(
            family="Sans-serif",
            size=15,
            color="Black"
        )
    )
    
    fig.show()
    fig.write_image(f"../lib/figures/{label}.png")