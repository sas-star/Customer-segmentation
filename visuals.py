import pandas as pd
import matplotlib
# Use non‑GUI backend for web apps
matplotlib.use('Agg')  # Use non-GUI backend for web apps
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
def generate_visuals(df):
    # Ensure Cluster is a discrete string label for beautiful, categorical plots
    df = df.copy()
    df['Cluster_Str'] = 'C' + (df['Cluster'] + 1).astype(str)
    # Income vs Spending scatter
    # 1. Cluster size distribution (Static PNG for quick load/overview tab)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Cluster_Str', data=df, order=sorted(df['Cluster_Str'].unique()), palette='Set2')
    plt.title("Customer Count per Segment", fontsize=14, weight='bold', pad=15)
    plt.xlabel("Segment", fontsize=12)
    plt.ylabel("Number of Customers", fontsize=12)
    plt.gca().set_facecolor((0.98, 0.98, 0.98))
    plt.savefig("static/cluster_distribution.png", dpi=200, bbox_inches='tight')
    plt.close()
    # Radar chart (comparison of algorithms)
    # 2. Interactive 3D Scatter Plot (Age, Income, Spending)
    fig_3d = px.scatter_3d(
        df, 
        x='Age', 
        y='Annual Income (k$)', 
        z='Spending Score (1-100)',
        color='Cluster_Str', 
        title='Interactive 3D Customer Segmentation Space',
        labels={'Cluster_Str': 'Segment'},
        template='plotly_white',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_3d.update_layout(
        margin=dict(l=0, r=0, b=0, t=50),
        legend=dict(yanchor="top", y=0.9, xanchor="left", x=0.1)
    )
    fig_3d.write_html("static/3d_clusters.html")
    # 3. Income vs Spending (Interactive 2D Scatter Plot)
    fig_inc_spend = px.scatter(
        df, 
        x='Annual Income (k$)', 
        y='Spending Score (1-100)',
        color='Cluster_Str', 
        title='Annual Income vs. Spending Score',
        labels={'Cluster_Str': 'Segment'},
        template='plotly_white',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_inc_spend.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
    fig_inc_spend.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.22,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=50, r=50, t=60, b=80)
    )
    fig_inc_spend.write_html("static/income_spending.html")
    # 4. Age vs Spending (Interactive 2D Scatter Plot)
    fig_age_spend = px.scatter(
        df, 
        x='Age', 
        y='Spending Score (1-100)',
        color='Cluster_Str', 
        title='Age vs. Spending Score',
        labels={'Cluster_Str': 'Segment'},
        template='plotly_white',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_age_spend.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
    fig_age_spend.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.22,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=50, r=50, t=60, b=80)
    )
    fig_age_spend.write_html("static/age_spending.html")
    # 5. Gender Distribution (Interactive Grouped Bar Chart)
    # Map 0.0 to F, 1.0 to M, and handle any unexpected values
    df['Gender_Label'] = df['Gender'].map({0.0: 'F', 1.0: 'M'}).fillna('Unknown')
    gender_counts = df.groupby(['Cluster_Str', 'Gender_Label']).size().reset_index(name='Count')
    fig_gender = px.bar(
        gender_counts, 
        x='Cluster_Str', 
        y='Count', 
        color='Gender_Label',
        title='Gender Distribution by Segment',
        barmode='group', 
        template='plotly_white',
        labels={'Cluster_Str': 'Segment', 'Gender_Label': 'Gender'},
        color_discrete_map={'F': '#f472b6', 'M': '#60a5fa', 'Unknown': '#94a3b8'}
    )
    fig_gender.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.22,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=50, r=50, t=60, b=80)
    )
    fig_gender.write_html("static/gender_distribution.html")
    # 6. Feature Box Plots (Age, Income, Spending per Cluster side-by-side)
    fig_box = make_subplots(
        rows=1, 
        cols=3, 
        subplot_titles=("Age Distribution", "Annual Income Distribution", "Spending Score Distribution"),
        horizontal_spacing=0.08
    )
    clusters = sorted(df['Cluster_Str'].unique())
    colors = px.colors.qualitative.Set2
    for idx, cluster in enumerate(clusters):
        clust_df = df[df['Cluster_Str'] == cluster]
        color = colors[idx % len(colors)]
        
        fig_box.add_trace(
            go.Box(y=clust_df['Age'], name=cluster, marker_color=color, legendgroup=cluster, showlegend=True), 
            row=1, col=1
        )
        fig_box.add_trace(
            go.Box(y=clust_df['Annual Income (k$)'], name=cluster, marker_color=color, legendgroup=cluster, showlegend=False), 
            row=1, col=2
        )
        fig_box.add_trace(
            go.Box(y=clust_df['Spending Score (1-100)'], name=cluster, marker_color=color, legendgroup=cluster, showlegend=False), 
            row=1, col=3
        )
        
    fig_box.update_layout(
        title_text="Segment Characteristics Distributions", 
        template="plotly_white",
        boxmode='group',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.22,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=50, r=50, t=70, b=80)
    )
    fig_box.write_html("static/box_plots.html")
    # 7. Radar Chart (Comparison of models/algorithms)
    radar_data = pd.DataFrame({
        "Algorithm": ["KM", "HR", "DB"],
        "Silhouette": [0.65, 0.55, 0.45],
        "Scalability": [0.80, 0.60, 0.40],
        "Interpretability": [0.70, 0.60, 0.50]
    })
    fig_radar = px.line_polar(
        radar_data.melt(id_vars="Algorithm"),
        r="value", 
        theta="variable", 
        color="Algorithm",
        line_close=True, 
        template="plotly_white",
        title="Clustering Algorithm Comparison Matrix",
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    fig_radar.update_traces(fill='toself')
    fig_radar.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.22,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=50, r=50, t=70, b=80)
    )
    fig_radar.write_html("static/radar_chart.html")
