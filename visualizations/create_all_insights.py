#!/usr/bin/env python
"""
Script pour créer toutes les visualisations d'insights pertinentes.

Ce script génère des visualisations qui racontent vraiment l'histoire
et démontrent les compétences en Data Science et storytelling.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

DATA_DIR = project_root / "data"
VIZ_DIR = project_root / "docs" / "visualizations"
REPORTS_DIR = project_root / "reports"

VIZ_DIR.mkdir(parents=True, exist_ok=True)


def write_story_page(title, subtitle, paragraphs, fig, output_path):
    """Crée une page HTML avec explications et graphique Plotly."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
    paragraphs_html = "\n".join([f"<p>{p}</p>" for p in paragraphs])

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} | Insights Churn</title>
    <link rel="stylesheet" href="../assets/css/style.css">
    <style>
        body {{ background: #f8fafc; }}
        .viz-wrapper {{ max-width: 1100px; margin: 40px auto; padding: 0 20px; }}
        .viz-header {{ margin-bottom: 16px; }}
        .viz-subtitle {{ color: #4b5563; margin-top: 8px; }}
        .viz-text {{ background: #ffffff; border-radius: 12px; padding: 20px; box-shadow: 0 10px 20px rgba(0,0,0,0.05); }}
        .viz-text p {{ margin: 0 0 12px 0; line-height: 1.6; }}
        .viz-chart {{ margin-top: 20px; background: #ffffff; border-radius: 12px; padding: 12px; box-shadow: 0 10px 20px rgba(0,0,0,0.05); }}
        .viz-actions {{ margin-top: 16px; display: flex; gap: 12px; flex-wrap: wrap; }}
        .viz-link {{ display: inline-block; padding: 10px 14px; border-radius: 8px; background: #111827; color: #ffffff; text-decoration: none; }}
        .viz-link.secondary {{ background: #e5e7eb; color: #111827; }}
    </style>
</head>
<body>
    <div class="viz-wrapper">
        <div class="viz-header">
            <a class="viz-link secondary" href="../index.html">← Retour à la page projet</a>
            <h1>{title}</h1>
            <p class="viz-subtitle">{subtitle}</p>
        </div>
        <div class="viz-text">
            {paragraphs_html}
        </div>
        <div class="viz-chart">
            {fig_html}
        </div>
        <div class="viz-actions">
            <a class="viz-link" href="../index.html#demo">Voir la démo complète</a>
            <a class="viz-link secondary" href="../index.html#impact">Retour à l'impact</a>
        </div>
    </div>
    <script src="../assets/js/main.js"></script>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    print(f"✅ {output_path.name}")


def load_data():
    """Charge les données."""
    print("📊 Chargement des données...")
    df_raw = pd.read_csv(DATA_DIR / "raw" / "customers.csv", parse_dates=['order_date'])
    df_features = pd.read_csv(DATA_DIR / "processed" / "features.csv")
    
    if 'last_order_date' in df_features.columns:
        df_features['last_order_date'] = pd.to_datetime(df_features['last_order_date'])
    
    metrics_path = REPORTS_DIR / "training_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
    else:
        metrics = {}
    
    print(f"✅ {len(df_raw)} transactions, {len(df_features)} clients")
    return df_raw, df_features, metrics


def create_insight_days_since_last_order(df_features):
    """Insight clé : Impact du délai depuis dernière commande."""
    print("\n🔍 Création insight : Délai depuis dernière commande...")
    
    # Créer des bins
    df_features['days_bin'] = pd.cut(
        df_features['days_since_last_order'],
        bins=[0, 30, 60, 90, 180, float('inf')],
        labels=['<30j', '30-60j', '60-90j', '90-180j', '>180j']
    )
    
    churn_by_days = df_features.groupby('days_bin')['churn'].agg(['mean', 'count'])
    churn_by_days['churn_pct'] = churn_by_days['mean'] * 100
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            'Taux de churn par délai depuis dernière commande',
            'Distribution : Fidèles vs À risque'
        )
    )
    
    # Graphique 1
    colors = ['#10b981' if x < 20 else '#f59e0b' if x < 50 else '#ef4444' 
             for x in churn_by_days['churn_pct']]
    
    fig.add_trace(
        go.Bar(
            x=churn_by_days.index,
            y=churn_by_days['churn_pct'],
            marker_color=colors,
            text=[f"{v:.1f}%" for v in churn_by_days['churn_pct']],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Taux: %{y:.1f}%<br>Clients: %{customdata}<extra></extra>',
            customdata=churn_by_days['count']
        ),
        row=1, col=1
    )
    
    # Graphique 2
    fig.add_trace(
        go.Histogram(
            x=df_features[df_features['churn']==0]['days_since_last_order'],
            name='Fidèles',
            marker_color='#10b981',
            opacity=0.7,
            nbinsx=30
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Histogram(
            x=df_features[df_features['churn']==1]['days_since_last_order'],
            name='À risque',
            marker_color='#ef4444',
            opacity=0.7,
            nbinsx=30
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=500,
        title_text="🔍 Insight Clé : Le délai depuis dernière commande est le meilleur prédicteur",
        showlegend=True,
        template="plotly_white"
    )
    fig.update_xaxes(title_text="Délai", row=1, col=1)
    fig.update_yaxes(title_text="Taux de churn (%)", row=1, col=1)
    fig.update_xaxes(title_text="Jours", row=1, col=2)
    fig.update_yaxes(title_text="Nombre de clients", row=1, col=2)
    
    output_path = VIZ_DIR / "insight_days_since_last_order.html"
    write_story_page(
        title="Insight : Délai depuis la dernière commande",
        subtitle="Le signal le plus clair pour anticiper le churn",
        paragraphs=[
            "Le taux de churn progresse rapidement après 60 jours sans achat, "
            "ce qui en fait un seuil opérationnel fort.",
            "La distribution met en évidence que les clients à risque sont concentrés "
            "dans les délais élevés, alors que les clients fidèles restent majoritairement "
            "sous 30 jours.",
            "Action recommandée : déclencher une campagne personnalisée dès 60 jours "
            "d’inactivité pour maximiser les chances de rétention."
        ],
        fig=fig,
        output_path=output_path
    )
    return fig


def create_insight_purchase_frequency(df_features):
    """Insight : Fréquence d'achat vs churn."""
    print("\n📊 Création insight : Fréquence d'achat...")
    
    df_features['freq_bin'] = pd.cut(
        df_features['purchase_frequency_30d'],
        bins=[0, 1, 3, 5, 10, float('inf')],
        labels=['0-1', '2-3', '4-5', '6-10', '>10']
    )
    
    churn_by_freq = df_features.groupby('freq_bin')['churn'].agg(['mean', 'count'])
    revenue_by_freq = df_features.groupby('freq_bin')['total_revenue'].mean()
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            'Taux de churn par fréquence',
            'Revenus moyens par fréquence'
        )
    )
    
    fig.add_trace(
        go.Bar(
            x=churn_by_freq.index,
            y=churn_by_freq['mean'] * 100,
            marker_color='#ef4444',
            text=[f"{v:.1f}%" for v in churn_by_freq['mean'] * 100],
            textposition='auto'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=revenue_by_freq.index,
            y=revenue_by_freq.values,
            marker_color='#10b981',
            text=[f"€{v:,.0f}" for v in revenue_by_freq.values],
            textposition='auto'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=500,
        title_text="📊 Insight : Fréquence d'achat inversement corrélée au churn",
        showlegend=False,
        template="plotly_white"
    )
    fig.update_xaxes(title_text="Fréquence (30j)", row=1, col=1)
    fig.update_yaxes(title_text="Taux de churn (%)", row=1, col=1)
    fig.update_xaxes(title_text="Fréquence (30j)", row=1, col=2)
    fig.update_yaxes(title_text="Revenus moyens (€)", row=1, col=2)
    
    output_path = VIZ_DIR / "insight_purchase_frequency.html"
    write_story_page(
        title="Insight : Fréquence d'achat",
        subtitle="Plus la fréquence est élevée, plus le churn diminue",
        paragraphs=[
            "Le taux de churn chute nettement quand la fréquence d’achat augmente, "
            "ce qui valide l’importance de l’engagement régulier.",
            "Les segments à faible fréquence présentent un churn élevé et une valeur "
            "client plus faible, ce qui oriente la priorisation des actions.",
            "Action recommandée : stimuler la fréquence avec des offres ciblées "
            "sur les segments 0-1 et 2-3 achats."
        ],
        fig=fig,
        output_path=output_path
    )
    return fig


def create_segmentation_rfm(df_features):
    """Segmentation RFM."""
    print("\n🎯 Création segmentation RFM...")
    
    # RFM
    df_features['R_score'] = pd.qcut(
        df_features['days_since_last_order'].rank(method='first'), 
        q=3, labels=[3, 2, 1], duplicates='drop'
    )
    df_features['F_score'] = pd.qcut(
        df_features['purchase_frequency_30d'].rank(method='first'), 
        q=3, labels=[1, 2, 3], duplicates='drop'
    )
    df_features['M_score'] = pd.qcut(
        df_features['total_revenue'].rank(method='first'), 
        q=3, labels=[1, 2, 3], duplicates='drop'
    )
    
    df_features['RFM_segment'] = (
        df_features['R_score'].astype(str) + 
        df_features['F_score'].astype(str) + 
        df_features['M_score'].astype(str)
    )
    
    def categorize_segment(rfm):
        if rfm in ['333', '332', '323', '322']:
            return 'Champions'
        elif rfm in ['313', '312', '311', '223', '222', '221']:
            return 'Clients fidèles'
        elif rfm in ['133', '132', '123', '122']:
            return 'À risque'
        elif rfm in ['111', '112', '121', '131']:
            return 'Perdus'
        else:
            return 'Autres'
    
    df_features['segment_name'] = df_features['RFM_segment'].apply(categorize_segment)
    
    segment_analysis = df_features.groupby('segment_name').agg({
        'churn': 'mean',
        'total_revenue': 'mean',
        'customer_id': 'count'
    }).round(2)
    segment_analysis.columns = ['Taux_churn', 'Revenu_moyen', 'Nb_clients']
    segment_analysis = segment_analysis.sort_values('Taux_churn', ascending=False)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=segment_analysis.index,
        y=segment_analysis['Taux_churn'] * 100,
        marker_color=['#ef4444' if x > 0.5 else '#f59e0b' if x > 0.3 else '#10b981' 
                     for x in segment_analysis['Taux_churn']],
        text=[f"{v:.1f}%" for v in segment_analysis['Taux_churn'] * 100],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="🎯 Segmentation RFM : Taux de churn par segment",
        xaxis_title="Segment",
        yaxis_title="Taux de churn (%)",
        height=500,
        template="plotly_white"
    )
    
    output_path = VIZ_DIR / "segmentation_rfm.html"
    write_story_page(
        title="Segmentation RFM",
        subtitle="Prioriser les actions selon la valeur et l'engagement",
        paragraphs=[
            "La segmentation RFM révèle des profils très différents : les ‘Champions’ "
            "ont un churn faible, tandis que les segments ‘À risque’ et ‘Perdus’ "
            "concentrent la majorité des départs.",
            "Cette vue facilite l’allocation du budget : retenir les clients à forte "
            "valeur plutôt que cibler indistinctement tout le portefeuille.",
            "Action recommandée : lancer des campagnes dédiées par segment avec des "
            "messages et offres différenciés."
        ],
        fig=fig,
        output_path=output_path
    )
    return fig, df_features


def create_feature_importance_viz(df_features):
    """Visualisation des features importantes."""
    print("\n🔍 Création visualisation features importantes...")
    
    feature_cols = [col for col in df_features.columns 
                   if col not in ['customer_id', 'first_order_date', 'last_order_date', 'churn', 
                                 'days_bin', 'freq_bin', 'R_score', 'F_score', 'M_score', 
                                 'RFM_segment', 'segment_name']]
    
    correlations = df_features[feature_cols + ['churn']].corr()['churn'].abs().sort_values(ascending=False)
    correlations = correlations[correlations.index != 'churn'].head(15)
    
    colors = ['#ef4444' if x > 0.3 else '#f59e0b' if x > 0.2 else '#10b981' 
             for x in correlations.values]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=correlations.values,
        y=correlations.index,
        orientation='h',
        marker_color=colors,
        text=[f"{v:.3f}" for v in correlations.values],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="🔍 Top 15 Features les Plus Corrélées au Churn",
        xaxis_title="Corrélation absolue",
        yaxis_title="Features",
        height=600,
        template="plotly_white"
    )
    
    output_path = VIZ_DIR / "feature_importance.html"
    write_story_page(
        title="Features Importantes",
        subtitle="Les variables qui expliquent réellement le churn",
        paragraphs=[
            "Les variables comportementales dominent l’explication du churn, "
            "notamment le délai depuis la dernière commande et la fréquence d’achat.",
            "Ces résultats confirment que l’engagement client est un meilleur prédicteur "
            "que les seuls indicateurs financiers.",
            "Action recommandée : suivre ces features dans un dashboard opérationnel "
            "pour anticiper les risques et déclencher des actions ciblées."
        ],
        fig=fig,
        output_path=output_path
    )
    return fig


def main():
    """Fonction principale."""
    print("=" * 60)
    print("🎨 CRÉATION DES VISUALISATIONS D'INSIGHTS")
    print("=" * 60)
    
    df_raw, df_features, metrics = load_data()
    
    # Créer toutes les visualisations d'insights
    create_insight_days_since_last_order(df_features.copy())
    create_insight_purchase_frequency(df_features.copy())
    _, df_features = create_segmentation_rfm(df_features.copy())
    create_feature_importance_viz(df_features)
    
    print("\n" + "=" * 60)
    print("✅ TOUTES LES VISUALISATIONS D'INSIGHTS ONT ÉTÉ CRÉÉES")
    print("=" * 60)
    print(f"\n📁 Fichiers dans : {VIZ_DIR}")


if __name__ == "__main__":
    main()
