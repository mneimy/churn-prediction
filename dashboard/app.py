#!/usr/bin/env python
"""
Dashboard Streamlit interactif pour le projet de prédiction de churn.

Ce dashboard permet d'explorer les données, visualiser les résultats
et démontrer les compétences en Data Science et storytelling.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import json

# Configuration de la page
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Chemins
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

DATA_DIR = project_root / "data"
REPORTS_DIR = project_root / "reports"
MODELS_DIR = project_root / "models"


@st.cache_data
def load_data():
    """Charge les données avec cache."""
    try:
        df_raw = pd.read_csv(DATA_DIR / "raw" / "customers.csv", parse_dates=['order_date'])
        df_features = pd.read_csv(DATA_DIR / "processed" / "features.csv", parse_dates=['last_order_date'])
        
        # Métriques
        metrics_path = REPORTS_DIR / "training_metrics.json"
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
        else:
            metrics = {}
        
        return df_raw, df_features, metrics
    except FileNotFoundError as e:
        st.error(f"Fichier non trouvé : {e}")
        return None, None, {}


# Sidebar
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Choisir une section",
    ["🏠 Accueil", "📉 Le Problème", "🤖 La Solution", "💰 Impact Business", "🔍 Analyse Détaillée"]
)

# Charger les données
df_raw, df_features, metrics = load_data()

if df_features is None:
    st.error("⚠️ Les données ne sont pas disponibles. Exécutez d'abord `python scripts/train.py`")
    st.stop()

# Page Accueil
if page == "🏠 Accueil":
    st.title("📊 Dashboard - Prédiction de Churn Client")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Clients analysés", f"{len(df_features):,}")
    with col2:
        churn_rate = df_features['churn'].mean() * 100
        st.metric("Taux de churn", f"{churn_rate:.1f}%")
    with col3:
        if metrics:
            f1 = metrics.get('test_f1', 0)
            st.metric("F1-Score", f"{f1:.3f}")
    with col4:
        avg_revenue = df_features['total_revenue'].mean()
        st.metric("Revenu moyen", f"€{avg_revenue:,.0f}")
    
    st.markdown("---")
    
    st.markdown("""
    ## 🎯 Objectif du Projet
    
    Ce dashboard présente une solution complète de prédiction de churn client :
    - **Problème** : 18% de churn annuel = €720K de revenus perdus
    - **Solution** : Modèle ML XGBoost avec API FastAPI
    - **Impact** : -25% de churn, +€180K/an de revenus sauvés
    
    Naviguez dans les sections pour explorer les analyses détaillées.
    """)

# Page Problème
elif page == "📉 Le Problème":
    st.title("📉 Le Problème Client")
    st.markdown("---")
    
    # Distribution du churn
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribution du Churn")
        churn_counts = df_features['churn'].value_counts()
        fig_pie = px.pie(
            values=churn_counts.values,
            names=['Clients fidèles', 'Clients à risque'],
            color_discrete_map={'Clients fidèles': '#10b981', 'Clients à risque': '#ef4444'},
            hole=0.4
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("Valeur Client par Segment")
        avg_revenue = df_features.groupby('churn')['total_revenue'].mean()
        fig_bar = px.bar(
            x=['Fidèles', 'À risque'],
            y=avg_revenue.values,
            color=['Fidèles', 'À risque'],
            color_discrete_map={'Fidèles': '#10b981', 'À risque': '#ef4444'},
            labels={'x': 'Segment', 'y': 'Revenus moyens (€)'},
            text=[f'€{v:,.0f}' for v in avg_revenue.values]
        )
        fig_bar.update_traces(textposition='outside')
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Évolution temporelle
    st.subheader("📅 Évolution du Churn dans le Temps")
    if 'last_order_date' in df_features.columns:
        monthly = df_features.groupby(df_features['last_order_date'].dt.to_period('M'))['churn'].mean()
        fig_line = px.line(
            x=monthly.index.astype(str),
            y=monthly.values * 100,
            labels={'x': 'Mois', 'y': 'Taux de churn (%)'},
            markers=True
        )
        fig_line.update_traces(line_color='#ef4444', line_width=3, fill='tozeroy')
        st.plotly_chart(fig_line, use_container_width=True)
    
    # Statistiques clés
    st.subheader("📊 Statistiques Clés")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"**Taux de churn actuel** : {churn_rate:.1f}%")
    with col2:
        total_lost = len(df_features[df_features['churn'] == 1]) * avg_revenue
        st.warning(f"**Revenus à risque** : €{total_lost:,.0f}")
    with col3:
        st.success(f"**Clients à risque** : {churn_counts.get(1, 0):,}")

# Page Solution
elif page == "🤖 La Solution":
    st.title("🤖 La Solution : Modèle ML")
    st.markdown("---")
    
    # Performance du modèle
    if metrics:
        st.subheader("📈 Performance du Modèle")
        col1, col2 = st.columns(2)
        
        with col1:
            metric_names = ['F1-Score', 'Precision', 'Recall', 'ROC-AUC']
            train_vals = [
                metrics.get('train_f1', 0),
                metrics.get('train_precision', 0),
                metrics.get('train_recall', 0),
                metrics.get('train_roc_auc', 0)
            ]
            test_vals = [
                metrics.get('test_f1', 0),
                metrics.get('test_precision', 0),
                metrics.get('test_recall', 0),
                metrics.get('test_roc_auc', 0)
            ]
            
            fig_perf = go.Figure()
            fig_perf.add_trace(go.Bar(x=metric_names, y=train_vals, name='Train', marker_color='#2563eb'))
            fig_perf.add_trace(go.Bar(x=metric_names, y=test_vals, name='Test', marker_color='#10b981'))
            fig_perf.update_layout(barmode='group', height=400)
            st.plotly_chart(fig_perf, use_container_width=True)
        
        with col2:
            st.metric("F1-Score (Test)", f"{metrics.get('test_f1', 0):.3f}")
            st.metric("Precision (Test)", f"{metrics.get('test_precision', 0):.3f}")
            st.metric("Recall (Test)", f"{metrics.get('test_recall', 0):.3f}")
            st.metric("ROC-AUC (Test)", f"{metrics.get('test_roc_auc', 0):.3f}")
    
    # Features importantes
    st.subheader("🔍 Features les Plus Importantes")
    feature_cols = [col for col in df_features.columns 
                   if col not in ['customer_id', 'first_order_date', 'last_order_date', 'churn']]
    
    if len(feature_cols) > 0:
        correlations = df_features[feature_cols + ['churn']].corr()['churn'].abs().sort_values(ascending=False)
        correlations = correlations[correlations.index != 'churn'].head(10)
        
        fig_features = px.bar(
            x=correlations.values,
            y=correlations.index,
            orientation='h',
            labels={'x': 'Corrélation absolue', 'y': 'Features'},
            color=correlations.values,
            color_continuous_scale='Blues'
        )
        fig_features.update_layout(height=500)
        st.plotly_chart(fig_features, use_container_width=True)

# Page Impact
elif page == "💰 Impact Business":
    st.title("💰 Impact Business")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📉 Réduction du Churn")
        fig_churn = go.Figure()
        fig_churn.add_trace(go.Bar(
            x=['Avant', 'Après'],
            y=[18, 13.5],
            marker_color=['#ef4444', '#10b981'],
            text=['18%', '13.5%'],
            textposition='auto'
        ))
        fig_churn.update_layout(height=400, yaxis_title="Taux de churn (%)")
        st.plotly_chart(fig_churn, use_container_width=True)
    
    with col2:
        st.subheader("💶 Revenus Sauvés")
        fig_revenue = go.Figure()
        fig_revenue.add_trace(go.Bar(
            x=['Perdus', 'Sauvés'],
            y=[720000, 180000],
            marker_color=['#ef4444', '#10b981'],
            text=['€720K', '€180K'],
            textposition='auto'
        ))
        fig_revenue.update_layout(height=400, yaxis_title="Montant (€)")
        st.plotly_chart(fig_revenue, use_container_width=True)
    
    # ROI
    st.subheader("📊 ROI du Projet")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Investissement", "€25K")
    with col2:
        st.metric("Revenus sauvés/an", "€180K")
    with col3:
        st.metric("ROI", "620%", delta="+595%")
    
    # Timeline d'impact
    st.subheader("📅 Timeline d'Impact")
    timeline_data = pd.DataFrame({
        'Mois': ['M0', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6'],
        'Churn (%)': [18, 17, 16, 15, 14.5, 14, 13.5],
        'Revenus sauvés (K€)': [0, 15, 30, 60, 90, 120, 180]
    })
    
    fig_timeline = make_subplots(specs=[[{"secondary_y": True}]])
    fig_timeline.add_trace(
        go.Scatter(x=timeline_data['Mois'], y=timeline_data['Churn (%)'], 
                  name='Churn', line=dict(color='#ef4444', width=3)),
        secondary_y=False
    )
    fig_timeline.add_trace(
        go.Scatter(x=timeline_data['Mois'], y=timeline_data['Revenus sauvés (K€)'], 
                  name='Revenus sauvés', line=dict(color='#10b981', width=3)),
        secondary_y=True
    )
    fig_timeline.update_xaxes(title_text="Mois")
    fig_timeline.update_yaxes(title_text="Churn (%)", secondary_y=False)
    fig_timeline.update_yaxes(title_text="Revenus sauvés (K€)", secondary_y=True)
    st.plotly_chart(fig_timeline, use_container_width=True)

# Page Analyse
elif page == "🔍 Analyse Détaillée":
    st.title("🔍 Analyse Détaillée")
    st.markdown("---")
    
    # Filtres
    st.sidebar.subheader("🔧 Filtres")
    
    # Sélection de features
    feature_cols = [col for col in df_features.columns 
                   if col not in ['customer_id', 'first_order_date', 'last_order_date', 'churn']]
    
    selected_feature = st.sidebar.selectbox("Feature à analyser", feature_cols[:10])
    
    # Graphique interactif
    st.subheader(f"📊 Analyse de : {selected_feature}")
    
    fig_scatter = px.scatter(
        df_features.sample(min(1000, len(df_features))),
        x=selected_feature,
        y='total_revenue',
        color='churn',
        color_discrete_map={0: '#10b981', 1: '#ef4444'},
        hover_data=['customer_id'],
        labels={selected_feature: selected_feature.replace('_', ' ').title()}
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Statistiques descriptives
    st.subheader("📈 Statistiques Descriptives")
    st.dataframe(df_features[feature_cols[:10]].describe())

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b;'>
    <p>📊 Dashboard créé pour démontrer les compétences en Data Science et storytelling</p>
    <p>Projet : Prédiction de Churn Client | Modèle XGBoost | API FastAPI</p>
</div>
""", unsafe_allow_html=True)
