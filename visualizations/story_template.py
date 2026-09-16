#!/usr/bin/env python
"""
Gabarit commun des pages de visualisation du portfolio.

Centralise la mise en page HTML (barre de navigation, boutons, navigation
precedent/suivant) pour que toutes les pages generees partagent le meme
design que la page projet.
"""

from pathlib import Path

import pandas as pd

# Parcours de lecture : l'ordre determine la navigation precedent / suivant.
STORY_ORDER = [
    ("01_probleme_churn.html", "Le probleme"),
    ("insight_days_since_last_order.html", "Insight : delai d'achat"),
    ("insight_purchase_frequency.html", "Insight : frequence"),
    ("segmentation_rfm.html", "Segmentation RFM"),
    ("02_solution_ml.html", "La solution ML"),
    ("feature_importance.html", "Features importantes"),
    ("03_impact_business.html", "Impact business"),
    ("00_dashboard_complet.html", "Dashboard complet"),
]


def _neighbours(filename):
    """Retourne (precedent, suivant) sous forme de tuples (fichier, libelle)."""
    names = [item[0] for item in STORY_ORDER]
    if filename not in names:
        return None, None
    idx = names.index(filename)
    previous = STORY_ORDER[idx - 1] if idx > 0 else None
    following = STORY_ORDER[idx + 1] if idx < len(STORY_ORDER) - 1 else None
    return previous, following


def normalize_recency(df_features, column="days_since_last_order"):
    """
    Recalcule le delai depuis la derniere commande a partir de la date de
    reference du jeu de donnees.

    Le jeu de donnees livre contient des valeurs negatives (la date de
    reference utilisee au feature engineering est anterieure aux dernieres
    commandes), ce qui vidait les graphiques par tranche de recence.
    On recalcule donc la recence par rapport a la derniere commande observee.
    """
    if "last_order_date" not in df_features.columns:
        return df_features

    df = df_features.copy()
    last_order = pd.to_datetime(df["last_order_date"])
    reference_date = last_order.max()
    df[column] = (reference_date - last_order).dt.days.clip(lower=0)
    return df


def write_story_page(title, subtitle, paragraphs, fig, output_path, badge=None):
    """Genere une page HTML : entete, lecture business, graphique, navigation."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
    paragraphs_html = "\n            ".join(f"<p>{p}</p>" for p in paragraphs)

    previous, following = _neighbours(output_path.name)
    prev_html = (
        f'<a class="btn btn-secondary" href="{previous[0]}">'
        f'<i class="fas fa-arrow-left"></i> {previous[1]}</a>'
        if previous
        else '<a class="btn btn-secondary" href="../index.html#visualisations">'
        '<i class="fas fa-arrow-left"></i> Toutes les visualisations</a>'
    )
    next_html = (
        f'<a class="btn btn-primary" href="{following[0]}">'
        f'{following[1]} <i class="fas fa-arrow-right"></i></a>'
        if following
        else '<a class="btn btn-primary" href="../index.html#contact">'
        'Discuter du projet <i class="fas fa-arrow-right"></i></a>'
    )
    badge_html = f'<span class="viz-tag {badge[0]}">{badge[1]}</span>\n            ' if badge else ""

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} | Prediction de Churn</title>
    <meta name="description" content="{subtitle}">
    <link rel="stylesheet" href="../assets/css/style.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
</head>
<body class="viz-page">
    <nav class="viz-topbar">
        <div class="container">
            <p class="viz-breadcrumb">
                <a href="../index.html">Prediction de Churn</a>
                &rsaquo; <a href="../index.html#visualisations">Visualisations</a>
                &rsaquo; {title}
            </p>
            <a class="btn btn-secondary" href="../index.html#visualisations">
                <i class="fas fa-arrow-left"></i> Retour au projet
            </a>
        </div>
    </nav>

    <main class="viz-wrapper">
        <header class="viz-header">
            {badge_html}<h1>{title}</h1>
            <p class="viz-subtitle">{subtitle}</p>
        </header>

        <section class="viz-text">
            {paragraphs_html}
        </section>

        <section class="viz-chart">
            {fig_html}
        </section>

        <nav class="viz-actions">
            {prev_html}
            <span class="spacer"></span>
            {next_html}
        </nav>
    </main>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    print(f"OK {output_path.name}")
