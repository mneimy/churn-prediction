"""
Module de calcul ROI et métriques business pour la prédiction de churn.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ROICalculator:
    """Calculeur de ROI et métriques business pour le projet de churn."""
    
    def __init__(
        self,
        project_cost: float = 25000.0,
        avg_customer_value: float = 1000.0,
        retention_campaign_cost_per_customer: float = 15.0,
        retention_campaign_success_rate: float = 0.65,
        initial_churn_rate: float = 0.18,
        model_precision: Optional[float] = None,
        model_recall: Optional[float] = None
    ):
        self.project_cost = project_cost
        self.avg_customer_value = avg_customer_value
        self.retention_cost = retention_campaign_cost_per_customer
        self.retention_success_rate = retention_campaign_success_rate
        self.initial_churn_rate = initial_churn_rate
        self.model_precision = model_precision
        self.model_recall = model_recall
        
    def calculate_roi(
        self,
        df_predictions: pd.DataFrame,
        total_customers: int,
        prediction_threshold: float = 0.7,
        time_horizon_months: int = 12
    ) -> Dict:
        """Calcule le ROI complet du projet."""
        at_risk_customers = len(df_predictions[
            df_predictions['churn_proba'] >= prediction_threshold
        ])
        
        if 'churn' in df_predictions.columns:
            actual_churners = df_predictions['churn'].sum()
        else:
            avg_churn_proba = df_predictions['churn_proba'].mean()
            actual_churners = int(total_customers * avg_churn_proba)
        
        targeted_customers = at_risk_customers
        saved_customers = int(targeted_customers * self.retention_success_rate)
        revenue_saved = saved_customers * self.avg_customer_value
        campaign_cost = targeted_customers * self.retention_cost
        net_benefit = revenue_saved - campaign_cost - self.project_cost
        roi_percentage = (net_benefit / self.project_cost) * 100 if self.project_cost > 0 else 0
        
        new_churn_rate = self.initial_churn_rate * (1 - (saved_customers / total_customers))
        churn_reduction = self.initial_churn_rate - new_churn_rate
        churn_reduction_percentage = (churn_reduction / self.initial_churn_rate) * 100
        campaign_roi = revenue_saved / campaign_cost if campaign_cost > 0 else 0
        
        monthly_benefit = (revenue_saved - campaign_cost) / time_horizon_months
        payback_period = self.project_cost / monthly_benefit if monthly_benefit > 0 else float('inf')
        
        return {
            'project_cost': self.project_cost,
            'total_customers': total_customers,
            'at_risk_customers': at_risk_customers,
            'targeted_customers': targeted_customers,
            'saved_customers': saved_customers,
            'revenue_saved': revenue_saved,
            'campaign_cost': campaign_cost,
            'net_benefit': net_benefit,
            'roi_percentage': roi_percentage,
            'initial_churn_rate': self.initial_churn_rate,
            'new_churn_rate': new_churn_rate,
            'churn_reduction': churn_reduction,
            'churn_reduction_percentage': churn_reduction_percentage,
            'campaign_roi': campaign_roi,
            'payback_period_months': payback_period,
            'time_horizon_months': time_horizon_months
        }
    
    def calculate_scenarios(self, df_predictions: pd.DataFrame, total_customers: int, prediction_threshold: float = 0.7) -> Dict[str, Dict]:
        """Calcule différents scénarios (optimiste, réaliste, pessimiste)."""
        scenarios = {}
        for name, success_rate in [('optimistic', 0.80), ('realistic', 0.65), ('pessimistic', 0.50)]:
            calc = ROICalculator(
                project_cost=self.project_cost,
                avg_customer_value=self.avg_customer_value,
                retention_campaign_cost_per_customer=self.retention_cost,
                retention_campaign_success_rate=success_rate,
                initial_churn_rate=self.initial_churn_rate
            )
            scenarios[name] = calc.calculate_roi(df_predictions, total_customers, prediction_threshold)
        return scenarios
    
    def calculate_timeline_impact(self, df_predictions: pd.DataFrame, total_customers: int, prediction_threshold: float = 0.7, months: int = 12) -> pd.DataFrame:
        """Calcule l'impact mois par mois."""
        base_metrics = self.calculate_roi(df_predictions, total_customers, prediction_threshold, time_horizon_months=months)
        timeline_data = []
        cumulative_revenue = 0
        cumulative_cost = self.project_cost
        
        for month in range(1, months + 1):
            progress_factor = min(1.0, month / 6)
            monthly_revenue_saved = (base_metrics['revenue_saved'] / months) * progress_factor
            monthly_campaign_cost = base_metrics['campaign_cost'] / months
            cumulative_revenue += monthly_revenue_saved
            cumulative_cost += monthly_campaign_cost
            net_benefit_month = cumulative_revenue - cumulative_cost
            churn_rate_month = self.initial_churn_rate - (base_metrics['churn_reduction'] * progress_factor)
            timeline_data.append({
                'month': month,
                'month_label': f'M{month}',
                'churn_rate': churn_rate_month,
                'monthly_revenue_saved': monthly_revenue_saved,
                'cumulative_revenue_saved': cumulative_revenue,
                'cumulative_cost': cumulative_cost,
                'net_benefit': net_benefit_month,
                'roi': (net_benefit_month / self.project_cost) * 100 if self.project_cost > 0 else 0
            })
        return pd.DataFrame(timeline_data)
    
    def format_metrics_report(self, metrics: Dict) -> str:
        """Formate les métriques en un rapport lisible."""
        return f"""
╔══════════════════════════════════════════════════════════════╗
║           RAPPORT ROI - PRÉDICTION DE CHURN                  ║
╚══════════════════════════════════════════════════════════════╝

📊 MÉTRIQUES BUSINESS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Clients analysés              : {metrics['total_customers']:,}
  • Clients à risque identifiés   : {metrics['at_risk_customers']:,}
  • Clients ciblés par campagne    : {metrics['targeted_customers']:,}
  • Clients sauvés (estimé)       : {metrics['saved_customers']:,}

💰 IMPACT FINANCIER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Coût du projet                 : €{metrics['project_cost']:,.0f}
  • Coût des campagnes             : €{metrics['campaign_cost']:,.0f}
  • Revenus sauvés                 : €{metrics['revenue_saved']:,.0f}
  • Bénéfice net                   : €{metrics['net_benefit']:,.0f}

📈 INDICATEURS DE PERFORMANCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • ROI                            : {metrics['roi_percentage']:.1f}%
  • ROI campagne                   : {metrics['campaign_roi']:.2f}x
  • Période de retour              : {metrics['payback_period_months']:.1f} mois
  • Réduction du churn             : {metrics['churn_reduction_percentage']:.1f}%
  • Taux de churn initial          : {metrics['initial_churn_rate']*100:.1f}%
  • Taux de churn après            : {metrics['new_churn_rate']*100:.1f}%

⏱️  HORIZON TEMPOREL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Période d'analyse              : {metrics['time_horizon_months']} mois

"""
