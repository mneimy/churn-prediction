"""
Tests de la couche conformité.

Ces tests ne vérifient pas que le code s'exécute : ils vérifient que les
règles juridiques sont effectivement appliquées. Un test qui passerait
encore si l'on retirait le filtre d'opposition ne sert à rien.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.compliance.audit import AuditLog
from src.compliance.consent import (
    CONSENT_COLUMNS,
    ConsentPolicy,
    ConsentRegistry,
    Purpose,
)
from src.compliance.privacy import (
    MissingPseudonymKeyError,
    check_k_anonymity,
    get_pseudonym_key,
    minimize,
    pseudonymize_id,
)
from src.compliance.retention import RetentionPolicy, apply_retention, erase_customer

NOW = datetime(2026, 1, 1)


def _consent_row(customer_id, purpose, **overrides):
    row = {
        "customer_id": customer_id,
        "purpose": purpose.value,
        "opt_in": False,
        "consent_date": pd.NaT,
        "consent_source": "test",
        "consent_wording_version": "v1",
        "withdrawn_at": pd.NaT,
        "objection_at": pd.NaT,
    }
    row.update(overrides)
    return row


@pytest.fixture
def registry() -> ConsentRegistry:
    rows = [
        # Consentement valide et récent.
        _consent_row("C1", Purpose.MARKETING_EMAIL, opt_in=True,
                     consent_date=NOW - timedelta(days=30)),
        # Consentement retiré.
        _consent_row("C2", Purpose.MARKETING_EMAIL, opt_in=True,
                     consent_date=NOW - timedelta(days=200),
                     withdrawn_at=NOW - timedelta(days=10)),
        # Consentement périmé (> 25 mois).
        _consent_row("C3", Purpose.MARKETING_EMAIL, opt_in=True,
                     consent_date=NOW - timedelta(days=900)),
        # Jamais consenti.
        _consent_row("C4", Purpose.MARKETING_EMAIL),
        # Opposition au scoring.
        _consent_row("C5", Purpose.CHURN_SCORING,
                     objection_at=NOW - timedelta(days=5)),
        _consent_row("C5", Purpose.MARKETING_EMAIL, opt_in=True,
                     consent_date=NOW - timedelta(days=20)),
    ]
    return ConsentRegistry(pd.DataFrame(rows, columns=CONSENT_COLUMNS))


class TestConsent:
    def test_opt_in_valide_est_retenu(self, registry):
        assert "C1" in registry.valid_opt_ins(Purpose.MARKETING_EMAIL, NOW)

    def test_consentement_retire_est_exclu(self, registry):
        assert "C2" not in registry.valid_opt_ins(Purpose.MARKETING_EMAIL, NOW)

    def test_consentement_perime_est_exclu(self, registry):
        """Au-delà de 25 mois sans manifestation, le consentement n'est plus éclairé."""
        assert "C3" not in registry.valid_opt_ins(Purpose.MARKETING_EMAIL, NOW)

    def test_absence_de_consentement_exclut(self, registry):
        assert "C4" not in registry.valid_opt_ins(Purpose.MARKETING_EMAIL, NOW)

    def test_scoring_inclut_tout_sauf_les_opposants(self, registry):
        """Intérêt légitime : tout le monde, sauf opposition (art. 21)."""
        eligible = registry.eligible_for_scoring(["C1", "C2", "C4", "C5"], NOW)
        assert eligible == {"C1", "C2", "C4"}
        assert "C5" not in eligible

    def test_un_score_ne_donne_pas_le_droit_de_contacter(self, registry):
        """
        La règle centrale du projet : scorer n'est pas contacter.
        C4 est scorable (pas d'opposition) mais pas contactable (pas d'opt-in).
        """
        assert "C4" in registry.eligible_for_scoring(["C4"], NOW)
        assert "C4" not in registry.eligible_for_targeting(
            ["C4"], Purpose.MARKETING_EMAIL, NOW
        )

    def test_ciblage_refuse_une_finalite_sans_consentement(self, registry):
        with pytest.raises(ValueError, match="ne repose pas sur le consentement"):
            registry.eligible_for_targeting(["C1"], Purpose.CHURN_SCORING, NOW)

    def test_opposition_enregistree_prend_effet(self, registry):
        assert "C1" in registry.eligible_for_scoring(["C1"], NOW)
        registry.record_objection("C1", Purpose.CHURN_SCORING, NOW - timedelta(days=1))
        assert "C1" not in registry.eligible_for_scoring(["C1"], NOW)

    def test_retrait_prend_effet(self, registry):
        assert "C1" in registry.valid_opt_ins(Purpose.MARKETING_EMAIL, NOW)
        registry.record_withdrawal("C1", Purpose.MARKETING_EMAIL, NOW - timedelta(days=1))
        assert "C1" not in registry.valid_opt_ins(Purpose.MARKETING_EMAIL, NOW)

    def test_consentement_anterieur_a_la_date_seulement(self, registry):
        """Un consentement futur ne vaut pas consentement aujourd'hui."""
        past = NOW - timedelta(days=60)
        assert "C1" not in registry.valid_opt_ins(Purpose.MARKETING_EMAIL, past)

    def test_version_de_mentions_exigee(self):
        rows = [_consent_row("C1", Purpose.MARKETING_EMAIL, opt_in=True,
                             consent_date=NOW - timedelta(days=10),
                             consent_wording_version="v0")]
        reg = ConsentRegistry(
            pd.DataFrame(rows, columns=CONSENT_COLUMNS),
            policy=ConsentPolicy(required_wording_version="v1"),
        )
        assert reg.valid_opt_ins(Purpose.MARKETING_EMAIL, NOW) == set()

    def test_referentiel_incomplet_refuse(self):
        with pytest.raises(ValueError, match="colonnes manquantes"):
            ConsentRegistry(pd.DataFrame({"customer_id": ["C1"]}))

    def test_filtre_dataframe_produit_une_trace(self, registry):
        df = pd.DataFrame({"customer_id": ["C1", "C5"], "v": [1, 2]})
        out, trace = registry.filter_dataframe(df, Purpose.CHURN_SCORING, NOW)
        assert list(out["customer_id"]) == ["C1"]
        assert trace["customers_excluded"] == 1
        assert trace["legal_basis"] == "interet_legitime"

    def test_fuseau_horaire_indifferent(self, registry):
        """Airflow fournit des dates aware, les fichiers métier des dates naive."""
        aware = pd.Timestamp(NOW, tz="UTC")
        assert registry.valid_opt_ins(
            Purpose.MARKETING_EMAIL, aware
        ) == registry.valid_opt_ins(Purpose.MARKETING_EMAIL, NOW)


class TestPrivacy:
    def test_pseudonyme_stable_avec_la_meme_cle(self):
        key = b"cle-de-test"
        assert pseudonymize_id("CUST_1", key) == pseudonymize_id("CUST_1", key)

    def test_pseudonyme_differe_selon_la_cle(self):
        """Sans cela, le pseudonyme serait cassable par table arc-en-ciel."""
        assert pseudonymize_id("CUST_1", b"a") != pseudonymize_id("CUST_1", b"b")

    def test_cle_absente_leve_plutot_que_de_degrader(self, monkeypatch):
        monkeypatch.delenv("CHURN_PSEUDONYM_KEY", raising=False)
        with pytest.raises(MissingPseudonymKeyError):
            get_pseudonym_key()

    def test_cle_ephemere_autorisee_explicitement(self, monkeypatch):
        monkeypatch.delenv("CHURN_PSEUDONYM_KEY", raising=False)
        assert len(get_pseudonym_key(allow_ephemeral=True)) == 32

    def test_identifiants_directs_supprimes(self):
        df = pd.DataFrame(
            {"customer_id": ["C1"], "email": ["a@b.c"], "phone": ["06"], "total": [10]}
        )
        out, report = minimize(df, pseudonymize=False)
        assert "email" not in out.columns and "phone" not in out.columns
        assert set(report.dropped_identifiers) == {"email", "phone"}
        assert "total" in out.columns

    def test_donnees_sensibles_bloquent_le_pipeline(self):
        """Les catégories art. 9 sont un incident, pas une option."""
        df = pd.DataFrame({"customer_id": ["C1"], "health": ["x"]})
        with pytest.raises(ValueError, match="art. 9"):
            minimize(df, pseudonymize=False)

    def test_k_anonymat_detecte_une_singularisation(self):
        df = pd.DataFrame({"ville": ["Paris"] * 9 + ["Guéret"], "age": [30] * 10})
        result = check_k_anonymity(df, ["ville", "age"], k=5)
        assert not result["conforme"]
        assert result["personnes_exposees"] == 1


class TestRetention:
    def test_purge_les_lignes_trop_anciennes(self):
        df = pd.DataFrame(
            {"d": pd.to_datetime(["2020-01-01", "2025-06-01"]), "v": [1, 2]}
        )
        out, trace = apply_retention(df, "d", 1095, datetime(2026, 1, 1), "test")
        assert len(out) == 1 and trace["lignes_purgees"] == 1

    def test_trace_exploitable_pour_le_journal(self):
        df = pd.DataFrame({"d": pd.to_datetime(["2020-01-01"]), "v": [1]})
        _, trace = apply_retention(df, "d", 1095, datetime(2026, 1, 1), "transactions")
        assert trace["dataset"] == "transactions"
        assert "date_limite" in trace

    def test_colonne_absente_leve(self):
        with pytest.raises(KeyError):
            apply_retention(pd.DataFrame({"x": [1]}), "d", 30, datetime.now())

    def test_effacement_signale_le_besoin_de_reentrainement(self):
        """
        Supprimer des fichiers ne suffit pas : le modèle a mémorisé la personne.
        """
        frames = {
            "tx": pd.DataFrame({"customer_id": ["C1", "C2"], "v": [1, 2]}),
            "features": pd.DataFrame({"customer_id": ["C1"], "f": [3]}),
        }
        cleaned, trace = erase_customer(frames, "C1")
        assert "C1" not in set(cleaned["tx"]["customer_id"])
        assert len(cleaned["features"]) == 0
        assert trace["reentrainement_requis"] is True

    def test_politique_exposee_pour_le_registre(self):
        assert RetentionPolicy().as_dict()["transactions_jours"] == 1095


class TestAuditLog:
    def test_entree_journalisee_et_relue(self, tmp_path):
        log = AuditLog(tmp_path / "audit.jsonl")
        log.record("build", "churn_scoring", {"lignes": 10})
        entries = log.read_all()
        assert len(entries) == 1 and entries[0]["event"] == "build"

    def test_donnee_personnelle_refusee(self, tmp_path):
        """Un journal de conformité ne doit pas devenir un fichier de données."""
        log = AuditLog(tmp_path / "audit.jsonl")
        with pytest.raises(ValueError, match="donnée personnelle"):
            log.record("build", "churn_scoring", {"customer_id": "C1"})

    def test_donnee_personnelle_imbriquee_refusee(self, tmp_path):
        log = AuditLog(tmp_path / "audit.jsonl")
        with pytest.raises(ValueError, match="donnée personnelle"):
            log.record("build", "p", {"detail": {"email": "a@b.c"}})

    def test_chaine_intacte_sur_un_journal_normal(self, tmp_path):
        log = AuditLog(tmp_path / "audit.jsonl")
        for i in range(5):
            log.record("e", "p", {"i": i})
        assert log.verify_integrity()["intact"] is True

    def test_alteration_detectee(self, tmp_path):
        """C'est tout l'intérêt du chaînage : rendre la falsification visible."""
        path = tmp_path / "audit.jsonl"
        log = AuditLog(path)
        for i in range(4):
            log.record("e", "p", {"i": i})

        lines = path.read_text(encoding="utf-8").splitlines()
        del lines[1]  # suppression d'une entrée gênante
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        integrity = log.verify_integrity()
        assert integrity["intact"] is False
        assert integrity["premiere_rupture"] == 1
