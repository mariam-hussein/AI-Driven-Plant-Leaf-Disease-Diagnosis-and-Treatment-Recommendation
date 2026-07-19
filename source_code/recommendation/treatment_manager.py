import json
import re
from typing import List, Dict, Any, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TreatmentManager:
    def __init__(self, treatments_file: str, top_k: int = 2):
        self.top_k = top_k

        # Load JSON
        with open(treatments_file, "r", encoding="utf-8") as f:
            self.treatments: Dict[str, Dict[str, List[Dict[str, Any]]]] = json.load(f)

        # Profile index
        self.profile_vectorizer = None
        self.profile_matrix = None
        self.profile_keys: List[Tuple[str, str]] = []

        # Lookup table: (disease, stage) -> list of treatments
        self.lookup_table: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

        self._build_profiles()

    # -----------------------------------------------------
    # text normalization
    # -----------------------------------------------------
    def _clean(self, text: str) -> str:
        if not text:
            return ""

        text = str(text).lower()
        text = text.replace("___", " ")
        text = re.sub(r"[\_\(\)\-\/\,\.]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # -----------------------------------------------------
    # extract disease name from class name
    # Example:
    # Apple___Apple_scab -> Apple_scab
    # Tomato___Late_blight -> Late_blight
    # -----------------------------------------------------
    def _extract_disease_name(self, predicted_class: str) -> str:
        if not predicted_class:
            return ""

        if "___" in predicted_class:
            return predicted_class.split("___", 1)[1].strip()

        return predicted_class.strip()

    # -----------------------------------------------------
    # build disease-stage profiles
    # JSON expected format:
    # {
    #   "Apple Scab": {
    #       "EARLY": [ {...}, {...} ],
    #       "MODERATE": [ ... ],
    #       "SEVERE": [ ... ]
    #   },
    #   ...
    # }
    # -----------------------------------------------------
    def _build_profiles(self):
        corpus = []

        for disease, stages in self.treatments.items():
            if not isinstance(stages, dict):
                continue

            for stage, treatments in stages.items():
                if not isinstance(treatments, list):
                    continue

                profile_text = self._clean(f"{disease} {stage}")

                corpus.append(profile_text)
                self.profile_keys.append((disease, stage))
                self.lookup_table[(disease, stage)] = treatments

        if not corpus:
            raise ValueError("No valid treatment profiles found in the JSON file.")

        self.profile_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.profile_matrix = self.profile_vectorizer.fit_transform(corpus)

        print(f"[TreatmentManager] Built {len(self.profile_keys)} disease-stage profiles")

    # -----------------------------------------------------
    # find closest disease-stage profile
    # input:
    #   disease_or_class: may be disease only or full predicted_class
    #   severity: EARLY / MODERATE / SEVERE
    # -----------------------------------------------------
    def _find_closest_profile(self, disease_or_class: str, severity: str) -> Tuple[str, str]:
        disease_name = self._extract_disease_name(disease_or_class)

        query = self._clean(f"{disease_name} {severity}")
        q_vec = self.profile_vectorizer.transform([query])

        sims = cosine_similarity(q_vec, self.profile_matrix).flatten()
        best_idx = sims.argmax()

        best_match = self.profile_keys[best_idx]
        best_score = float(sims[best_idx])

        print(f"[Recommender] Query='{disease_name} + {severity}'")
        print(f"[Recommender] Matched with: {best_match} | score={best_score:.3f}")

        return best_match

    # -----------------------------------------------------
    # sort by rank
    # -----------------------------------------------------
    def _sort_by_rank(self, treatments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        def rank_key(t):
            if isinstance(t, dict):
                r = t.get("rank", "999")
                return int(r) if str(r).isdigit() else 999
            return 999

        return sorted(treatments, key=rank_key)

    # -----------------------------------------------------
    # public API
    # input may be:
    #   predicted_class = "Apple___Apple_scab"
    #   or disease_name = "Apple_scab"
    # -----------------------------------------------------
    def get_treatments(self, predicted_class: str, severity: str) -> List[Dict[str, Any]]:
        if not predicted_class or not severity:
            return []

        severity_clean = severity.strip().upper()

        # Healthy case -> no treatment
        if severity_clean == "HEALTHY":
            print("[Recommender] Healthy case detected -> no treatment returned")
            return []

        # Step 1: find closest disease-stage
        disease, stage = self._find_closest_profile(predicted_class, severity_clean)

        # Step 2: get treatments
        treatments = self.lookup_table.get((disease, stage), [])
        treatments = self._sort_by_rank(treatments)

        return treatments[:self.top_k] if self.top_k else treatments

    # -----------------------------------------------------
    # optional formatted output for GUI
    # -----------------------------------------------------
    def get_formatted_treatments(self, predicted_class: str, severity: str) -> Dict[str, Any]:
        treatments = self.get_treatments(predicted_class, severity)

        option_1 = self._format_option(treatments[0]) if len(treatments) > 0 else None
        option_2 = self._format_option(treatments[1]) if len(treatments) > 1 else None

        return {
            "option_1": option_1,
            "option_2": option_2
        }

    # -----------------------------------------------------
    # format one treatment option
    # -----------------------------------------------------
    def _format_option(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "name": item.get("treatment_name", "N/A"),
            "type": item.get("treatment_type", "N/A"),
            "mode_of_action": item.get("mode_of_action", "N/A"),
            "details": item.get("description", "N/A"),
            "precaution": item.get("precautions", "N/A"),
            "rank": item.get("rank", "N/A")
        }


# =========================================================
# Example usage
# =========================================================
if __name__ == "__main__":
    TREATMENTS_JSON_PATH = r"C:\Users\DELL-MT\Desktop\prototype_gui\workflow_4_6\treatments2.json"

    manager = TreatmentManager(
        treatments_file=TREATMENTS_JSON_PATH,
        top_k=2
    )

    # Example 1: full class name
    predicted_class = "Apple___Apple_scab"
    severity = "EARLY"

    treatments = manager.get_treatments(predicted_class, severity)

    print("\n=== RAW TREATMENTS ===")
    for i, t in enumerate(treatments, 1):
        print(f"\nOption {i}")
        print(json.dumps(t, indent=4, ensure_ascii=False))

    # Example 2: formatted for GUI
    formatted = manager.get_formatted_treatments(predicted_class, severity)

    print("\n=== FORMATTED FOR GUI ===")
    print(json.dumps(formatted, indent=4, ensure_ascii=False))