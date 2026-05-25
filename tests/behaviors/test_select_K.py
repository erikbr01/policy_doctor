import unittest

from policy_doctor.behaviors.select_K import (
    select_K_by_silhouette_gamma,
    select_clustering_by_silhouette,
)


class TestSelectKBySilhouetteGamma(unittest.TestCase):
    def test_picks_smallest_k_on_ascending_side(self):
        Ks = [5, 8, 10, 12, 15, 18]
        sils = [0.40, 0.55, 0.60, 0.62, 0.61, 0.50]
        # peak at K=12 (0.62); γ=0.9 → target 0.558; K=10 qualifies (0.60)
        self.assertEqual(
            select_K_by_silhouette_gamma(Ks, sils, gamma=0.9, K_min=5),
            10,
        )

    def test_monotone_decreasing_returns_none(self):
        Ks = [5, 8, 10, 12]
        sils = [0.62, 0.55, 0.50, 0.45]
        self.assertIsNone(select_K_by_silhouette_gamma(Ks, sils, gamma=0.9))

    def test_high_gamma_can_return_none(self):
        Ks = [5, 8, 10, 12]
        sils = [0.40, 0.55, 0.60, 0.62]
        self.assertIsNone(select_K_by_silhouette_gamma(Ks, sils, gamma=1.0))


class TestSelectClusteringBySilhouette(unittest.TestCase):
    def _entry(self, rep, ordering, w, s, k, sil):
        return {
            "rep": rep,
            "ordering": ordering,
            "w": w,
            "s": s,
            "k": k,
            "metrics": {"silhouette_mean": sil},
        }

    def test_picks_setting_with_best_gamma_selected_k(self):
        entries = [
            self._entry("a", "umap_first", 5, 1, k, sil)
            for k, sil in [(5, 0.30), (8, 0.50), (10, 0.55), (12, 0.60), (15, 0.58)]
        ] + [
            self._entry("b", "umap_first", 3, 1, k, sil)
            for k, sil in [(5, 0.35), (8, 0.45), (10, 0.48), (12, 0.50), (15, 0.49)]
        ]
        pick = select_clustering_by_silhouette(entries, gamma=0.9)
        self.assertEqual(pick["rep"], "a")
        self.assertEqual(pick["w"], 5)
        self.assertEqual(pick["s"], 1)
        self.assertEqual(pick["k"], 10)

    def test_falls_back_to_max_silhouette_when_no_gamma_pick(self):
        entries = [
            self._entry("a", "umap_first", 5, 1, 8, 0.70),
            self._entry("b", "umap_first", 3, 1, 6, 0.80),
        ]
        pick = select_clustering_by_silhouette(entries, min_k_per_setting=3)
        self.assertEqual(pick["rep"], "b")
        self.assertEqual(pick["k"], 6)


if __name__ == "__main__":
    unittest.main()
