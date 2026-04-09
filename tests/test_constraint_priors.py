import pytest
import torch
from src.models.constraint_priors import (
    get_constraint_matrix,
    get_gr_neutro_constraints,
    get_aml_matek_constraints,
    get_bmc_constraints,
)


class TestGRNeutro:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.C = get_gr_neutro_constraints()

    def test_shape(self):
        assert self.C.shape == (7, 7)

    def test_symmetry(self):
        assert torch.allclose(self.C, self.C.T)

    def test_diagonal_zero(self):
        assert torch.allclose(self.C.diag(), torch.zeros(7))

    def test_values_in_range(self):
        assert self.C.min() >= -1.0
        assert self.C.max() <= 1.0

    def test_normal_vs_abnormalities(self):
        """Normal (idx 0) must be mutually exclusive with all abnormalities."""
        for i in range(1, 7):
            assert self.C[0, i].item() == -1.0, f"C[0, {i}] should be -1"
            assert self.C[i, 0].item() == -1.0, f"C[{i}, 0] should be -1"

    def test_hypergran_hypogran_exclusive(self):
        assert self.C[3, 5].item() == -1.0
        assert self.C[5, 3].item() == -1.0

    def test_hyperseg_hyposeg_exclusive(self):
        assert self.C[4, 6].item() == -1.0
        assert self.C[6, 4].item() == -1.0

    def test_chromatin_dohle_cooccurrence(self):
        assert self.C[1, 2].item() == pytest.approx(0.3)

    def test_chromatin_hypergran_cooccurrence(self):
        assert self.C[1, 3].item() == pytest.approx(0.4)


class TestAMLMatek:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.C = get_aml_matek_constraints()

    def test_shape(self):
        assert self.C.shape == (15, 15)

    def test_symmetry(self):
        assert torch.allclose(self.C, self.C.T)

    def test_diagonal_zero(self):
        assert torch.allclose(self.C.diag(), torch.zeros(15))

    def test_values_in_range(self):
        assert self.C.min() >= -1.0
        assert self.C.max() <= 1.0

    def test_maturation_cooccurrence(self):
        """Adjacent maturation stages should have positive co-occurrence."""
        assert self.C[0, 1].item() > 0  # Myeloblast-Promyelocyte
        assert self.C[1, 2].item() > 0  # Promyelocyte-Myelocyte
        assert self.C[4, 5].item() > 0  # Band-Segmented


class TestBMC:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.C = get_bmc_constraints()

    def test_shape(self):
        assert self.C.shape == (21, 21)

    def test_symmetry(self):
        assert torch.allclose(self.C, self.C.T)

    def test_diagonal_zero(self):
        assert torch.allclose(self.C.diag(), torch.zeros(21))

    def test_values_in_range(self):
        assert self.C.min() >= -1.0
        assert self.C.max() <= 1.0

    def test_erythroid_maturation(self):
        """Erythroid lineage adjacents should co-occur."""
        assert self.C[13, 14].item() > 0
        assert self.C[14, 15].item() > 0
        assert self.C[15, 16].item() > 0
        assert self.C[16, 17].item() > 0


class TestAMLMatekDetailed:
    """More specific checks for AML Matek constraint values."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.C = get_aml_matek_constraints()

    def test_myeloid_lymphoid_exclusivity(self):
        """Myeloid cells (0-5) mutually exclusive with Lymphocyte (9)."""
        for i in range(6):
            assert self.C[i, 9].item() == -1.0, f"C[{i}, 9] should be -1"

    def test_late_myeloid_early_exclusivity(self):
        """Late myeloid stages exclude early ones."""
        # Metamyelocyte (3) excludes Myeloblast (0)
        assert self.C[3, 0].item() == -1.0
        # Band (4) excludes Promyelocyte (1)
        assert self.C[4, 1].item() == -1.0
        # Segmented (5) excludes Myelocyte (2) and Metamyelocyte (3)
        assert self.C[5, 2].item() == -1.0
        assert self.C[5, 3].item() == -1.0

    def test_eosinophil_myeloblast_exclusive(self):
        assert self.C[6, 0].item() == -1.0

    def test_lymphocyte_plasma_cooccurrence(self):
        assert self.C[9, 10].item() == pytest.approx(0.2)

    def test_erythroblast_rbc_cooccurrence(self):
        assert self.C[11, 12].item() == pytest.approx(0.3)


class TestBMCDetailed:
    """More specific checks for BMC constraint values."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.C = get_bmc_constraints()

    def test_myeloid_lineage_cooccurrence(self):
        """Myeloid maturation chain should all be 0.3."""
        assert self.C[0, 1].item() == pytest.approx(0.3)
        assert self.C[1, 2].item() == pytest.approx(0.3)
        assert self.C[2, 3].item() == pytest.approx(0.3)
        assert self.C[3, 4].item() == pytest.approx(0.3)
        assert self.C[4, 5].item() == pytest.approx(0.3)

    def test_band_segmented_exclude_erythroid(self):
        """Band (4) and Segmented (5) exclude erythroid stages (13-17)."""
        for i in [4, 5]:
            for j in range(13, 18):
                assert self.C[i, j].item() == -1.0, f"C[{i},{j}] should be -1"

    def test_artifact_smudge_other_cooccurrence(self):
        assert self.C[18, 19].item() == pytest.approx(0.2)
        assert self.C[18, 20].item() == pytest.approx(0.2)
        assert self.C[19, 20].item() == pytest.approx(0.2)

    def test_smudge_blast_weak(self):
        assert self.C[19, 0].item() == pytest.approx(0.15)
        assert self.C[19, 9].item() == pytest.approx(0.15)


class TestSaveConstraintMatrix:
    def test_save_and_reload(self, tmp_path):
        from src.models.constraint_priors import save_constraint_matrix
        out_path = str(tmp_path / 'test_constraints.pt')
        save_constraint_matrix('gr_neutro', out_path)
        loaded = torch.load(out_path, weights_only=True)
        expected = get_constraint_matrix('gr_neutro')
        assert torch.equal(loaded, expected)

    def test_save_all_datasets(self, tmp_path):
        from src.models.constraint_priors import save_constraint_matrix
        for name, expected_shape in [('gr_neutro', (7, 7)), ('aml_matek', (15, 15)), ('bmc', (21, 21))]:
            out_path = str(tmp_path / f'{name}.pt')
            save_constraint_matrix(name, out_path)
            loaded = torch.load(out_path, weights_only=True)
            assert loaded.shape == expected_shape


class TestFactory:
    def test_gr_neutro(self):
        C = get_constraint_matrix('gr_neutro')
        assert C.shape == (7, 7)

    def test_aml_matek(self):
        C = get_constraint_matrix('aml_matek')
        assert C.shape == (15, 15)

    def test_bmc(self):
        C = get_constraint_matrix('bmc')
        assert C.shape == (21, 21)

    def test_case_insensitive(self):
        C = get_constraint_matrix('GR_NEUTRO')
        assert C.shape == (7, 7)

    def test_mixed_case(self):
        C = get_constraint_matrix('Gr_Neutro')
        assert C.shape == (7, 7)

    def test_unknown_dataset(self):
        with pytest.raises(ValueError):
            get_constraint_matrix('unknown')
