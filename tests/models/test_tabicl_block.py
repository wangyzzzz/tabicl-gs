import numpy as np

from tabicl_gs.models.tabicl import TabICLVectorMetadata, average_embedding_batches


def test_average_embedding_batches_reduces_ensemble_axis():
    embeddings = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    reduced = average_embedding_batches(embeddings)
    assert reduced.shape == (3, 4)
    assert np.allclose(reduced, embeddings.mean(axis=0))


def test_tabicl_vector_metadata_supports_diagnostic_fields():
    meta = TabICLVectorMetadata(
        raw_embedding_dim=512,
        reduced_embedding_dim=16,
        device="cuda",
        explained_variance_ratio_sum=0.96,
        explained_variance_curve=[0.5, 0.7, 0.9],
    )
    assert meta.explained_variance_curve[-1] == 0.9
