from risk_window.matcher import match_reference_prototypes
from risk_window.types import ReferencePrototype


def test_match_reference_prototypes_returns_anchor_near_best_progress():
    prototype = ReferencePrototype(
        task_id="0",
        init_state_idx=1,
        phase_id="contact_manipulate",
        progress_points=[0.0, 0.5, 1.0],
        feature_vectors=[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
    )
    history = [
        [0.1, 0.1],
        [0.9, 0.9],
        [1.1, 1.1],
    ]
    result = match_reference_prototypes(history, [prototype], topk=1, band=2, cosine_weight=0.7)
    assert result is not None
    assert abs(result["anchor_progress"] - 0.5) < 1e-6
    assert result["phase_id"] == "contact_manipulate"
    assert result["score"] > 0.5
