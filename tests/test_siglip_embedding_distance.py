import csv
import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "evaluation_tool" / "eval_siglip_embedding_distance.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("test_siglip_embedding_distance_module", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"test")
    return path


def test_collect_image_paths_filters_directory_scan(tmp_path):
    module = _load_module()
    orig = _touch(tmp_path / "orig.png")
    variant = _touch(tmp_path / "variant_00.png")
    _touch(tmp_path / "contact_sheet.png")
    _touch(tmp_path / "compare_triptych.png")
    _touch(tmp_path / "notes.txt")

    paths = module.collect_image_paths(
        input_path=str(tmp_path),
        include_ext=".png,.jpg",
        exclude_substrings="contact_sheet,compare,triptych",
    )

    assert paths == [orig.resolve(), variant.resolve()]


def test_choose_reference_image_prefers_orig(tmp_path):
    module = _load_module()
    variant = _touch(tmp_path / "variant_00.png").resolve()
    orig = _touch(tmp_path / "orig.png").resolve()
    other = _touch(tmp_path / "variant_01.png").resolve()

    reference = module.choose_reference_image([variant, other, orig])

    assert reference == orig


def test_choose_reference_image_honors_explicit_reference(tmp_path):
    module = _load_module()
    orig = _touch(tmp_path / "orig.png").resolve()
    reference = _touch(tmp_path / "manual_ref.png").resolve()

    chosen = module.choose_reference_image([orig], reference_image=str(reference))

    assert chosen == reference


def test_compute_reference_distances_uses_one_minus_cosine(tmp_path):
    module = _load_module()
    ref = _touch(tmp_path / "orig.png").resolve()
    same = _touch(tmp_path / "same.png").resolve()
    ortho = _touch(tmp_path / "ortho.png").resolve()

    rows = module.compute_reference_distances(
        image_paths=[ref, same, ortho],
        reference_path=ref,
        embeddings={
            ref: [1.0, 0.0],
            same: [1.0, 0.0],
            ortho: [0.0, 1.0],
        },
    )

    assert len(rows) == 2
    assert rows[0]["image_path"] == str(same)
    assert rows[0]["cosine_similarity"] == 1.0
    assert rows[0]["siglip_distance"] == 0.0
    assert rows[1]["image_path"] == str(ortho)
    assert rows[1]["cosine_similarity"] == 0.0
    assert rows[1]["siglip_distance"] == 1.0


def test_summarize_reference_distances_reports_mean_min_max(tmp_path):
    module = _load_module()
    ref = _touch(tmp_path / "orig.png").resolve()
    rows = [
        {"reference_path": str(ref), "image_path": "a.png", "cosine_similarity": 0.9, "siglip_distance": 0.1},
        {"reference_path": str(ref), "image_path": "b.png", "cosine_similarity": 0.5, "siglip_distance": 0.5},
        {"reference_path": str(ref), "image_path": "c.png", "cosine_similarity": 0.2, "siglip_distance": 0.8},
    ]

    summary = module.summarize_reference_distances(reference_path=ref, distance_rows=rows)

    assert summary["reference_path"] == str(ref)
    assert summary["image_count"] == 4
    assert abs(summary["mean_distance"] - (1.4 / 3.0)) < 1e-9
    assert summary["min_distance"] == 0.1
    assert summary["max_distance"] == 0.8


def test_build_distinguishable_labels_adds_parent_dirs_for_duplicate_names(tmp_path):
    module = _load_module()
    path_a = _touch(tmp_path / "set_a" / "variant_00.png").resolve()
    path_b = _touch(tmp_path / "set_b" / "variant_00.png").resolve()
    path_c = _touch(tmp_path / "set_b" / "variant_01.png").resolve()

    labels = module.build_distinguishable_labels([path_a, path_b, path_c])

    assert labels[path_a].endswith("set_a/variant_00.png")
    assert labels[path_b].endswith("set_b/variant_00.png")
    assert labels[path_a] != labels[path_b]
    assert labels[path_c] == "variant_01.png"


def test_run_evaluation_with_fake_embedder_writes_two_image_outputs(tmp_path, monkeypatch):
    module = _load_module()
    orig = _touch(tmp_path / "orig.png").resolve()
    variant = _touch(tmp_path / "variant_00.png").resolve()
    output_dir = tmp_path / "out"

    def fake_load_siglip_model(model_name, device):
        return object(), "cpu"

    def fake_compute_embeddings(image_paths, model, device, batch_size):
        assert image_paths == [orig, variant]
        assert batch_size == 4
        return {
            orig: [1.0, 0.0],
            variant: [0.6, 0.8],
        }

    monkeypatch.setattr(module, "load_siglip_model", fake_load_siglip_model)
    monkeypatch.setattr(module, "compute_embeddings", fake_compute_embeddings)
    monkeypatch.setattr(
        module,
        "create_summary_visualization",
        lambda reference_path, distance_rows, summary_row, output_path, model_name: _touch(output_path),
    )

    result = module.run_evaluation(
        input_path=str(tmp_path),
        batch_size=4,
        output_dir=str(output_dir),
    )

    reference_distances_path = result["reference_distances_path"]
    summary_path = result["summary_path"]
    visualization_path = result["visualization_path"]
    assert reference_distances_path.exists()
    assert summary_path.exists()
    assert visualization_path.exists()

    with reference_distances_path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1
    assert rows[0]["reference_path"] == str(orig)
    assert rows[0]["image_path"] == str(variant)
    assert abs(float(rows[0]["cosine_similarity"]) - 0.6) < 1e-9
    assert abs(float(rows[0]["siglip_distance"]) - 0.4) < 1e-9

    with summary_path.open(newline="") as file:
        summary_rows = list(csv.DictReader(file))
    assert len(summary_rows) == 1
    assert summary_rows[0]["reference_path"] == str(orig)
    assert int(summary_rows[0]["image_count"]) == 2
    assert abs(float(summary_rows[0]["mean_distance"]) - 0.4) < 1e-9
    assert abs(float(summary_rows[0]["min_distance"]) - 0.4) < 1e-9
    assert abs(float(summary_rows[0]["max_distance"]) - 0.4) < 1e-9


def test_run_evaluation_summary_values_for_three_images(tmp_path, monkeypatch):
    module = _load_module()
    orig = _touch(tmp_path / "orig.png").resolve()
    variant_a = _touch(tmp_path / "variant_00.png").resolve()
    variant_b = _touch(tmp_path / "variant_01.png").resolve()
    output_dir = tmp_path / "summary_out"

    monkeypatch.setattr(module, "load_siglip_model", lambda model_name, device: (object(), "cpu"))
    monkeypatch.setattr(
        module,
        "compute_embeddings",
        lambda image_paths, model, device, batch_size: {
            orig: [1.0, 0.0],
            variant_a: [0.8, 0.6],
            variant_b: [0.0, 1.0],
        },
    )
    monkeypatch.setattr(
        module,
        "create_summary_visualization",
        lambda reference_path, distance_rows, summary_row, output_path, model_name: _touch(output_path),
    )

    result = module.run_evaluation(
        input_path=str(tmp_path),
        output_dir=str(output_dir),
    )

    summary_row = result["summary_row"]
    assert result["visualization_path"].exists()
    assert summary_row["image_count"] == 3
    assert abs(summary_row["mean_distance"] - 0.6) < 1e-9
    assert abs(summary_row["min_distance"] - 0.2) < 1e-9
    assert abs(summary_row["max_distance"] - 1.0) < 1e-9
