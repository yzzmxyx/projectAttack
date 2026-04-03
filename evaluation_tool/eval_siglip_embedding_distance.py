import argparse
import csv
import math
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_NAME = "google/siglip-so400m-patch14-384"
DEFAULT_INCLUDE_EXT = ".png,.jpg,.jpeg,.webp,.bmp"
DEFAULT_EXCLUDE_SUBSTRINGS = "contact_sheet,compare,triptych"
DEFAULT_REFERENCE_BASENAMES = ("orig.png", "original.png")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "run" / "siglip_metric"
DEFAULT_VISUALIZATION_NAME = "reference_distance_summary.png"


def parse_csv_values(raw_value: Optional[str], lowercase: bool = True) -> List[str]:
    if raw_value is None:
        return []
    values = []
    for item in str(raw_value).split(","):
        text = item.strip()
        if not text:
            continue
        values.append(text.lower() if lowercase else text)
    return values


def parse_include_exts(raw_value: Optional[str]) -> Tuple[str, ...]:
    values = []
    for item in parse_csv_values(raw_value, lowercase=True):
        ext = item if item.startswith(".") else f".{item}"
        values.append(ext)
    return tuple(values)


def normalize_path(path_like: str) -> Path:
    return Path(path_like).expanduser().resolve()


def validate_image_path(path: Path, include_exts: Sequence[str]) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Image path does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Expected an image file, but got: {path}")
    if include_exts and path.suffix.lower() not in include_exts:
        raise ValueError(f"Unsupported image extension for `{path}`. Allowed: {', '.join(include_exts)}")
    return path


def should_exclude_path(path: Path, exclude_substrings: Sequence[str]) -> bool:
    text = path.name.lower()
    return any(token in text for token in exclude_substrings)


def dedupe_paths(paths: Iterable[Path]) -> List[Path]:
    unique = OrderedDict()
    for path in paths:
        unique[path] = None
    return list(unique.keys())


def collect_image_paths(
    input_path: Optional[str] = None,
    image_paths: Optional[Sequence[str]] = None,
    include_ext: Optional[str] = DEFAULT_INCLUDE_EXT,
    exclude_substrings: Optional[str] = DEFAULT_EXCLUDE_SUBSTRINGS,
) -> List[Path]:
    include_exts = parse_include_exts(include_ext)
    exclude_tokens = parse_csv_values(exclude_substrings, lowercase=True)

    explicit_paths = [normalize_path(path) for path in (image_paths or [])]
    if explicit_paths:
        return dedupe_paths(validate_image_path(path, include_exts) for path in explicit_paths)

    if not input_path:
        raise ValueError("Either `input_path` or `image_paths` must be provided.")

    resolved_input = normalize_path(input_path)
    if resolved_input.is_file():
        return [validate_image_path(resolved_input, include_exts)]

    if not resolved_input.exists():
        raise FileNotFoundError(f"Input path does not exist: {resolved_input}")
    if not resolved_input.is_dir():
        raise ValueError(f"Input path must be a file or directory: {resolved_input}")

    candidates = []
    for child in sorted(resolved_input.iterdir()):
        if not child.is_file():
            continue
        if include_exts and child.suffix.lower() not in include_exts:
            continue
        if should_exclude_path(child, exclude_tokens):
            continue
        candidates.append(child.resolve())
    return dedupe_paths(candidates)


def choose_reference_image(image_paths: Sequence[Path], reference_image: Optional[str] = None) -> Path:
    if reference_image:
        return normalize_path(reference_image)

    if not image_paths:
        raise ValueError("No candidate images available to choose a reference image.")

    basename_to_path = {path.name.lower(): path for path in image_paths}
    for basename in DEFAULT_REFERENCE_BASENAMES:
        match = basename_to_path.get(basename)
        if match is not None:
            return match
    return image_paths[0]


def resolve_device(requested_device: str, torch_module) -> object:
    requested_text = str(requested_device).strip().lower()
    if requested_text in ("", "auto"):
        return torch_module.device("cuda:0" if torch_module.cuda.is_available() else "cpu")
    if requested_text.startswith("cuda") and not torch_module.cuda.is_available():
        raise RuntimeError("CUDA device requested, but CUDA is not available in the current environment.")
    return torch_module.device(str(requested_device))


def load_siglip_model(
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "auto",
):
    import torch
    from transformers import SiglipModel

    resolved_device = resolve_device(device, torch)
    model = SiglipModel.from_pretrained(model_name).to(resolved_device)
    model.eval()
    return model, resolved_device


def load_image_tensor(image_path: Path):
    from PIL import Image
    from torchvision import transforms

    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
    to_tensor = transforms.ToTensor()
    return to_tensor(rgb_image)


def compute_embeddings(
    image_paths: Sequence[Path],
    model,
    device,
    batch_size: int = 8,
) -> Dict[Path, List[float]]:
    import torch
    import torch.nn.functional as F
    from torchvision import transforms

    preprocess = transforms.Compose(
        [
            transforms.Resize(384, antialias=True),
            transforms.CenterCrop(384),
        ]
    )

    batch_size = max(1, int(batch_size))
    embeddings: Dict[Path, List[float]] = {}
    pending_paths: List[Path] = []
    pending_tensors: List = []

    def flush_batch():
        if not pending_paths:
            return
        pixel_values = torch.stack(pending_tensors, dim=0).to(device)
        features = model.get_image_features(pixel_values=pixel_values)
        features = F.normalize(features, dim=-1)
        features_list = features.detach().cpu().tolist()
        for path, embedding in zip(pending_paths, features_list):
            embeddings[path] = embedding
        pending_paths.clear()
        pending_tensors.clear()

    with torch.no_grad():
        for image_path in image_paths:
            image_tensor = load_image_tensor(image_path)
            image_tensor = torch.clamp(image_tensor, 0.0, 1.0)
            image_tensor = preprocess(image_tensor)
            pending_paths.append(image_path)
            pending_tensors.append(image_tensor)
            if len(pending_paths) >= batch_size:
                flush_batch()
        flush_batch()

    return embeddings


def cosine_similarity(left_embedding: Sequence[float], right_embedding: Sequence[float]) -> float:
    if len(left_embedding) != len(right_embedding):
        raise ValueError("Embedding vectors must have the same length.")

    numerator = math.fsum(float(a) * float(b) for a, b in zip(left_embedding, right_embedding))
    left_norm = math.sqrt(math.fsum(float(a) * float(a) for a in left_embedding))
    right_norm = math.sqrt(math.fsum(float(b) * float(b) for b in right_embedding))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("Encountered zero-norm embedding while computing cosine similarity.")
    similarity = numerator / (left_norm * right_norm)
    return max(-1.0, min(1.0, similarity))


def lookup_embedding(embeddings: Dict, image_path: Path) -> Sequence[float]:
    if image_path in embeddings:
        return embeddings[image_path]
    text_key = str(image_path)
    if text_key in embeddings:
        return embeddings[text_key]
    raise KeyError(f"Missing embedding for image: {image_path}")


def compute_reference_distances(
    image_paths: Sequence[Path],
    reference_path: Path,
    embeddings: Dict,
) -> List[Dict[str, float]]:
    reference_embedding = lookup_embedding(embeddings, reference_path)
    rows = []
    for image_path in image_paths:
        if image_path == reference_path:
            continue
        image_embedding = lookup_embedding(embeddings, image_path)
        similarity = cosine_similarity(reference_embedding, image_embedding)
        rows.append(
            {
                "reference_path": str(reference_path),
                "image_path": str(image_path),
                "cosine_similarity": similarity,
                "siglip_distance": 1.0 - similarity,
            }
        )
    return rows


def summarize_reference_distances(reference_path: Path, distance_rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not distance_rows:
        raise ValueError("At least one non-reference image is required to summarize distances.")

    distances = [float(row["siglip_distance"]) for row in distance_rows]
    return {
        "reference_path": str(reference_path),
        "image_count": int(len(distance_rows) + 1),
        "mean_distance": float(sum(distances) / len(distances)),
        "min_distance": float(min(distances)),
        "max_distance": float(max(distances)),
    }


def resolve_output_dir(
    output_dir: Optional[str],
    input_path: Optional[str],
    selected_paths: Sequence[Path],
) -> Path:
    if output_dir:
        return normalize_path(output_dir)

    if input_path:
        resolved_input = normalize_path(input_path)
        if resolved_input.exists() and resolved_input.is_dir():
            return resolved_input / "siglip_metric"

    if selected_paths:
        common_parent = selected_paths[0].parent
        if all(path.parent == common_parent for path in selected_paths):
            return common_parent / "siglip_metric"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_ROOT / timestamp


def build_distinguishable_labels(paths: Sequence[Path]) -> Dict[Path, str]:
    normalized_paths = [Path(path) for path in paths]
    unique_paths = dedupe_paths(normalized_paths)
    if not unique_paths:
        return {}

    labels = {path: path.name for path in unique_paths}
    max_depth = max(len(path.parts) for path in unique_paths)

    for depth in range(2, max_depth + 1):
        grouped = {}
        for path, label in labels.items():
            grouped.setdefault(label, []).append(path)

        collisions = [group for group in grouped.values() if len(group) > 1]
        if not collisions:
            break

        for group in collisions:
            for path in group:
                suffix_parts = path.parts[-depth:] if depth <= len(path.parts) else path.parts
                labels[path] = "/".join(suffix_parts)

    return labels


def write_csv(path: Path, rows: Sequence[Dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def create_summary_visualization(
    reference_path: Path,
    distance_rows: Sequence[Dict[str, float]],
    summary_row: Dict[str, float],
    output_path: Path,
    model_name: str,
) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    output_path.parent.mkdir(parents=True, exist_ok=True)

    panel_background = (246, 247, 250)
    card_background = (255, 255, 255)
    canvas_background = (239, 241, 245)
    text_primary = (25, 28, 36)
    text_secondary = (90, 98, 112)
    border_neutral = (206, 212, 224)

    padding = 24
    section_gap = 24
    card_gap = 18
    reference_panel_size = (320, 320)
    compare_panel_size = (220, 220)
    summary_panel_width = 460

    resampling_namespace = getattr(Image, "Resampling", Image)
    resample = resampling_namespace.LANCZOS

    def load_font(font_size: int, bold: bool = False):
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        ]
        for candidate in candidates:
            try:
                return ImageFont.truetype(candidate, font_size)
            except Exception:
                continue
        return ImageFont.load_default()

    title_font = load_font(30, bold=True)
    heading_font = load_font(22, bold=True)
    body_font = load_font(18, bold=False)
    body_bold_font = load_font(18, bold=True)
    small_font = load_font(16, bold=False)

    def text_size(draw_obj, text: str, font) -> Tuple[int, int]:
        left, top, right, bottom = draw_obj.textbbox((0, 0), text, font=font)
        return right - left, bottom - top

    def truncate_text(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[: max(0, max_chars - 3)] + "..."

    def contain_image(image_path: Path, panel_size: Tuple[int, int]) -> Image.Image:
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            rgb.thumbnail(panel_size, resample=resample)
            canvas = Image.new("RGB", panel_size, panel_background)
            offset_x = (panel_size[0] - rgb.width) // 2
            offset_y = (panel_size[1] - rgb.height) // 2
            canvas.paste(rgb, (offset_x, offset_y))
            return canvas

    def distance_color(distance_value: float, min_distance: float, max_distance: float) -> Tuple[int, int, int]:
        low_color = (46, 125, 50)
        high_color = (198, 40, 40)
        if max_distance <= min_distance + 1e-12:
            mix = 0.5
        else:
            mix = (float(distance_value) - float(min_distance)) / (float(max_distance) - float(min_distance))
            mix = max(0.0, min(1.0, mix))
        return tuple(int(round(low_color[i] * (1.0 - mix) + high_color[i] * mix)) for i in range(3))

    sorted_rows = list(distance_rows)
    labeled_paths = [reference_path] + [Path(str(row["image_path"])) for row in sorted_rows]
    display_labels = build_distinguishable_labels(labeled_paths)
    compared_count = len(sorted_rows)
    columns = min(3, max(1, compared_count))
    card_width = compare_panel_size[0] + 24
    card_height = compare_panel_size[1] + 98
    grid_rows = int(math.ceil(compared_count / float(columns)))
    grid_width = columns * card_width + max(0, columns - 1) * card_gap
    top_width = reference_panel_size[0] + section_gap + summary_panel_width
    canvas_width = padding * 2 + max(grid_width, top_width)

    title_probe = Image.new("RGB", (10, 10), canvas_background)
    title_draw = ImageDraw.Draw(title_probe)
    title_height = text_size(title_draw, "SigLIP Reference Distance Summary", title_font)[1]
    heading_height = text_size(title_draw, "Reference Image", heading_font)[1]
    body_height = text_size(title_draw, "Distance mean: 0.000000", body_font)[1]
    small_height = text_size(title_draw, "filename.png", small_font)[1]

    top_section_height = max(reference_panel_size[1] + heading_height + 16, heading_height + 8 + 7 * (body_height + 8))
    grid_section_height = compared_count * 0
    if compared_count > 0:
        grid_section_height = heading_height + section_gap + grid_rows * card_height + max(0, grid_rows - 1) * card_gap
    canvas_height = padding + title_height + section_gap + top_section_height + padding
    if grid_section_height > 0:
        canvas_height += section_gap + grid_section_height

    canvas = Image.new("RGB", (canvas_width, canvas_height), canvas_background)
    draw = ImageDraw.Draw(canvas)

    title_text = "SigLIP Reference Distance Summary"
    draw.text((padding, padding), title_text, fill=text_primary, font=title_font)
    content_top = padding + title_height + section_gap

    min_row = min(sorted_rows, key=lambda row: float(row["siglip_distance"]))
    max_row = max(sorted_rows, key=lambda row: float(row["siglip_distance"]))

    draw.text((padding, content_top), "Reference Image", fill=text_primary, font=heading_font)
    reference_panel_top = content_top + heading_height + 12
    reference_panel = contain_image(reference_path, reference_panel_size)
    canvas.paste(reference_panel, (padding, reference_panel_top))
    draw.rounded_rectangle(
        (
            padding - 1,
            reference_panel_top - 1,
            padding + reference_panel_size[0],
            reference_panel_top + reference_panel_size[1],
        ),
        radius=10,
        outline=border_neutral,
        width=2,
    )
    draw.text(
        (padding, reference_panel_top + reference_panel_size[1] + 10),
        truncate_text(display_labels.get(reference_path, reference_path.name), 34),
        fill=text_secondary,
        font=small_font,
    )

    summary_left = padding + reference_panel_size[0] + section_gap
    draw.text((summary_left, content_top), "Run Summary", fill=text_primary, font=heading_font)
    min_image_path = Path(str(min_row["image_path"]))
    max_image_path = Path(str(max_row["image_path"]))
    summary_lines = [
        f"Reference: {truncate_text(display_labels.get(reference_path, reference_path.name), 38)}",
        f"Compared images: {compared_count}",
        f"Metric: 1 - cosine similarity",
        f"Mean distance: {float(summary_row['mean_distance']):.6f}",
        f"Min distance: {float(summary_row['min_distance']):.6f} ({truncate_text(display_labels.get(min_image_path, min_image_path.name), 26)})",
        f"Max distance: {float(summary_row['max_distance']):.6f} ({truncate_text(display_labels.get(max_image_path, max_image_path.name), 26)})",
        f"Model: {truncate_text(model_name, 42)}",
    ]
    summary_top = content_top + heading_height + 10
    for line in summary_lines:
        draw.text((summary_left, summary_top), line, fill=text_primary, font=body_font)
        summary_top += body_height + 8

    if compared_count > 0:
        grid_top = content_top + top_section_height + section_gap
        draw.text((padding, grid_top), "Compared Images", fill=text_primary, font=heading_font)
        cards_top = grid_top + heading_height + section_gap
        min_distance = float(summary_row["min_distance"])
        max_distance = float(summary_row["max_distance"])

        for index, row in enumerate(sorted_rows):
            column = index % columns
            row_idx = index // columns
            card_left = padding + column * (card_width + card_gap)
            card_top = cards_top + row_idx * (card_height + card_gap)
            card_right = card_left + card_width
            card_bottom = card_top + card_height
            color = distance_color(float(row["siglip_distance"]), min_distance, max_distance)

            draw.rounded_rectangle(
                (card_left, card_top, card_right, card_bottom),
                radius=14,
                fill=card_background,
                outline=color,
                width=3,
            )

            compare_panel = contain_image(Path(str(row["image_path"])), compare_panel_size)
            panel_left = card_left + (card_width - compare_panel_size[0]) // 2
            panel_top = card_top + 12
            canvas.paste(compare_panel, (panel_left, panel_top))

            text_top = panel_top + compare_panel_size[1] + 12
            compare_path = Path(str(row["image_path"]))
            compare_name = truncate_text(display_labels.get(compare_path, compare_path.name), 28)
            draw.text((card_left + 12, text_top), f"#{index:02d} {compare_name}", fill=text_primary, font=body_bold_font)
            draw.text(
                (card_left + 12, text_top + body_height + 8),
                f"distance: {float(row['siglip_distance']):.6f}",
                fill=color,
                font=body_bold_font,
            )
            draw.text(
                (card_left + 12, text_top + 2 * (body_height + 8)),
                f"cosine: {float(row['cosine_similarity']):.6f}",
                fill=text_secondary,
                font=small_font,
            )

    canvas.save(output_path)
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute SigLIP embedding distance for two or more images.")
    parser.add_argument("--input_path", type=str, default=None, help="Single image path or directory of images.")
    parser.add_argument("--image_paths", nargs="*", default=None, help="Explicit image paths. Overrides input_path.")
    parser.add_argument("--reference_image", type=str, default=None, help="Optional reference image path.")
    parser.add_argument("--device", type=str, default="auto", help="Device to run on: auto, cpu, cuda:0, ...")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for SigLIP image embedding inference.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to store CSV outputs.")
    parser.add_argument("--include_ext", type=str, default=DEFAULT_INCLUDE_EXT)
    parser.add_argument("--exclude_substrings", type=str, default=DEFAULT_EXCLUDE_SUBSTRINGS)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    return parser


def run_evaluation(
    input_path: Optional[str] = None,
    image_paths: Optional[Sequence[str]] = None,
    reference_image: Optional[str] = None,
    device: str = "auto",
    batch_size: int = 8,
    output_dir: Optional[str] = None,
    include_ext: str = DEFAULT_INCLUDE_EXT,
    exclude_substrings: str = DEFAULT_EXCLUDE_SUBSTRINGS,
    model_name: str = DEFAULT_MODEL_NAME,
) -> Dict[str, object]:
    include_exts = parse_include_exts(include_ext)
    candidate_paths = collect_image_paths(
        input_path=input_path,
        image_paths=image_paths,
        include_ext=include_ext,
        exclude_substrings=exclude_substrings,
    )

    reference_path = choose_reference_image(candidate_paths, reference_image=reference_image)
    reference_path = validate_image_path(reference_path, include_exts)
    selected_paths = dedupe_paths([reference_path, *candidate_paths])

    if len(selected_paths) < 2:
        raise ValueError("Need at least two valid images to compute SigLIP embedding distance.")

    output_root = resolve_output_dir(output_dir=output_dir, input_path=input_path, selected_paths=selected_paths)
    output_root.mkdir(parents=True, exist_ok=True)

    model, resolved_device = load_siglip_model(model_name=model_name, device=device)
    embeddings = compute_embeddings(
        image_paths=selected_paths,
        model=model,
        device=resolved_device,
        batch_size=batch_size,
    )
    distance_rows = compute_reference_distances(
        image_paths=selected_paths,
        reference_path=reference_path,
        embeddings=embeddings,
    )
    summary_row = summarize_reference_distances(reference_path=reference_path, distance_rows=distance_rows)

    reference_distances_path = output_root / "reference_distances.csv"
    summary_path = output_root / "summary.csv"
    visualization_path = output_root / DEFAULT_VISUALIZATION_NAME
    write_csv(
        reference_distances_path,
        distance_rows,
        fieldnames=["reference_path", "image_path", "cosine_similarity", "siglip_distance"],
    )
    write_csv(
        summary_path,
        [summary_row],
        fieldnames=["reference_path", "image_count", "mean_distance", "min_distance", "max_distance"],
    )
    create_summary_visualization(
        reference_path=reference_path,
        distance_rows=distance_rows,
        summary_row=summary_row,
        output_path=visualization_path,
        model_name=model_name,
    )

    return {
        "reference_path": reference_path,
        "image_paths": selected_paths,
        "model_name": model_name,
        "device": resolved_device,
        "output_dir": output_root,
        "reference_distances_path": reference_distances_path,
        "summary_path": summary_path,
        "visualization_path": visualization_path,
        "distance_rows": distance_rows,
        "summary_row": summary_row,
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    result = run_evaluation(
        input_path=args.input_path,
        image_paths=args.image_paths,
        reference_image=args.reference_image,
        device=args.device,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        include_ext=args.include_ext,
        exclude_substrings=args.exclude_substrings,
        model_name=args.model_name,
    )

    reference_path = result["reference_path"]
    distance_rows = result["distance_rows"]
    summary_row = result["summary_row"]

    print(
        f"[SigLIP] model={result['model_name']} device={result['device']} "
        f"reference={reference_path} image_count={summary_row['image_count']}"
    )
    for row in distance_rows:
        print(
            f"image={row['image_path']} cosine_similarity={row['cosine_similarity']:.6f} "
            f"siglip_distance={row['siglip_distance']:.6f}"
        )
    print(
        f"[Summary] mean_distance={summary_row['mean_distance']:.6f} "
        f"min_distance={summary_row['min_distance']:.6f} "
        f"max_distance={summary_row['max_distance']:.6f}"
    )
    print(f"reference_distances_csv: {result['reference_distances_path']}")
    print(f"summary_csv:             {result['summary_path']}")
    print(f"summary_visualization:   {result['visualization_path']}")


if __name__ == "__main__":
    main()
