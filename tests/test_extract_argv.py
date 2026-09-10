from pathlib import Path

from backend.extract import build_argv, parse_progress, parse_saved_line


def test_build_argv_order_and_defaults(tmp_path):
    python = tmp_path / "python.exe"
    script = tmp_path / "extract_mod.py"
    vae = tmp_path / "vae.safetensors"
    out = tmp_path / "refmods"
    images = [tmp_path / "01.png", tmp_path / "03.png"]
    videos = [tmp_path / "walk.mp4"]
    argv = build_argv(
        python=python,
        script=script,
        vae=vae,
        output_dir=out,
        name="minimaxh3_feliciaday_v1_refmod",
        description="Felicia Day primary persona",
        concept_type="identity",
        extract={"mode": "encode", "resolution": 1024, "max_tokens": 8192, "pool": 16, "identity": 0, "multiplier": 1, "device": "auto"},
        images=images,
        videos=videos,
    )
    assert argv[:3] == [str(python), "-u", str(script)]
    assert argv[argv.index("--mode") + 1] == "encode"
    assert argv[argv.index("--concept-type") + 1] == "identity"
    assert argv[argv.index("--max-tokens") + 1] == "8192"
    assert argv[argv.index("--image") + 1] == str(images[0])
    image_flags = [argv[i + 1] for i, a in enumerate(argv) if a == "--image"]
    assert image_flags == [str(p) for p in images]
    assert argv[argv.index("--video") + 1] == str(videos[0])
    assert "--identity" not in argv


def test_build_argv_training_passes_identity(tmp_path):
    argv = build_argv(
        python=tmp_path / "python.exe",
        script=tmp_path / "extract_mod.py",
        vae=tmp_path / "vae.safetensors",
        output_dir=tmp_path / "refmods" / "style",
        name="mod",
        description="",
        concept_type="style",
        extract={"mode": "training", "resolution": 1024, "max_tokens": 5120, "pool": 16, "identity": 500, "multiplier": 1, "device": "auto"},
        images=[tmp_path / "01.png"],
        videos=[],
    )
    assert argv[argv.index("--output") + 1] == str(tmp_path / "refmods" / "style")
    assert argv[argv.index("--identity") + 1] == "500"


def test_parse_saved_line():
    line = "[extract] saved video mod 'minimaxh3_x_v1_refmod' (4096 tokens, 1.23 MB) -> C:\\ComfyUI\\models\\refmods\\minimaxh3_x_v1_refmod.safetensors"
    parsed = parse_saved_line(line)
    assert parsed["token_count"] == 4096
    assert parsed["mb"] == 1.23
    assert parsed["name"] == "minimaxh3_x_v1_refmod"
    assert parsed["path"].endswith("minimaxh3_x_v1_refmod.safetensors")


def test_parse_progress_encoding():
    log = [
        "[extract] loading VAE models/vae/minimax_h3_video_vae_fp16.safetensors (device=auto)",
        "[extract] image C:\\ds\\01.png: (1, 1024, 1024, 3) (mode=encode)",
        "[extract] image C:\\ds\\02.png: (1, 1024, 1024, 3) (mode=encode)",
    ]
    prog = parse_progress(log, total=8, status="running")
    assert prog["stage"] == "encoding"
    assert prog["encoded"] == 2
    assert prog["current"] == "02.png"
    assert 8 < prog["percent"] < 50
    done = parse_progress(log + ["[extract] saved video mod 'x' (100 tokens, 1.10 MB) -> C:\\out\\x.safetensors"], 8, "done")
    assert done["percent"] == 100
    assert done["stage"] == "saved"
