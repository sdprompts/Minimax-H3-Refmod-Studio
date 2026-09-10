# H3 RefMod Studio

Local web app for building **MiniMax-H3 RefMods** — small `.safetensors` identity adapters for ComfyUI.

This repo is source only. Datasets, extracted mods, thumbnails, and `config.json` stay on your machine (gitignored).

You keep a named dataset of stills (AI Toolkit-style). When the set is good, the studio runs the official `extract_mod.py` from [ComfyUI-MiniMaxH3Mod](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod) and writes the file into `ComfyUI/models/refmods/<folder>/`. Change a shot later and re-extract.

This does **not** generate video. It only encodes references through the MiniMax-H3 video VAE.

## Loop

1. Create a dataset (or import / link an existing folder).
2. Drop 8–20 stills. Tag front / three-quarter / profile / body if you want the mix meter.
3. Choose a **RefMod folder** (new identity sets default to `identity/`). Extract. The mod lands in that folder under `models/refmods/` and a copy stays next to the dataset.
4. Swap a blurry frame, hit **Re-extract**. ComfyUI’s loader name stays the same unless you bump the version or change the folder.

**ComfyUI** lists files in `models/refmods/` (what the loader dropdown sees). **Studio** lists this app’s dataset copies, grouped by the same folders, and can copy a missing or stale file back into ComfyUI.

Select RefMods on either page and **Upload selected** to a Hugging Face dataset or model repo. Setup stores the token and repo (`user/name`). Uploads keep the ComfyUI folder path (`identity/name.safetensors`). The studio remembers sha/size so Hub stale shows after you re-extract.

## Prerequisites

Extract needs a ComfyUI install that already has:

- `custom_nodes/ComfyUI-MiniMaxH3Mod` ([clone it](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod))
- `models/vae/minimax_h3_video_vae_fp16.safetensors`

You can curate datasets without that. Extract stays disabled until Setup is healthy.

Guides:

- [Creation guide](https://huggingface.co/datasets/malcolmrey/various/blob/main/h3-center/docs/MINIMAX_H3_REFMOD_CREATION_GUIDE.md)
- [Install & use in ComfyUI](https://huggingface.co/datasets/malcolmrey/various/blob/main/h3-center/docs/MINIMAX_H3_REFMODS_INSTALLATION_AND_USAGE_GUIDE.md)

## Run

```text
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open the URL it prints (`http://127.0.0.1:8765` or the next free port). Bound to localhost only.

On first launch, open **Setup** and set the ComfyUI root if it was not auto-detected.

**ComfyUI-Easy-Install** does not use a venv. It ships an embedded interpreter:

```text
ComfyUI-Easy-Install\
  python_embeded\python.exe      ← this is the Python to use
  ComfyUI\                       ← paste this, or the parent folder
    main.py
    custom_nodes\
    models\vae\
```

Paste either `...\ComfyUI-Easy-Install` or `...\ComfyUI-Easy-Install\ComfyUI`. Leave Python blank and Save — the studio fills `python_embeded\python.exe`. Datasets root can point at this repo’s `datasets/` folder or an existing AI Toolkit datasets directory.

## ComfyUI

In a MiniMax-H3 workflow:

`Load H3 RefMods` → `Apply H3 RefMod` between the conditioning node and the guider.

Pick the mod you extracted (`identity/sdprompts_minimaxh3_<name>_v1_refmod` after a folder extract). Strength `1.0`, copies `1`. After MiniMaxH3Mod 0.2.x, leave Apply on `constant` / `linear` / `1.0` so stacked stills inject at full strength. Describe the subject in the prompt; there is no `<Subject 1>` trigger.

If the loader dropdown is stale after extract, use **Refresh RefMods** on the loader (0.2.5+) instead of a full ComfyUI model refresh.

Identity defaults used by this studio: Full Reference (`encode`), concept type `identity`, short edge `1024`, max tokens `8192`, folder `identity/`. Filenames default to `sdprompts_minimaxh3_<name>_v1_refmod` (prefix is set in Setup).

Already-extracted datasets that still live in the RefMod root stay there on re-extract. New sets go into a folder so ComfyUI can group them.

## Tests

```text
pytest
```
