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
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open the URL it prints (`http://127.0.0.1:8765` or the next free port). Bound to localhost only.

On first launch, open **Setup** and set the ComfyUI root if it was not auto-detected. This studio talks to **any local ComfyUI** — Desktop, Windows portable, Easy-Install, Stability Matrix, or a manual git clone. It does not have to be Easy-Install.

Paste the folder that contains `main.py`, or the wrapper folder that contains a `ComfyUI` subfolder. Leave **Python** blank and Save — the studio fills the interpreter that install actually uses.

| Install | What to paste | Python (auto-filled if blank) |
| --- | --- | --- |
| Windows portable | `ComfyUI_windows_portable` or the inner `ComfyUI` folder | `python_embeded\python.exe` next to `ComfyUI` |
| Easy-Install | `ComfyUI-Easy-Install` or the inner `ComfyUI` folder | same `python_embeded\python.exe` (there is no venv) |
| ComfyUI Desktop | the **user data** folder you chose at install (often `Documents\ComfyUI`), not the app under `AppData\Local\Programs` | `.venv\Scripts\python.exe` (macOS/Linux: `.venv/bin/python`) |
| Stability Matrix / manual venv | the `ComfyUI` folder with `main.py` | `venv` or `.venv` inside that folder |
| git clone, system/conda Python | the clone folder with `main.py` | browse to the `python` that launches ComfyUI |

Extract still needs, inside that install (or on an `extra_model_paths.yaml` search path):

- `custom_nodes/ComfyUI-MiniMaxH3Mod`
- `models/vae/minimax_h3_video_vae_fp16.safetensors`

RefMods are written to `models/refmods/<folder>/` under the same root. Datasets root can point at this repo’s `datasets/` folder or an existing AI Toolkit datasets directory.

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
