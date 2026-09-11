# H3 RefMod Studio

Local web app for building **MiniMax-H3 RefMods** — small `.safetensors` identity adapters for ComfyUI.

This repo is source only. Datasets, extracted mods, thumbnails, and `config.json` stay on your machine (gitignored).

You keep a named dataset of stills (and optional clips). When the set is good, the studio runs the official `extract_mod.py` from [ComfyUI-MiniMaxH3Mod](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod) and writes the file into `ComfyUI/models/refmods/<folder>/`. Change a shot later and re-extract.

This does **not** generate video. It only encodes references through the MiniMax-H3 video VAE.

## Features

- **Datasets** — named stills folders with covers, notes, concept type, and a live mix meter
- **Import** — copy a folder in, or **link** an existing AI Toolkit folder in place
- **Split sheet** — cut a contact sheet into panels and make one dataset per sheet
- **Sheet editor** — drop PNG/JPG/WebP stills and mp4/webm/mov clips, toggle **use**, set **cover**, drag to reorder
- **Quality check** — scores faces 0–100, labels close-up / medium / full-body and head angle, recommends a diverse 10, and lists what to shoot next
- **Watermark clean** — scan edge marks, drag a crop per still, keep originals under `_originals/`
- **Extract** — Full Reference (`encode`) or Compressed Reference (`training`), live progress + log, overwrite or bump `_v2`
- **Library** — ComfyUI `models/refmods/` vs Studio copies next to each dataset; rename folders; **Sync to ComfyUI**
- **Hugging Face** — upload selected mods to a dataset or model repo, keeping the ComfyUI folder path
- **Sample prompts** — copy-paste MiniMax-H3 T2VA shots in 5s, 10s, and 15s, one subject, plus R-rated spicy tests
- **CLI** — `score_refs.py` ranks a folder of stills from the terminal

Extract needs ComfyUI + the MiniMaxH3Mod pack. You can still curate datasets before Setup is healthy.

## Pages

| Page | What it is |
| --- | --- |
| **Datasets** | Card grid of your sets. Search, filter by RefMod folder, create / import / split. |
| **Dataset sheet** | The stills for one set. Drop files, extract, quality, clean. |
| **Library → ComfyUI** | What the ComfyUI loader dropdown sees (`models/refmods/`). |
| **Library → Studio** | Copies kept next to each dataset. Sync / copy / replace into ComfyUI. |
| **Sample Prompts** | Ready-made `<Subject 1>` T2VA tests in 5s / 10s / 15s. |
| **Instructions** | In-app guide (same material as below, with jump links). |
| **Setup** | ComfyUI root, Python, VAE, output folder, datasets root, Hub token. |

The header health chip turns green when extract is ready.

## Loop

1. Point **Setup** at your ComfyUI (needed for extract only).
2. Create a dataset, or import / link an existing folder.
3. Drop **8–20 stills** for an identity set (boosters usually 6–12). Add clips if you want.
4. Pick a **RefMod folder** (new identity sets default to `identity/`). Extract.
5. The mod lands in `models/refmods/<folder>/`. A copy stays next to the dataset.
6. Swap a weak still, hit **Re-extract**. The ComfyUI loader name stays the same unless you bump the version or change the folder.

## Datasets

**New dataset** asks for a name, optional display name, and a RefMod folder.

**Import folder** has two modes:

- **Copy** — duplicates stills (and clips) into this app’s `datasets/` root
- **Link** — uses the folder in place (AI Toolkit-style). Cleaning writes into that source folder; originals still go to `_originals/`

Tick **Recursive** to pull nested images. Optional slug + folder as usual.

**Split sheet** takes one or more contact-sheet images, previews cut lines, lets you skip panels, then creates one dataset per sheet.

The Datasets list shows cover, still/clip counts, concept, folder, and status: `new` / `extracted` / `updated`. Search matches name, slug, mod name, folder, and notes.

### Sheet

Drop files onto the sheet (or click the drop zone). PNG / JPG / WebP stills get thumbnails. Clips (`.mp4`, `.webm`, `.mov`, `.mkv`, `.avi`, `.m4v`) show as a **CLIP** tile — no thumbnail, cannot be the list cover, watermark clean skips them. Audio is ignored.

On each tile:

- **use** — leave the file in the folder without encoding it
- **cover** — Datasets-list thumbnail (does not mark the set dirty)
- **×** — delete the file
- Drag to reorder (the first still anchors the extract canvas)

Low-res stills (short edge under 1024) are marked. Pixel size is on the thumbnail.

Identity mix to aim for: about 40% front, 30% three-quarter, 20% profile, 10% body. Vary lighting and backgrounds so the encoder locks on the face, not the room.

Clips add motion and extra angles; they do not replace a good still mix. Extract samples at most **60** frames per clip, Full Reference keeps **16** of those (snapped to H3’s frame grid). Stills and clips share one token cap (default **8192**).

### Sidebar

- **Display name** — UI only
- **Description** — stored *on* the RefMod (loaders can surface it)
- **Notes** — yours only; not written into the mod; does not trigger re-extract
- **Concept** — `identity`, `generic`, `pose_motion`, `clothing`, `background`, `style`
- **Mod name** / **folder** — ComfyUI lists `folder/mod_name`
- **Advanced extract** — mode, resolution, max tokens, pool, refinement steps, multiplier, device (`auto` / `cuda` / `cpu`)
- **Overwrite** vs **bump version** (`_v2_refmod`, …) so the old loader entry can stay

Identity defaults: Full Reference (`encode`), concept `identity`, short edge `1024`, max tokens `8192`, folder `identity/`. Filenames default to `sdprompts_minimaxh3_<name>_v1_refmod` (prefix is set in Setup).

A dataset is **updated** (dirty) when enabled files, order, description, concept, mod name, folder, or extract settings change. Cover, display name, notes, captions, device, and disabled files do **not** mark it dirty. A successful extract (or installing a matching copy into ComfyUI) clears it.

### Quality

**Check quality** ranks stills 0–100 from face sharpness, size, lighting, and pose (YuNet). It labels close-up / medium / full-body plus head angle, flags crowded near-duplicate poses, marks a diverse recommended 10, and lists what to shoot next (missing 3/4s, a medium, a full-body, sharper close-ups). Clips are skipped.

Click a score chip for the breakdown. **Use recommended** unchecks everything except those picks. You can sort the sheet by quality or by picks. Rank-only until you apply recommended.

From a folder, without the UI:

```text
python score_refs.py path\to\stills
```

### Clean watermarks

**Clean watermarks** scans for edge marks / caption bars. Drag handles per still (or **Use scan**). Crop this still, crop all, or copy this crop to all. **Don't use** unchecks the still; **Ignore** keeps it as-is. Originals are saved under `_originals/` and can be undone.

Linked datasets write into the source folder.

## Library

Two panes, same folders:

- **ComfyUI** — files in `models/refmods/` (loader dropdown). Delete, rename folders, open on disk.
- **Studio** — `.safetensors` copies next to each dataset. Status: in ComfyUI / out of date / not in ComfyUI. **Copy to ComfyUI** / **Replace in ComfyUI** for one mod.

**Sync to ComfyUI** rebuilds `models/refmods/` to match Studio folders: missing folders created, misplaced mods moved, missing mods copied, empty leftovers removed. Unrelated ComfyUI-only files stay put.

Folder names *are* the loader path: `identity/name`, `celebs/june/name`. Nested names use `/`. Rename from the folder row; datasets that used that folder are updated.

A healthy identity RefMod is about **1.1–1.6 MB**. Under ~100 KB usually means encode failed (shown as **tiny**).

After a sync or extract, use **Refresh RefMods** on the ComfyUI loader (0.2.5+) if the dropdown looks stale.

## Hugging Face

On Setup, save a write token and a repo (`user/name`), dataset or model, private or public. On either Library pane, select mods and **Upload selected**. Uploads keep the ComfyUI folder path (`identity/name.safetensors`). The studio remembers sha/size so **Hub stale** shows after you re-extract.

## Sample prompts

Copy-paste MiniMax-H3 T2VA blocks sized for **5s, 10s, and 15s**. One main subject. Edit the Subject 1 line to the person’s name and wardrobe. Match the same duration on the H3 sampler — later shot timestamps stay inside that length.

Use `<Subject 1>`, not `<Picture 1>`. Prompt shape: `subject_definitions`, then `integrated_multimodal_description`, `overall_soundscape`, `non_diegetic_music`. Spicy is R-rated (lingerie, undressing, implied heat), not explicit.

## Prerequisites

Extract needs a ComfyUI install that already has:

- `custom_nodes/ComfyUI-MiniMaxH3Mod` ([clone it](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod))
- `models/vae/minimax_h3_video_vae_fp16.safetensors` (or another folder listed in `extra_model_paths.yaml`)

You can curate datasets without that. Extract stays disabled until Setup is healthy.

Community guides by [malcolmrey](https://huggingface.co/malcolmrey):

- [Creation guide](https://huggingface.co/datasets/malcolmrey/various/blob/main/h3-center/docs/MINIMAX_H3_REFMOD_CREATION_GUIDE.md)
- [Install & use in ComfyUI](https://huggingface.co/datasets/malcolmrey/various/blob/main/h3-center/docs/MINIMAX_H3_REFMODS_INSTALLATION_AND_USAGE_GUIDE.md)

## Run

Windows: double-click `start.bat` (creates `.venv`, installs deps, launches).

Or:

```text
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open the URL it prints (`http://127.0.0.1:8765` or the next free port). Bound to localhost only.

On first launch, open **Setup** and set the ComfyUI root if it was not auto-detected. This studio talks to **any local ComfyUI** — Desktop, Windows portable, Easy-Install, Stability Matrix, or a manual git clone.

Paste the folder that contains `main.py`, or the wrapper folder that contains a `ComfyUI` subfolder. Leave **Python** blank and Save — the studio fills the interpreter that install actually uses.

| Install | What to paste | Python (auto-filled if blank) |
| --- | --- | --- |
| Windows portable | `ComfyUI_windows_portable` or the inner `ComfyUI` folder | `python_embeded\python.exe` next to `ComfyUI` |
| Easy-Install | `ComfyUI-Easy-Install` or the inner `ComfyUI` folder | same `python_embeded\python.exe` (there is no venv) |
| ComfyUI Desktop | the **user data** folder you chose at install (often `Documents\ComfyUI`), not the app under `AppData\Local\Programs` | `.venv\Scripts\python.exe` (macOS/Linux: `.venv/bin/python`) |
| Stability Matrix / manual venv | the `ComfyUI` folder with `main.py` | `venv` or `.venv` inside that folder |
| git clone, system/conda Python | the clone folder with `main.py` | browse to the `python` that launches ComfyUI |

RefMods are written to `models/refmods/<folder>/` under that root. Datasets root can point at this repo’s `datasets/` folder or an existing AI Toolkit datasets directory.

This app’s `.venv` is only for the studio UI. Extract always uses ComfyUI’s Python.

## Use in ComfyUI

In a MiniMax-H3 workflow:

`Load H3 RefMods` → `Apply H3 RefMod` between the conditioning node and the guider.

Pick the mod you extracted (`identity/sdprompts_minimaxh3_<name>_v1_refmod` after a folder extract). Strength `1.0`, copies `1`. After MiniMaxH3Mod 0.2.x, leave Apply on `constant` / `linear` / `1.0` so stacked stills inject at full strength. Describe the subject in the prompt; there is no `<Subject 1>` trigger on the Apply node — that token lives in the prompt (see Sample Prompts).

Already-extracted datasets that still live in the RefMod root stay there on re-extract. New sets go into a folder so ComfyUI can group them.
