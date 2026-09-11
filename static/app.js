const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const state = {
  health: null,
  datasets: [],
  dataset: null,
  library: [],
  folders: [],
  studio: [],
  studioFolders: [],
  query: "",
  folderFilter: "",
  datasetFolderFilter: "",
  studioFolderFilter: "",
  libraryCollapsed: {},
  studioCollapsed: {},
  hubSelected: {},
  hubUploading: false,
  hubLog: [],
  hubError: null,
  extracting: false,
  log: [],
  result: null,
  error: null,
  progress: null,
  extractFiles: [],
  clean: null,
  sheetSort: "",
  qualityBusy: false,
  libraryLoading: false,
  studioLoading: false,
  datasetsLoading: false,
  promptQuery: "",
};

let renderGen = 0;

function route() {
  const hash = location.hash.replace(/^#/, "") || "/datasets";
  const parts = hash.split("/").filter(Boolean);
  return { view: parts[0] || "datasets", slug: parts[1] || "" };
}

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  let data = null;
  const text = await res.text();
  try { data = text ? JSON.parse(text) : null; } catch { data = { detail: text }; }
  if (!res.ok) {
    const detail = data && data.detail ? data.detail : res.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return data;
}

function fmtBytes(n) {
  if (!n) return "0 B";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(2)} MB`;
}

function pixelSize(item) {
  const size = item && (item.size || item);
  if (size && size.w && size.h) return `${size.w}×${size.h}`;
  return "";
}

function notesPreview(text) {
  const line = String(text || "").trim().split(/\n/)[0];
  if (!line) return "";
  return line.length > 72 ? `${line.slice(0, 72)}…` : line;
}

function qualityScore(item) {
  const s = item && item.quality && item.quality.score;
  return typeof s === "number" ? s : null;
}

function qualityBand(score) {
  if (score == null) return "skip";
  if (score >= 20) return "hi";
  if (score >= 8) return "mid";
  return "lo";
}

function scaleShort(scale) {
  return ({ closeup: "CU", medium: "MD", full: "FB" })[scale] || "";
}

function poseShort(bin) {
  return ({
    front: "front",
    "three-quarter-left": "3/4 L",
    "three-quarter-right": "3/4 R",
    "profile-left": "prof L",
    "profile-right": "prof R",
  })[bin] || "";
}

function poseMix(items) {
  const faced = (items || []).filter((it) => it.kind === "image" && qualityScore(it) > 0);
  const scales = { closeup: 0, medium: 0, full: 0 };
  const poses = {};
  const bins = {};
  for (const it of faced) {
    const dbg = (it.quality && it.quality.debug) || {};
    const sc = dbg.scale || "closeup";
    const pb = dbg.pose_bin || "unknown";
    if (scales[sc] != null) scales[sc] += 1;
    poses[pb] = (poses[pb] || 0) + 1;
    const key = `${sc} ${pb}`;
    bins[key] = (bins[key] || 0) + 1;
  }
  const n = faced.length;
  const crowded = Object.entries(bins)
    .filter(([, c]) => n >= 3 && c >= 4 && c / n >= 0.3)
    .sort((a, b) => b[1] - a[1]);
  const picks = (items || []).filter((it) => it.quality && it.quality.picked).map((it) => it.file);
  return { faced: n, scales, poses, crowded, picks };
}

function sheetRows(d) {
  const rows = (d.items || []).map((it, i) => ({ it, i }));
  if (state.sheetSort === "quality") {
    return rows.sort((a, b) => {
      const av = qualityScore(a.it);
      const bv = qualityScore(b.it);
      return (bv == null ? -1 : bv) - (av == null ? -1 : av) || a.i - b.i;
    });
  }
  if (state.sheetSort === "picks") {
    const picked = (it) => (it.quality && it.quality.picked ? 1 : 0);
    return rows.sort((a, b) => {
      const pd = picked(b.it) - picked(a.it);
      if (pd) return pd;
      const av = qualityScore(a.it);
      const bv = qualityScore(b.it);
      return (bv == null ? -1 : bv) - (av == null ? -1 : av) || a.i - b.i;
    });
  }
  return rows;
}

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function setChip() {
  const el = $("#health-chip");
  const h = state.health;
  if (!h) { el.textContent = "checking…"; el.className = "health-chip"; return; }
  if (h.ready) {
    const pack = h.pack_version ? ` · pack ${h.pack_version}` : "";
    el.textContent = `extract ready · ${h.device}${pack}`;
    el.className = "health-chip ok";
  } else {
    el.textContent = "extract blocked";
    el.className = "health-chip bad";
  }
  const banner = $("#banner");
  if (!h.ready) {
    banner.classList.remove("hidden");
    banner.innerHTML = `Extract needs a ComfyUI install. Missing: ${escapeHtml((h.missing || []).join("; "))}. You can still curate datasets. <a href="#/setup">Open Setup</a>.`;
  } else {
    banner.classList.add("hidden");
    banner.innerHTML = "";
  }
}

function libraryPane() {
  const { view, slug } = route();
  if (view === "studio" || (view === "library" && slug === "studio")) return "studio";
  return "comfy";
}

function librarySwitchHtml() {
  const pane = libraryPane();
  return `<div class="lib-switch" role="tablist" aria-label="Library source">
    <a href="#/library" class="${pane === "comfy" ? "active" : ""}" role="tab" aria-selected="${pane === "comfy"}">ComfyUI</a>
    <a href="#/library/studio" class="${pane === "studio" ? "active" : ""}" role="tab" aria-selected="${pane === "studio"}">Studio</a>
  </div>`;
}

function busyHtml(loading, hasRows) {
  if (!loading) return "";
  return `<span class="lib-loading">${hasRows ? "Refreshing…" : "Loading…"}</span>`;
}

function setTab() {
  const { view } = route();
  const tab = view === "studio" ? "library" : view;
  $$("#tabs a").forEach((a) => a.classList.toggle("active", a.dataset.view === tab));
}

async function loadHealth() {
  state.health = await api("/api/health");
  setChip();
}

async function loadDatasets() {
  const data = await api("/api/datasets");
  state.datasets = data.datasets || [];
}

async function loadDataset(slug) {
  state.dataset = await api(`/api/datasets/${encodeURIComponent(slug)}`);
}

async function loadLibrary(lite = false) {
  const data = await api(lite ? "/api/library?lite=1" : "/api/library");
  state.library = data.mods || [];
  state.folders = data.folders || [];
  return data;
}

async function loadStudio() {
  const data = await api("/api/studio");
  state.studio = data.mods || [];
  state.studioFolders = data.folders || [];
}

function closeModal() {
  const el = $("#modal");
  el.classList.add("hidden");
  el.innerHTML = "";
}

function openModal(html, extraClass = "") {
  const el = $("#modal");
  el.classList.remove("hidden");
  el.innerHTML = `<div class="dialog ${extraClass}">${html}</div>`;
  el.addEventListener("click", (e) => { if (e.target === el) closeModal(); }, { once: true });
}

function pickRow(id, value, kind, extra = {}) {
  const types = extra.filetypes ? escapeHtml(JSON.stringify(extra.filetypes)) : "";
  return `<div class="pick">
    <input id="${id}" type="text" value="${escapeHtml(value || "")}" placeholder="${escapeHtml(extra.placeholder || "")}">
    <button type="button" class="btn pick-btn" data-kind="${kind}" data-target="${id}" data-title="${escapeHtml(extra.title || "")}" data-filetypes="${types}">Browse</button>
  </div>`;
}

async function browsePath(kind, targetId, title, filetypes) {
  const target = $(`#${targetId}`);
  const body = { kind, title: title || "", initial: target ? target.value : "" };
  if (filetypes) body.filetypes = filetypes;
  const res = await api("/api/pick", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (res.path && target) target.value = res.path;
  return res.path || "";
}

function bindPickers() {
  $$(".pick-btn").forEach((btn) => {
    btn.onclick = async (e) => {
      e.preventDefault();
      try {
        const types = btn.dataset.filetypes ? JSON.parse(btn.dataset.filetypes) : null;
        await browsePath(btn.dataset.kind, btn.dataset.target, btn.dataset.title, types);
      } catch (err) {
        alert(err.message);
      }
    };
  });
}

function conceptOptions(selected) {
  const types = (state.health && state.health.concept_types) || ["identity", "generic", "style"];
  return types.map((t) => `<option ${t === selected ? "selected" : ""}>${t}</option>`).join("");
}

function knownFolders() {
  const types = (state.health && state.health.concept_types) || [];
  const fromHealth = (state.health && state.health.refmod_folders) || [];
  const all = [...new Set([...fromHealth, ...(state.folders || []), ...types].filter(Boolean))];
  all.sort((a, b) => a.localeCompare(b));
  return all;
}

function rememberFolder(name) {
  if (!name) return;
  if (!state.folders.includes(name)) state.folders.push(name);
  if (state.health) {
    const list = state.health.refmod_folders || [];
    if (!list.includes(name)) state.health.refmod_folders = [...list, name];
  }
}

function folderSelectHtml(id, selected) {
  const sel = selected || "";
  const options = [
    `<option value="" ${sel === "" ? "selected" : ""}>(root)</option>`,
    ...knownFolders().map((f) => `<option value="${escapeHtml(f)}" ${f === sel ? "selected" : ""}>${escapeHtml(f)}</option>`),
    `<option value="__new__">New folder…</option>`,
  ].join("");
  return `<label class="field">RefMod folder
    <select id="${id}">${options}</select>
    <input id="${id}-new" type="text" class="hidden" placeholder="celebs or identity/june">
  </label>`;
}

function bindFolderSelect(id) {
  const sel = $(`#${id}`);
  const neu = $(`#${id}-new`);
  if (!sel || !neu) return;
  const sync = () => {
    const isNew = sel.value === "__new__";
    neu.classList.toggle("hidden", !isNew);
    if (isNew) neu.focus();
  };
  sel.addEventListener("change", sync);
  sync();
}

function folderSelectValue(id) {
  const sel = $(`#${id}`);
  const neu = $(`#${id}-new`);
  if (!sel) return "";
  if (sel.value === "__new__") return (neu && neu.value.trim()) || "";
  return sel.value;
}

async function createFolder(name) {
  const res = await api("/api/library/folders", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  const created = (res && res.name) || name;
  rememberFolder(created);
  if (res && res.folders) state.folders = res.folders;
  return created;
}

async function resolveFolderSelect(id) {
  const sel = $(`#${id}`);
  const value = folderSelectValue(id);
  if (sel && sel.value === "__new__") {
    if (!value) throw new Error("name the new folder");
    return createFolder(value);
  }
  return value;
}

function promptCreateFolder(after) {
  openModal(`
    <h3>New RefMod folder</h3>
    <label>Name<input id="m-name" type="text" placeholder="identity or celebs/june"></label>
    <p class="meta">Created under <span class="mono">models/refmods/</span>. Nested names use <span class="mono">/</span>.</p>
    <p class="warn hidden" id="m-err"></p>
    <div class="row">
      <button class="btn" id="m-cancel">Cancel</button>
      <button class="btn primary" id="m-ok">Create</button>
    </div>
  `);
  $("#m-name").focus();
  $("#m-cancel").onclick = closeModal;
  $("#m-ok").onclick = async () => {
    const name = $("#m-name").value.trim();
    const err = $("#m-err");
    if (!name) return;
    try {
      await createFolder(name);
      closeModal();
      if (after) await after(name);
    } catch (ex) {
      err.classList.remove("hidden");
      err.textContent = ex.message;
    }
  };
}

function showSyncResult(res) {
  const created = (res.folders_created || []).map(escapeHtml).join("<br>");
  const removed = (res.folders_removed || []).map(escapeHtml).join("<br>");
  const errs = (res.errors || []).map((e) => escapeHtml(`${e.slug}: ${e.error}`)).join("<br>");
  openModal(`
    <h3>ComfyUI folders synced</h3>
    <p class="meta">Moved ${res.moved || 0} · Copied ${res.copied || 0} · Replaced ${res.replaced || 0} · Already in place ${res.kept || 0}${res.skipped ? ` · Skipped ${res.skipped}` : ""}</p>
    ${created ? `<div class="field">Created folders<div class="mono">${created}</div></div>` : ""}
    ${removed ? `<div class="field">Removed empty<div class="mono">${removed}</div></div>` : ""}
    ${errs ? `<p class="warn">${errs}</p>` : ""}
    <div class="row"><button class="btn primary" id="m-ok">OK</button></div>
  `);
  $("#m-ok").onclick = closeModal;
}

async function syncComfyLayout(after) {
  if (!confirm("Rebuild models/refmods to match Studio folders? Dataset mods are moved or copied. Unrelated ComfyUI files are left alone.")) return;
  try {
    const res = await api("/api/studio/sync-comfy", { method: "POST" });
    if (after) await after();
    showSyncResult(res);
  } catch (err) {
    alert(err.message);
  }
}

function promptRenameFolder(src, after) {
  if (!src) return;
  openModal(`
    <h3>Rename folder</h3>
    <p class="meta">Moves every RefMod in <span class="mono">${escapeHtml(src)}</span> and updates datasets that use it (including nested paths).</p>
    <label>New name<input id="m-name" type="text" value="${escapeHtml(src)}"></label>
    <p class="warn hidden" id="m-err"></p>
    <div class="row">
      <button class="btn" id="m-cancel">Cancel</button>
      <button class="btn primary" id="m-ok">Rename</button>
    </div>
  `);
  $("#m-name").focus();
  $("#m-cancel").onclick = closeModal;
  $("#m-ok").onclick = async () => {
    const dst = $("#m-name").value.trim();
    const err = $("#m-err");
    if (!dst || dst === src) return;
    try {
      const res = await api("/api/library/folders", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ src, dst }),
      });
      if (res.folders) state.folders = res.folders;
      if (state.health) state.health.refmod_folders = res.folders || state.health.refmod_folders;
      rememberFolder(dst);
      closeModal();
      if (after) await after(dst);
    } catch (ex) {
      err.classList.remove("hidden");
      err.textContent = ex.message;
    }
  };
}

function hfBadge(m) {
  const hf = m.hf || {};
  if (hf.status === "uploaded") {
    const label = hf.url
      ? `<a href="${escapeHtml(hf.url)}" target="_blank" rel="noopener">on Hub</a>`
      : "on Hub";
    return `<span class="badge ok">${label}</span>`;
  }
  if (hf.status === "stale") return `<span class="badge dirty">Hub stale</span>`;
  return `<span class="badge">not on Hub</span>`;
}

function selectedHubIds(mods) {
  return (mods || []).map((m) => m.id).filter((id) => id && state.hubSelected[id]);
}

function setHubSelected(id, on) {
  if (on) state.hubSelected[id] = true;
  else delete state.hubSelected[id];
}

function hubDockHtml() {
  if (!state.hubUploading && !state.hubError && !(state.hubLog || []).length) return "";
  const log = (state.hubLog || []).map(escapeHtml).join("\n");
  const cls = state.hubError ? "fail" : (state.hubUploading ? "live" : "done");
  const label = state.hubError ? state.hubError : (state.hubUploading ? "Uploading to Hugging Face…" : "Upload finished");
  return `<section class="extract-dock ${cls}" id="hub-viz">
    <div class="extract-status">
      <h3>Hugging Face</h3>
      <div class="label">${escapeHtml(label)}</div>
    </div>
    <div class="extract-logbox">
      <div class="extract-meta">Upload log</div>
      <pre class="log">${log || "starting…"}</pre>
    </div>
  </section>`;
}

async function uploadSelected(mods, after) {
  const ids = selectedHubIds(mods);
  if (!ids.length) {
    alert("Select one or more RefMods to upload.");
    return;
  }
  const h = state.health || {};
  if (!h.hub_ready) {
    alert("Set a Hugging Face token and repo in Setup first.");
    return;
  }
  state.hubUploading = true;
  state.hubError = null;
  state.hubLog = ["starting…"];
  if (after) after();
  try {
    await api("/api/huggingface/upload", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ids }),
    });
  } catch (err) {
    state.hubUploading = false;
    state.hubError = err.message;
    if (after) after();
    return;
  }
  const tick = async () => {
    const st = await api("/api/huggingface/status");
    state.hubLog = st.log || [];
    if (st.status === "running") {
      if (after) after();
      setTimeout(tick, 400);
      return;
    }
    state.hubUploading = false;
    if (st.status === "error") state.hubError = st.error || "upload failed";
    if (after) await after(true);
  };
  setTimeout(tick, 300);
}

function bindHubChecks(mods, redraw) {
  $$("[data-hub-id]").forEach((box) => {
    box.onchange = () => {
      setHubSelected(box.dataset.hubId, box.checked);
      const n = selectedHubIds(mods).length;
      const btn = $("#btn-hub-upload");
      if (btn) btn.textContent = n ? `Upload selected (${n})` : "Upload selected";
    };
  });
  $$("[data-fold-select]").forEach((box) => {
    box.onclick = (e) => e.stopPropagation();
    box.onchange = (e) => {
      e.stopPropagation();
      const folder = box.dataset.foldSelect === "__root__" ? "" : box.dataset.foldSelect;
      mods.filter((m) => (m.subfolder || "") === folder).forEach((m) => {
        if (m.id) setHubSelected(m.id, box.checked);
      });
      redraw();
    };
  });
  const upload = $("#btn-hub-upload");
  if (upload) upload.onclick = () => uploadSelected(mods, async (reload) => {
    if (reload) {
      if (libraryPane() === "studio") {
        await loadStudio();
        renderStudio();
      } else {
        await loadLibrary();
        renderLibrary();
      }
      return;
    }
    redraw();
  });
}

function loaderName(d) {
  const sub = (d.subfolder || "").replace(/\\/g, "/").replace(/^\/+|\/+$/g, "");
  return sub ? `${sub}/${d.mod_name}` : d.mod_name;
}

function datasetFolders() {
  const used = (state.datasets || []).map((d) => d.subfolder).filter(Boolean);
  return [...new Set([...knownFolders(), ...used])].sort((a, b) => a.localeCompare(b));
}

function datasetInFolder(d, filter) {
  const folder = d.subfolder || "";
  if (!filter) return true;
  if (filter === "__root__") return !folder;
  return folder === filter || folder.startsWith(`${filter}/`);
}

function renderDatasets() {
  const q = state.query.trim().toLowerCase();
  const filter = state.datasetFolderFilter || "";
  const list = state.datasets.filter((d) => {
    if (!datasetInFolder(d, filter)) return false;
    const blob = `${d.display_name} ${d.slug} ${d.mod_name} ${d.subfolder || ""} ${d.notes || ""}`.toLowerCase();
    return !q || blob.includes(q);
  });
  const cards = list.map((d) => {
    const cover = d.cover
      ? `style="background-image:url('/api/datasets/${encodeURIComponent(d.slug)}/files/${encodeURIComponent(d.cover)}/thumb')"`
      : "";
    const dirty = d.dirty ? `<span class="badge dirty">updated</span>` : (d.last_extract ? `<span class="badge ok">extracted</span>` : `<span class="badge">new</span>`);
    const dim = pixelSize(d.cover_size) || pixelSize((d.items || []).find((it) => it.file === d.cover));
    const note = notesPreview(d.notes);
    return `<article class="card" data-open="${escapeHtml(d.slug)}">
      <div class="card-cover ${d.cover ? "" : "empty"}" ${cover}>${dim ? `<span class="card-size">${escapeHtml(dim)}</span>` : `<span></span>`}${dirty}</div>
      <div class="card-body">
        <h3>${escapeHtml(d.display_name)}</h3>
        <div class="meta">${d.image_count} stills${d.video_count ? ` · ${d.video_count} clips` : ""} · ${escapeHtml(d.concept_type)}${d.subfolder ? ` · ${escapeHtml(d.subfolder)}/` : ""}</div>
        <div class="mono">${escapeHtml(d.subfolder ? `${d.subfolder}/${d.mod_name}` : d.mod_name)}</div>
        ${note ? `<div class="meta notes-preview">${escapeHtml(note)}</div>` : ""}
      </div>
    </article>`;
  }).join("");
  const folders = datasetFolders();
  const folderOpts = [
    `<option value="" ${!filter ? "selected" : ""}>all folders</option>`,
    `<option value="__root__" ${filter === "__root__" ? "selected" : ""}>(root)</option>`,
    ...folders.map((f) => `<option value="${escapeHtml(f)}" ${filter === f ? "selected" : ""}>${escapeHtml(f)}</option>`),
  ].join("");
  const empty = state.datasetsLoading && !state.datasets.length
    ? `<p class="empty">Loading datasets…</p>`
    : (state.datasets.length
      ? `<p class="empty">No datasets in this folder${q ? " match the search" : ""}.</p>`
      : `<p class="empty">No datasets yet. Create one and drop 8–20 stills. Extract when the sheet looks right. Come back and re-extract when you swap a shot.</p>`);
  $("#app").innerHTML = `
    <div class="toolbar">
      <h2>Datasets</h2>
      ${busyHtml(state.datasetsLoading, Boolean(state.datasets.length))}
      <input class="search" id="search" placeholder="Search" value="${escapeHtml(state.query)}">
      <select id="dataset-folder-filter" class="search" style="max-width:220px">${folderOpts}</select>
      <button class="btn" id="btn-import">Import folder</button>
      <button class="btn" id="btn-split">Split sheet</button>
      <button class="btn primary" id="btn-new">New dataset</button>
    </div>
    ${list.length ? `<div class="cards">${cards}</div>` : empty}
  `;
  $("#search").addEventListener("input", (e) => { state.query = e.target.value; renderDatasets(); });
  const folderSel = $("#dataset-folder-filter");
  if (folderSel) folderSel.onchange = () => { state.datasetFolderFilter = folderSel.value; renderDatasets(); };
  $("#btn-new").onclick = promptNewDataset;
  $("#btn-import").onclick = promptImport;
  $("#btn-split").onclick = promptSplitSheet;
  $$(".card").forEach((el) => el.onclick = () => { location.hash = `#/datasets/${el.dataset.open}`; });
}

function promptNewDataset() {
  openModal(`
    <h3>New dataset</h3>
    <label>Name<input id="m-name" type="text" placeholder="feliciaday"></label>
    <label>Display name<input id="m-display" type="text" placeholder="Felicia Day"></label>
    ${folderSelectHtml("m-folder", "identity")}
    <p class="meta">Extracts land in <span class="mono">models/refmods/&lt;folder&gt;/</span>. Pick an existing folder or New folder.</p>
    <p class="warn hidden" id="m-err"></p>
    <div class="row">
      <button class="btn" id="m-cancel">Cancel</button>
      <button class="btn primary" id="m-ok">Create</button>
    </div>
  `);
  bindFolderSelect("m-folder");
  $("#m-cancel").onclick = closeModal;
  $("#m-ok").onclick = async () => {
    const name = $("#m-name").value.trim();
    const display_name = $("#m-display").value.trim();
    const err = $("#m-err");
    if (!name) return;
    try {
      const subfolder = await resolveFolderSelect("m-folder");
      const created = await api("/api/datasets", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, display_name: display_name || name, subfolder }),
      });
      closeModal();
      location.hash = `#/datasets/${created.slug}`;
    } catch (ex) {
      if (err) {
        err.classList.remove("hidden");
        err.textContent = ex.message;
      } else {
        alert(ex.message);
      }
    }
  };
}

function fileStem(name) {
  return String(name || "sheet").replace(/\.[^.]+$/, "") || "sheet";
}

function promptSplitSheet() {
  openModal(`
    <h3>Split sheet</h3>
    <p class="meta">Drop a character sheet, pick the cut set that looks right, then create the dataset. Click a box to skip that panel. Multiple sheets create one dataset each.</p>
    <label>Name<input id="m-name" type="text" placeholder="feliciaday"></label>
    <label>Display name<input id="m-display" type="text" placeholder="Felicia Day"></label>
    ${folderSelectHtml("m-folder", "identity")}
    <div class="dropzone" id="m-drop">Drop sheet(s) here, or click to choose</div>
    <input id="m-files" type="file" accept="image/*" multiple hidden>
    <div class="meta" id="m-file-list"></div>
    <div id="m-preview-wrap" class="hidden">
      <div class="row" id="m-sheet-tabs"></div>
      <div class="split-stage"><div class="split-frame" id="m-frame">
        <img id="m-preview" alt="Sheet preview">
        <canvas id="m-overlay"></canvas>
      </div></div>
      <div class="split-methods" id="m-methods"></div>
      <p class="meta" id="m-cut-meta"></p>
    </div>
    <p class="warn hidden" id="m-err"></p>
    <div class="row">
      <button class="btn" id="m-cancel">Cancel</button>
      <button class="btn primary" id="m-ok" disabled>Use these cuts</button>
    </div>
  `, "wide");
  bindFolderSelect("m-folder");
  const drop = $("#m-drop");
  const fileIn = $("#m-files");
  const list = $("#m-file-list");
  const err = $("#m-err");
  const nameEl = $("#m-name");
  const displayEl = $("#m-display");
  const wrap = $("#m-preview-wrap");
  const splitState = {
    sheets: [],
    index: 0,
    methodId: [],
    skipped: [],
  };

  const currentSheet = () => splitState.sheets[splitState.index] || null;
  const currentMethod = () => {
    const sheet = currentSheet();
    if (!sheet) return null;
    const id = splitState.methodId[splitState.index];
    return (sheet.methods || []).find((m) => m.id === id) || sheet.methods[0] || null;
  };
  const showErr = (msg) => {
    err.textContent = msg || "";
    err.classList.toggle("hidden", !msg);
  };
  const enabledCount = (method, skipped) => {
    if (!method) return 0;
    return method.boxes.filter((_, i) => !skipped.has(i)).length;
  };

  const drawOverlay = () => {
    const sheet = currentSheet();
    const method = currentMethod();
    const canvas = $("#m-overlay");
    const img = $("#m-preview");
    if (!sheet || !method || !canvas || !img || !img.naturalWidth) return;
    const dpr = window.devicePixelRatio || 1;
    const cw = img.clientWidth;
    const ch = img.clientHeight;
    canvas.width = Math.max(1, Math.round(cw * dpr));
    canvas.height = Math.max(1, Math.round(ch * dpr));
    canvas.style.width = `${cw}px`;
    canvas.style.height = `${ch}px`;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cw, ch);
    const sx = cw / sheet.width;
    const sy = ch / sheet.height;
    const skipped = splitState.skipped[splitState.index] || new Set();
    method.boxes.forEach((box, i) => {
      const [x0, y0, x1, y1] = box;
      const x = x0 * sx;
      const y = y0 * sy;
      const bw = (x1 - x0) * sx;
      const bh = (y1 - y0) * sy;
      const off = skipped.has(i);
      const pw = Math.max(1, Math.round(x1 - x0));
      const ph = Math.max(1, Math.round(y1 - y0));
      if (off) {
        ctx.fillStyle = "rgba(17,15,12,0.5)";
        ctx.fillRect(x, y, bw, bh);
      }
      ctx.strokeStyle = off ? "#6a5d4a" : "#d7a44a";
      ctx.lineWidth = off ? 1.5 : 2.5;
      ctx.setLineDash(off ? [5, 4] : []);
      ctx.strokeRect(x + 1, y + 1, Math.max(0, bw - 2), Math.max(0, bh - 2));
      ctx.setLineDash([]);
      const idx = String(i + 1).padStart(2, "0");
      const dim = `${pw}×${ph}`;
      const label = bw >= 88 ? `${idx}  ${dim}` : idx;
      ctx.font = "11px Consolas, ui-monospace, monospace";
      ctx.textBaseline = "top";
      const tw = ctx.measureText(label).width;
      ctx.fillStyle = off ? "#3a3429" : "#d7a44a";
      ctx.fillRect(x + 4, y + 4, tw + 8, 16);
      ctx.fillStyle = off ? "#9a8f7c" : "#1a1408";
      ctx.fillText(label, x + 8, y + 6);
      if (label === idx && bh > 36) {
        const dw = ctx.measureText(dim).width;
        ctx.fillStyle = off ? "#3a3429" : "#d7a44a";
        ctx.fillRect(x + 4, y + 22, dw + 8, 16);
        ctx.fillStyle = off ? "#9a8f7c" : "#1a1408";
        ctx.fillText(dim, x + 8, y + 24);
      }
    });
    const kept = enabledCount(method, skipped);
    const hint = method.hint ? ` · ${method.hint}` : "";
    $("#m-cut-meta").textContent = `${kept} of ${method.boxes.length} cuts${hint} · click a box to skip it`;
    $("#m-ok").disabled = kept < 1;
  };

  const renderMethods = () => {
    const sheet = currentSheet();
    const box = $("#m-methods");
    if (!sheet) { box.innerHTML = ""; return; }
    const selected = splitState.methodId[splitState.index];
    box.innerHTML = sheet.methods.map((m) => {
      const active = m.id === selected ? " active" : "";
      return `<button type="button" class="btn small${active}" data-method="${escapeHtml(m.id)}" title="${escapeHtml(m.hint)}">${escapeHtml(m.label)} · ${m.count}</button>`;
    }).join("");
    $$("[data-method]", box).forEach((btn) => {
      btn.onclick = () => {
        splitState.methodId[splitState.index] = btn.dataset.method;
        splitState.skipped[splitState.index] = new Set();
        renderMethods();
        drawOverlay();
      };
    });
  };

  const renderSheetTabs = () => {
    const tabs = $("#m-sheet-tabs");
    if (splitState.sheets.length < 2) { tabs.innerHTML = ""; return; }
    tabs.innerHTML = splitState.sheets.map((s, i) => {
      const active = i === splitState.index ? " active" : "";
      return `<button type="button" class="btn small${active}" data-sheet="${i}">${escapeHtml(s.filename)}</button>`;
    }).join("");
    $$("[data-sheet]", tabs).forEach((btn) => {
      btn.onclick = () => {
        splitState.index = Number(btn.dataset.sheet);
        $("#m-preview").src = currentSheet().preview;
        renderSheetTabs();
        renderMethods();
      };
    });
  };

  const showSheet = () => {
    const sheet = currentSheet();
    if (!sheet) return;
    wrap.classList.remove("hidden");
    const img = $("#m-preview");
    img.onload = () => drawOverlay();
    img.src = sheet.preview;
    renderSheetTabs();
    renderMethods();
  };

  const pickDefaultMethod = (sheet) => {
    const good = (sheet.methods || []).find((m) => m.count >= 2);
    return (good || sheet.methods[0] || {}).id || "auto";
  };

  const runPreview = async () => {
    const files = [...fileIn.files];
    if (!files.length) {
      wrap.classList.add("hidden");
      $("#m-ok").disabled = true;
      return;
    }
    showErr("");
    $("#m-ok").disabled = true;
    $("#m-cut-meta").textContent = "Finding cuts…";
    wrap.classList.remove("hidden");
    const body = new FormData();
    files.forEach((f) => body.append("files", f));
    try {
      const res = await api("/api/datasets/split-sheet/preview", { method: "POST", body });
      splitState.sheets = res.sheets || [];
      splitState.index = 0;
      splitState.methodId = splitState.sheets.map(pickDefaultMethod);
      splitState.skipped = splitState.sheets.map(() => new Set());
      if (!splitState.sheets.length) {
        showErr("No cuts found.");
        wrap.classList.add("hidden");
        return;
      }
      showSheet();
    } catch (ex) {
      wrap.classList.add("hidden");
      showErr(ex.message);
    }
  };

  const describeFiles = () => {
    const files = [...fileIn.files];
    list.textContent = files.length === 1
      ? files[0].name
      : files.length
        ? `${files.length} sheets — first uses the name above, others use the filename`
        : "";
    if (files.length && !nameEl.value.trim()) {
      nameEl.value = fileStem(files[0].name);
      if (!displayEl.value.trim()) displayEl.value = fileStem(files[0].name);
    }
    runPreview();
  };
  drop.onclick = () => fileIn.click();
  fileIn.onchange = describeFiles;
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => {
    e.preventDefault();
    drop.classList.add("hot");
  }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => {
    e.preventDefault();
    drop.classList.remove("hot");
  }));
  drop.addEventListener("drop", (e) => {
    const incoming = e.dataTransfer && e.dataTransfer.files;
    if (!incoming || !incoming.length) return;
    const dt = new DataTransfer();
    [...incoming].forEach((f) => dt.items.add(f));
    fileIn.files = dt.files;
    describeFiles();
  });
  $("#m-overlay").addEventListener("click", (ev) => {
    const sheet = currentSheet();
    const method = currentMethod();
    const canvas = $("#m-overlay");
    if (!sheet || !method || !canvas) return;
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const px = (ev.clientX - rect.left) / rect.width * sheet.width;
    const py = (ev.clientY - rect.top) / rect.height * sheet.height;
    let hit = -1;
    for (let i = method.boxes.length - 1; i >= 0; i -= 1) {
      const [x0, y0, x1, y1] = method.boxes[i];
      if (px >= x0 && px < x1 && py >= y0 && py < y1) { hit = i; break; }
    }
    if (hit < 0) return;
    const skipped = splitState.skipped[splitState.index];
    if (skipped.has(hit)) skipped.delete(hit);
    else skipped.add(hit);
    drawOverlay();
  });
  window.addEventListener("resize", drawOverlay);
  const finish = () => {
    window.removeEventListener("resize", drawOverlay);
    closeModal();
  };
  $("#m-cancel").onclick = finish;
  $("#m-ok").onclick = async () => {
    const files = [...fileIn.files];
    if (!files.length) {
      showErr("Choose a character sheet first.");
      return;
    }
    if (!splitState.sheets.length) {
      showErr("Wait for the cut preview, or drop the sheet again.");
      return;
    }
    const plans = splitState.sheets.map((sheet, i) => {
      const id = splitState.methodId[i];
      const method = (sheet.methods || []).find((m) => m.id === id) || sheet.methods[0];
      const skipped = splitState.skipped[i] || new Set();
      const boxes = (method.boxes || []).filter((_, idx) => !skipped.has(idx));
      return { filename: sheet.filename, boxes };
    });
    if (plans.some((p) => !p.boxes.length)) {
      showErr("Each sheet needs at least one cut.");
      return;
    }
    const name = nameEl.value.trim() || fileStem(files[0].name);
    const display_name = displayEl.value.trim();
    let subfolder = "";
    try {
      subfolder = await resolveFolderSelect("m-folder");
    } catch (ex) {
      showErr(ex.message);
      return;
    }
    const body = new FormData();
    body.append("name", name);
    body.append("display_name", display_name);
    body.append("subfolder", subfolder);
    body.append("plans", JSON.stringify(plans));
    files.forEach((f) => body.append("files", f));
    $("#m-ok").disabled = true;
    $("#m-ok").textContent = "Splitting…";
    showErr("");
    try {
      const res = await api("/api/datasets/split-sheet", { method: "POST", body });
      const created = (res.datasets || [])[0];
      window.removeEventListener("resize", drawOverlay);
      closeModal();
      if (created && created.slug) location.hash = `#/datasets/${created.slug}`;
      else location.hash = "#/datasets";
    } catch (ex) {
      showErr(ex.message);
      $("#m-ok").disabled = false;
      $("#m-ok").textContent = "Use these cuts";
    }
  };
}

function promptImport() {
  openModal(`
    <h3>Import folder</h3>
    <label>Folder${pickRow("m-path", "", "folder", { title: "Dataset folder", placeholder: "Choose a folder of stills" })}</label>
    <label>Slug (optional)<input id="m-slug" type="text"></label>
    ${folderSelectHtml("m-folder", "identity")}
    <label>Mode
      <select id="m-mode">
        <option value="copy">Copy into studio datasets</option>
        <option value="link">Link in place (AI Toolkit folder)</option>
      </select>
    </label>
    <label class="row"><input id="m-rec" type="checkbox"> Recursive</label>
    <div class="row">
      <button class="btn" id="m-cancel">Cancel</button>
      <button class="btn primary" id="m-ok">Import</button>
    </div>
  `);
  bindPickers();
  bindFolderSelect("m-folder");
  $("#m-cancel").onclick = closeModal;
  $("#m-ok").onclick = async () => {
    const path = $("#m-path").value.trim();
    if (!path) return;
    try {
      const subfolder = await resolveFolderSelect("m-folder");
      const created = await api("/api/datasets/import", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          path,
          slug: $("#m-slug").value.trim() || null,
          mode: $("#m-mode").value,
          recursive: $("#m-rec").checked,
          subfolder,
        }),
      });
      closeModal();
      location.hash = `#/datasets/${created.slug}`;
    } catch (ex) {
      alert(ex.message);
    }
  };
}

function countWarn(d) {
  const n = d.image_count || 0;
  const identity = d.concept_type === "identity";
  if (identity && (n < 8 || n > 20)) return `<div class="warn">Identity sets work best at 8–20 stills (now ${n}).</div>`;
  if (!identity && n && (n < 6 || n > 12)) return `<div class="warn">Booster / concept sets usually want 6–12 targeted stills (now ${n}).</div>`;
  return "";
}

function renderDataset() {
  const d = state.dataset;
  if (!d) { $("#app").innerHTML = `<p class="empty">Dataset not found.</p>`; return; }
  if (state.clean) {
    renderClean(d);
    return;
  }
  const frames = sheetRows(d).map(({ it: item, i }) => {
    const src = `/api/datasets/${encodeURIComponent(d.slug)}/files/${encodeURIComponent(item.file)}/thumb?b=${item.bytes || 0}`;
    const media = item.kind === "image"
      ? `<img src="${src}" alt="" data-file="${escapeHtml(item.file)}">`
      : `<div class="clip">CLIP</div>`;
    const mark = item.clean === "flagged" ? " · mark" : item.clean === "cleaned" ? " · cleaned" : "";
    const isCover = item.kind === "image" && d.cover && item.file === d.cover;
    const isPick = !!(item.quality && item.quality.picked);
    const cls = `${item.enabled ? "" : "disabled"} ${item.clean === "flagged" ? "flagged" : ""} ${item.clean === "cleaned" ? "cleaned" : ""} ${isCover ? "is-cover" : ""} ${isPick ? "is-pick" : ""}`;
    const coverMark = isCover ? `<span class="cover-mark">cover</span>` : "";
    const coverCheck = item.kind === "image"
      ? `<label class="check"><input type="checkbox" class="cv" ${isCover ? "checked" : ""}> cover</label>`
      : "";
    const dim = pixelSize(item);
    const dbg = (item.quality && item.quality.debug) || {};
    const shotTag = [scaleShort(dbg.scale), poseShort(dbg.pose_bin)].filter(Boolean).join(" ");
    const sizeLabel = [dim || (item.kind === "video" && item.bytes ? fmtBytes(item.bytes) : ""), shotTag].filter(Boolean).join(" · ");
    const q = item.quality;
    const qScore = qualityScore(item);
    const qChip = q && item.kind === "image"
      ? `<button type="button" class="q-chip ${qualityBand(qScore)}" data-qfile="${escapeHtml(item.file)}" title="${escapeHtml((q.reason || "") + (qScore == null ? "" : ` · ${qScore}`))}">${qScore == null ? "—" : Math.round(qScore)}</button>`
      : "";
    return `<article class="frame ${cls}" draggable="${state.sheetSort ? "false" : "true"}" data-file="${escapeHtml(item.file)}">
      <span class="frame-num">${String(i + 1).padStart(2, "0")}${item.low_res ? " · low-res" : ""}${mark}</span>
      ${isPick ? `<span class="pick-mark">pick</span>` : ""}
      ${coverMark}
      <div class="frame-media">
        ${media}
        ${sizeLabel ? `<span class="frame-size">${escapeHtml(sizeLabel)}</span>` : ""}
        ${qChip}
      </div>
      <div class="frame-tools">
        <div class="row">
          <label class="check"><input type="checkbox" class="en" ${item.enabled ? "checked" : ""}> use</label>
          ${coverCheck}
        </div>
        <button class="btn small danger del">×</button>
      </div>
    </article>`;
  }).join("");
  const ready = state.health && state.health.ready;
  const extractLabel = d.dirty && d.last_extract ? "Re-extract RefMod" : "Extract RefMod";
  const flagged = (d.items || []).filter((it) => it.clean === "flagged").length;
  const cleaned = (d.items || []).filter((it) => it.clean === "cleaned" || it.has_original).length;
  const scored = (d.items || []).filter((it) => it.kind === "image" && it.quality && typeof it.quality.score === "number");
  const qHi = scored.filter((it) => it.quality.score >= 20).length;
  const qMid = scored.filter((it) => it.quality.score >= 8 && it.quality.score < 20).length;
  const qLo = scored.filter((it) => it.quality.score < 8).length;
  const mix = poseMix(d.items);
  $("#app").innerHTML = `
  <div class="dataset-page">
    <div class="toolbar">
      <a href="#/datasets">← Datasets</a>
      <h2>${escapeHtml(d.display_name)}</h2>
      ${d.dirty ? `<span class="badge dirty">updated — re-extract</span>` : (d.last_extract ? `<span class="badge ok">in sync</span>` : "")}
      ${scored.length ? `<button class="btn small${state.sheetSort === "quality" ? " primary" : ""}" id="btn-q-sort">Sort by quality</button>` : ""}
      ${mix.picks.length ? `<button class="btn small${state.sheetSort === "picks" ? " primary" : ""}" id="btn-q-picks">Sort by picks</button>` : ""}
    </div>
    <div class="layout">
      <section>
        <div class="dropzone" id="drop">Drop stills here, or click to add files</div>
        <input id="file-in" type="file" accept="image/*,video/*" multiple hidden>
        ${d.items.length ? `<div class="sheet-bar">
          <button class="btn small" id="btn-use-all">Select all</button>
          <button class="btn small" id="btn-use-none">Select none</button>
          <span class="meta" id="sheet-use-count">${(d.items || []).filter((it) => it.enabled).length} of ${d.items.length} in use</span>
        </div>
        <div class="sheet" id="sheet">${frames}</div>` : `<p class="empty">Empty sheet. Drop 8–20 stills. First image anchors the canvas for the rest.</p>`}
      </section>
      <aside class="side">
        <h3>Extract</h3>
        <label class="field">Display name<input id="f-display" type="text" value="${escapeHtml(d.display_name)}"></label>
        <label class="field">Description <span class="meta">(stored on the mod)</span><textarea id="f-desc">${escapeHtml(d.description || d.display_name || "")}</textarea></label>
        <label class="field">Notes <span class="meta">(yours only — not on the mod, does not re-extract)</span><textarea id="f-notes" placeholder="Shooting notes, source, what to try next…">${escapeHtml(d.notes || "")}</textarea></label>
        <label class="field">Concept<select id="f-concept">${conceptOptions(d.concept_type)}</select></label>
        <label class="field">Mod name<input id="f-mod" type="text" value="${escapeHtml(d.mod_name)}"></label>
        ${folderSelectHtml("f-sub", d.subfolder || "")}
        <div class="row folder-actions">
          ${d.subfolder ? `<button class="btn small" id="btn-rename-folder">Rename folder</button>` : ""}
        </div>
        <div class="meta">ComfyUI lists this as <span class="mono">${escapeHtml(loaderName(d))}</span></div>
        <div class="meta">List thumbnail: <span class="mono">${escapeHtml(d.cover || "first still")}</span> — tick Cover on a still to change</div>
        ${countWarn(d)}
        <details class="adv">
          <summary>Advanced extract</summary>
          <label class="field">Mode
            <select id="f-mode">
              <option value="encode" ${d.extract.mode === "encode" ? "selected" : ""}>Full Reference (encode)</option>
              <option value="training" ${d.extract.mode === "training" ? "selected" : ""}>Compressed Reference (training)</option>
            </select>
          </label>
          <label class="field">Resolution<input id="f-res" type="number" value="${d.extract.resolution}"></label>
          <label class="field">Max tokens<input id="f-tokens" type="number" value="${d.extract.max_tokens}"></label>
          <label class="field"${d.extract.mode === "encode" ? " hidden" : ""}>Pool<input id="f-pool" type="number" value="${d.extract.pool}"></label>
          <label class="field"${d.extract.mode === "encode" ? " hidden" : ""}>Refinement steps<input id="f-id" type="number" value="${d.extract.identity}"></label>
          <label class="field">Multiplier<input id="f-mult" type="number" value="${d.extract.multiplier}"></label>
          <label class="field">Device
            <select id="f-device">
              <option ${d.extract.device === "auto" ? "selected" : ""}>auto</option>
              <option ${d.extract.device === "cuda" ? "selected" : ""}>cuda</option>
              <option ${d.extract.device === "cpu" ? "selected" : ""}>cpu</option>
            </select>
          </label>
        </details>
        <div class="checks">
          <label class="check"><input id="f-over" type="checkbox"> overwrite</label>
          <label class="check"><input id="f-bump" type="checkbox"> bump version</label>
        </div>
        <button class="btn primary" id="btn-extract" ${ready && !state.extracting ? "" : "disabled"}>${state.extracting ? "Extracting…" : extractLabel}</button>
        <div class="row">
          <button class="btn small" id="btn-quality" ${state.qualityBusy ? "disabled" : ""}>${state.qualityBusy ? "Checking…" : "Check quality"}</button>
          ${mix.picks.length ? `<button class="btn small" id="btn-q-apply">Use recommended ${mix.picks.length}</button>` : ""}
          <button class="btn small" id="btn-clean">Clean watermarks</button>
          <button class="btn small" id="btn-folder">Open folder</button>
          <button class="btn small danger" id="btn-delete-ds">Delete dataset</button>
        </div>
        ${scored.length ? `<div class="meta">Quality: ${qHi} high · ${qMid} mid · ${qLo} low — chip opens the breakdown.</div>` : ""}
        ${scored.length ? `<div class="meta">Shots: ${mix.scales.closeup} close-up · ${mix.scales.medium} medium · ${mix.scales.full} full-body. Angles: ${["front", "three-quarter-left", "three-quarter-right", "profile-left", "profile-right"].map((k) => mix.poses[k] ? `${poseShort(k)} ${mix.poses[k]}` : "").filter(Boolean).join(" · ") || "—"}${mix.picks.length ? `. Recommended ${mix.picks.length} (gold pick) spread across angles.` : ""}</div>` : ""}
        ${mix.crowded.map(([key, c]) => `<div class="warn">Similar poses: ${escapeHtml(key)} × ${c} of ${mix.faced}. The extra copies add little once you have one sharp still from that angle.</div>`).join("")}
        ${(d.quality_advice || []).length ? `<div class="field">To improve this set
          <ul class="advice">${(d.quality_advice || []).map((a) => `<li class="${escapeHtml(a.level || "improve")}">${escapeHtml(a.text)}</li>`).join("")}</ul>
        </div>` : ""}
        ${flagged ? `<div class="warn">${flagged} still${flagged === 1 ? "" : "s"} flagged for marks.</div>` : ""}
        ${cleaned ? `<div class="meta">${cleaned} still${cleaned === 1 ? " has" : "s have"} a kept original — undo from Clean.</div>` : ""}
        <div class="mono path">${escapeHtml(d.folder)}</div>
      </aside>
    </div>
    ${extractPanelHtml()}
  </div>
  `;
  bindDatasetEvents();
  if (state.progress) applyExtractProgress(state.progress);
}

function extractPanelHtml() {
  const log = (state.log || []).map(escapeHtml).join("\n");
  const p = state.progress;
  const show = state.extracting || log || state.result || state.error;
  if (!show) return "";
  const cls = state.error ? "fail" : (state.result ? "done" : "");
  const label = state.error ? state.error : (p && p.label) || (state.extracting ? "Starting…" : "Extract");
  const pct = state.error ? 100 : (p && p.percent) || (state.result ? 100 : 0);
  const now = (p && p.current) ? p.current : "";
  const result = state.result ? `<div class="result">
      <div><b>${escapeHtml(state.result.mod_name)}</b> · ${state.result.token_count ?? "?"} tokens · ${state.result.mb ?? "?"} MB</div>
      <div class="mono path">${escapeHtml(state.result.path || "")}</div>
    </div>` : "";
  return `<section class="extract-dock ${cls}${state.extracting ? " live" : ""}" id="extract-viz">
      <div class="extract-status">
        <h3>Run</h3>
        <div class="label" id="extract-label">${escapeHtml(label)}</div>
        <div class="extract-track"><div class="extract-fill" id="extract-fill" style="width:${pct}%"></div></div>
        <div class="extract-meta" id="extract-meta">${p ? `${p.encoded || 0} / ${p.total || 0} encoded` : ""}</div>
        <div class="now" id="extract-now">${escapeHtml(now)}</div>
        ${state.error ? `<div class="warn">${escapeHtml(state.error)}</div>` : ""}
        ${result}
      </div>
      <div class="extract-logbox">
        <div class="extract-meta">Extractor log</div>
        <pre class="log" id="extract-log">${log || "starting…"}</pre>
      </div>
    </section>`;
}

function applyExtractProgress(p) {
  if (!p) return;
  const viz = $("#extract-viz");
  const label = $("#extract-label");
  const fill = $("#extract-fill");
  const meta = $("#extract-meta");
  const now = $("#extract-now");
  if (label) label.textContent = p.label || "";
  if (fill) fill.style.width = `${p.percent || 0}%`;
  if (meta) meta.textContent = `${p.encoded || 0} / ${p.total || 0} encoded`;
  if (now) now.textContent = p.current || "";
  if (viz) {
    viz.classList.toggle("done", p.stage === "saved");
    viz.classList.toggle("fail", p.stage === "error");
  }
  const current = (p.current || "").toLowerCase();
  const files = state.extractFiles || [];
  const idx = files.findIndex((f) => f.toLowerCase() === current);
  $$(".frame").forEach((el, i) => {
    const name = (el.dataset.file || "").toLowerCase();
    const done = idx >= 0 ? i < idx : files.slice(0, p.encoded || 0).some((f) => f.toLowerCase() === name);
    el.classList.toggle("encoding", name === current);
    el.classList.toggle("encoded", done && name !== current);
  });
}

function gatherPatch() {
  const current = (state.dataset && state.dataset.extract) || {};
  return {
    display_name: $("#f-display").value,
    description: $("#f-desc").value,
    notes: $("#f-notes") ? $("#f-notes").value : "",
    concept_type: $("#f-concept").value,
    mod_name: $("#f-mod").value,
    subfolder: folderSelectValue("f-sub"),
    extract: {
      mode: $("#f-mode").value,
      resolution: Number($("#f-res").value),
      max_tokens: Number($("#f-tokens").value),
      pool: Number(($("#f-pool") && $("#f-pool").value) || current.pool || 16),
      identity: Number(($("#f-id") && $("#f-id").value) || current.identity || 0),
      multiplier: Number($("#f-mult").value),
      device: $("#f-device").value,
    },
  };
}

async function saveMeta() {
  const d = state.dataset;
  const patch = gatherPatch();
  const sel = $("#f-sub");
  if (sel && sel.value === "__new__") {
    if (!patch.subfolder) throw new Error("name the new folder");
    patch.subfolder = await createFolder(patch.subfolder);
  }
  state.dataset = await api(`/api/datasets/${encodeURIComponent(d.slug)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
}

function showQualityDebug(item) {
  const q = item.quality || {};
  const dbg = q.debug || {};
  const row = (label, value, extra) => `<tr><th>${escapeHtml(label)}</th><td>${escapeHtml(String(value ?? "—"))}</td><td class="meta">${escapeHtml(String(extra ?? ""))}</td></tr>`;
  openModal(`
    <h3>Quality</h3>
    <p class="mono">${escapeHtml(item.file)}</p>
    <p class="meta">${q.score == null ? "—" : q.score} / 100 · ${escapeHtml(q.reason || "")}${dbg.n_faces ? ` · ${dbg.n_faces} face${Number(dbg.n_faces) === 1 ? "" : "s"}` : ""}</p>
    <table class="table">
      <tbody>
        ${row("Sharp", dbg.s_sharp, `Laplacian ${dbg.laplacian_var}`)}
        ${row("Face", dbg.s_face, `IED ${dbg.ied_px}px`)}
        ${row("Light", dbg.s_light, `mean ${dbg.face_mean} · clip ${dbg.clip_frac} · split ${dbg.split}`)}
        ${row("Shot", dbg.scale, poseShort(dbg.pose_bin) || dbg.pose_bin)}
        ${row("Pose", dbg.s_pose, `yaw ${dbg.yaw_deg}° · pitch ${dbg.pitch_deg}°`)}
      </tbody>
    </table>
    <p class="meta">Product of the four 0–1 terms × 100. 20+ is a strong identity still. Rank only — this does not change Use or Cover.</p>
    <div class="row"><button class="btn primary" id="m-ok">OK</button></div>
  `);
  $("#m-ok").onclick = closeModal;
}

async function applyQualityPicks() {
  const d = state.dataset;
  if (!d) return;
  const picks = new Set((d.items || []).filter((it) => it.quality && it.quality.picked).map((it) => it.file));
  if (!picks.size) return;
  if (!confirm(`Use the ${picks.size} recommended stills and uncheck the other images? Clips stay as they are. This marks the dataset updated if you already extracted.`)) return;
  const items = (d.items || []).map((it) => ({
    file: it.file,
    enabled: it.kind === "video" ? !!it.enabled : picks.has(it.file),
  }));
  state.dataset = await api(`/api/datasets/${encodeURIComponent(d.slug)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ items }),
  });
  renderDataset();
}

async function runQuality() {
  const d = state.dataset;
  if (!d) return;
  state.qualityBusy = true;
  renderDataset();
  try {
    const res = await api(`/api/datasets/${encodeURIComponent(d.slug)}/quality`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    state.dataset = res.dataset || state.dataset;
  } catch (err) {
    alert(err.message);
  }
  state.qualityBusy = false;
  renderDataset();
}

async function saveItems() {
  const d = state.dataset;
  const items = $$(".frame").map((el) => ({
    file: el.dataset.file,
    enabled: $(".en", el).checked,
  }));
  state.dataset = await api(`/api/datasets/${encodeURIComponent(d.slug)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ items }),
  });
}

function sheetScrollEl() {
  return document.querySelector(".dataset-page .layout > section");
}

function renderDatasetKeepingScroll() {
  const el = sheetScrollEl();
  const top = el ? el.scrollTop : 0;
  renderDataset();
  const next = sheetScrollEl();
  if (next) next.scrollTop = top;
}

async function setSheetEnabled(on) {
  const d = state.dataset;
  if (!d || !(d.items || []).length) return;
  const items = (d.items || []).map((it) => ({ file: it.file, enabled: on }));
  const el = sheetScrollEl();
  const top = el ? el.scrollTop : 0;
  state.dataset = await api(`/api/datasets/${encodeURIComponent(d.slug)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ items }),
  });
  renderDataset();
  const next = sheetScrollEl();
  if (next) next.scrollTop = top;
}

function bindDatasetEvents() {
  const d = state.dataset;
  const drop = $("#drop");
  const fileIn = $("#file-in");
  drop.onclick = () => fileIn.click();
  const sendFiles = async (files) => {
    if (!files.length) return;
    const body = new FormData();
    [...files].forEach((f) => body.append("files", f));
    state.dataset = await api(`/api/datasets/${encodeURIComponent(d.slug)}/files`, { method: "POST", body });
    renderDataset();
  };
  fileIn.onchange = () => sendFiles(fileIn.files);
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("hot"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("hot"); }));
  drop.addEventListener("drop", (e) => sendFiles(e.dataTransfer.files));

  ["#f-display", "#f-desc", "#f-notes", "#f-mod"].forEach((sel) => {
    const el = $(sel);
    if (el) el.addEventListener("change", () => saveMeta());
  });
  ["#f-concept", "#f-mode", "#f-res", "#f-tokens", "#f-pool", "#f-id", "#f-mult", "#f-device"].forEach((sel) => {
    const el = $(sel);
    if (el) el.addEventListener("change", async () => { await saveMeta(); renderDataset(); });
  });
  bindFolderSelect("f-sub");
  const folderSel = $("#f-sub");
  if (folderSel) folderSel.addEventListener("change", async () => {
    if (folderSel.value === "__new__") return;
    await saveMeta();
    renderDataset();
  });
  const folderNew = $("#f-sub-new");
  if (folderNew) folderNew.addEventListener("change", async () => {
    if (!folderNew.value.trim()) return;
    await saveMeta();
    renderDataset();
  });
  const renameBtn = $("#btn-rename-folder");
  if (renameBtn) renameBtn.onclick = () => promptRenameFolder(d.subfolder, async () => {
    state.dataset = await api(`/api/datasets/${encodeURIComponent(d.slug)}`);
    renderDataset();
  });

  const qBtn = $("#btn-quality");
  if (qBtn) qBtn.onclick = () => runQuality();
  const qSort = $("#btn-q-sort");
  if (qSort) qSort.onclick = () => {
    state.sheetSort = state.sheetSort === "quality" ? "" : "quality";
    renderDataset();
  };
  const qPickSort = $("#btn-q-picks");
  if (qPickSort) qPickSort.onclick = () => {
    state.sheetSort = state.sheetSort === "picks" ? "" : "picks";
    renderDataset();
  };
  const qApply = $("#btn-q-apply");
  if (qApply) qApply.onclick = () => applyQualityPicks();
  const useAll = $("#btn-use-all");
  if (useAll) useAll.onclick = () => setSheetEnabled(true);
  const useNone = $("#btn-use-none");
  if (useNone) useNone.onclick = () => setSheetEnabled(false);
  $$("[data-qfile]").forEach((btn) => {
    btn.onclick = (e) => {
      e.stopPropagation();
      const item = (d.items || []).find((it) => it.file === btn.dataset.qfile);
      if (item) showQualityDebug(item);
    };
  });

  $$(".frame").forEach((frame) => {
    frame.addEventListener("dragstart", (e) => {
      if (state.sheetSort || e.target.closest("textarea, select, input, button")) { e.preventDefault(); return; }
      frame.classList.add("dragging");
      e.dataTransfer.setData("text/plain", frame.dataset.file);
    });
    frame.addEventListener("dragend", () => frame.classList.remove("dragging"));
    frame.addEventListener("dragover", (e) => e.preventDefault());
    frame.addEventListener("drop", async (e) => {
      e.preventDefault();
      const src = e.dataTransfer.getData("text/plain");
      const dst = frame.dataset.file;
      if (!src || src === dst) return;
      const order = $$(".frame").map((el) => el.dataset.file);
      const from = order.indexOf(src);
      const to = order.indexOf(dst);
      order.splice(to, 0, ...order.splice(from, 1));
      const byFile = Object.fromEntries(state.dataset.items.map((it) => [it.file, it]));
      const items = order.map((file) => ({ file, enabled: byFile[file].enabled }));
      state.dataset = await api(`/api/datasets/${encodeURIComponent(d.slug)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ items }),
      });
      renderDataset();
    });
    $(".en", frame).onchange = async () => { await saveItems(); renderDatasetKeepingScroll(); };
    const coverBox = $(".cv", frame);
    if (coverBox) coverBox.onchange = async () => {
      const file = coverBox.checked ? frame.dataset.file : "";
      state.dataset = await api(`/api/datasets/${encodeURIComponent(d.slug)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cover: file }),
      });
      renderDatasetKeepingScroll();
    };
    $(".del", frame).onclick = async (e) => {
      e.stopPropagation();
      if (!confirm(`Remove ${frame.dataset.file}?`)) return;
      state.dataset = await api(`/api/datasets/${encodeURIComponent(d.slug)}/files/${encodeURIComponent(frame.dataset.file)}`, { method: "DELETE" });
      renderDataset();
    };
    const img = $("img", frame);
    if (img) img.onclick = () => {
      const item = (d.items || []).find((it) => it.file === frame.dataset.file);
      const box = $("#lightbox");
      box.classList.remove("hidden");
      box.innerHTML = `<img src="/api/datasets/${encodeURIComponent(d.slug)}/files/${encodeURIComponent(frame.dataset.file)}/media?b=${item && item.bytes ? item.bytes : 0}" alt="">`;
      box.onclick = () => { box.classList.add("hidden"); box.innerHTML = ""; };
    };
  });

  const cleanBtn = $("#btn-clean");
  if (cleanBtn) cleanBtn.onclick = () => startClean();
  $("#btn-folder").onclick = () => api("/api/open-folder", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path: d.folder }),
  });
  $("#btn-delete-ds").onclick = async () => {
    const msg = d.linked ? "Unlink this dataset? The original folder is left in place." : "Delete this dataset folder and its files?";
    if (!confirm(msg)) return;
    await api(`/api/datasets/${encodeURIComponent(d.slug)}`, { method: "DELETE" });
    location.hash = "#/datasets";
  };
  $("#btn-extract").onclick = runExtract;
}

function cleanCurrent() {
  if (!state.clean || !state.clean.items.length) return null;
  return state.clean.items[state.clean.index] || null;
}

function cleanMediaUrl(d, file, kind, bytes) {
  return `/api/datasets/${encodeURIComponent(d.slug)}/files/${encodeURIComponent(file)}/${kind}?b=${bytes || Date.now()}`;
}

async function startClean() {
  const d = state.dataset;
  state.clean = { items: [], index: 0, selected: -1, err: "", busy: true, linked: false, flagged: 0 };
  renderDataset();
  try {
    const scan = await api(`/api/datasets/${encodeURIComponent(d.slug)}/clean/scan`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    state.dataset = scan.dataset || state.dataset;
    const items = (scan.items || []).map((it) => ({
      ...it,
      keep: it.keep || keepFromBoxes(it.boxes || []),
    }));
    const flaggedFirst = items.findIndex((it) => it.flagged);
    state.clean = {
      items,
      index: flaggedFirst >= 0 ? flaggedFirst : 0,
      selected: -1,
      err: "",
      busy: false,
      linked: !!scan.linked,
      flagged: scan.flagged || 0,
    };
  } catch (err) {
    state.clean.busy = false;
    state.clean.err = err.message;
  }
  renderDataset();
}

function renderClean(d) {
  const c = state.clean;
  const item = cleanCurrent();
  const strip = (c.items || []).map((it, i) => {
    const on = i === c.index ? " on" : "";
    const flag = it.flagged ? " flagged" : (it.clean === "cleaned" ? " cleaned" : "");
    const src = cleanMediaUrl(d, it.file, "thumb", it.bytes || i);
    return `<button type="button" class="clean-thumb${on}${flag}" data-idx="${i}" title="${escapeHtml(it.file)}">
      <img src="${src}" alt="">
    </button>`;
  }).join("");
  const keep = item ? keepOf(item) : { x0: 0, y0: 0, x1: 1, y1: 1 };
  const cropped = item && !keepIsFull(keep);
  const anyCropped = (c.items || []).some((it) => !keepIsFull(keepOf(it)));
  const scanHint = (item && item.boxes && item.boxes.length)
    ? `Scan suggested: ${(item.boxes.map((b) => b.edge || "edge").join(", "))}`
    : "Scan found nothing — drag the frame yourself.";
  $("#app").innerHTML = `
    <div class="toolbar">
      <button class="btn" id="btn-clean-exit">← Sheet</button>
      <h2>Clean ${escapeHtml(d.display_name)}</h2>
      <span class="badge">${c.flagged || 0} flagged</span>
    </div>
    <div class="layout">
      <section>
        ${c.busy ? `<p class="empty">Working…</p>` : ""}
        ${c.err ? `<p class="warn">${escapeHtml(c.err)}</p>` : ""}
        ${!c.busy && item ? `<div class="clean-stage"><div class="clean-frame" id="clean-frame">
          <img id="clean-img" alt="" src="${cleanMediaUrl(d, item.file, "media", item.bytes || item.width)}">
          <canvas id="clean-overlay"></canvas>
        </div></div>
        <div class="clean-strip">${strip}</div>` : (!c.busy ? `<p class="empty">No stills to clean.</p>` : "")}
      </section>
      <aside class="side">
        <h3>Crop</h3>
        ${c.linked ? `<div class="warn">Linked dataset — cleaning writes into the source folder. Originals are kept in _originals/.</div>` : ""}
        ${item ? `<div class="meta">${escapeHtml(item.file)}</div>` : ""}
        <div class="clean-res" id="clean-keep-size">${keepSizeHtml(item, keep)}</div>
        <p class="meta">${escapeHtml(scanHint)}</p>
        <p class="meta">Drag the handles on each still. Crop all stills uses the frame you set on each one — it does not copy this still to the others.</p>
        <div class="row">
          <button class="btn small" id="btn-clean-prev">Prev</button>
          <button class="btn small" id="btn-clean-next">Next</button>
        </div>
        <div class="row">
          <button class="btn small" id="btn-clean-reset">Reset to full</button>
          <button class="btn small" id="btn-clean-usescan" ${item && item.boxes && item.boxes.length ? "" : "disabled"}>Use scan</button>
        </div>
        <button class="btn primary" id="btn-clean-apply" ${cropped ? "" : "disabled"}>Crop this still</button>
        <button class="btn" id="btn-clean-all" ${anyCropped ? "" : "disabled"}>Crop all stills</button>
        <button class="btn small" id="btn-clean-copy" ${cropped ? "" : "disabled"}>Copy this crop to all</button>
        <button class="btn" id="btn-clean-disable" ${item ? "" : "disabled"}>Don't use this still</button>
        <button class="btn" id="btn-clean-ignore" ${item ? "" : "disabled"}>Ignore (keep as-is)</button>
        <div class="row">
          <button class="btn small" id="btn-clean-rescan">Scan again</button>
          <button class="btn small" id="btn-clean-undo" ${d.items.some((it) => it.has_original) ? "" : "disabled"}>Undo originals</button>
        </div>
      </aside>
    </div>
  `;
  bindCleanEvents();
}

function bindCleanEvents() {
  const d = state.dataset;
  const c = state.clean;
  const img = $("#clean-img");
  const canvas = $("#clean-overlay");
  $("#btn-clean-exit").onclick = () => { state.clean = null; renderDataset(); };
  $("#btn-clean-rescan").onclick = () => startClean();
  $("#btn-clean-prev").onclick = () => {
    if (!c.items.length) return;
    c.index = (c.index + c.items.length - 1) % c.items.length;
    c.selected = -1;
    renderDataset();
  };
  $("#btn-clean-next").onclick = () => {
    if (!c.items.length) return;
    c.index = (c.index + 1) % c.items.length;
    c.selected = -1;
    renderDataset();
  };
  $$(".clean-thumb").forEach((btn) => {
    btn.onclick = () => {
      c.index = Number(btn.dataset.idx);
      c.selected = -1;
      renderDataset();
    };
  });
  $("#btn-clean-reset").onclick = () => {
    const item = cleanCurrent();
    if (!item) return;
    item.keep = { x0: 0, y0: 0, x1: 1, y1: 1 };
    renderDataset();
  };
  $("#btn-clean-usescan").onclick = () => {
    const item = cleanCurrent();
    if (!item) return;
    item.keep = keepFromBoxes(item.boxes || []);
    renderDataset();
  };
  $("#btn-clean-apply").onclick = () => applyCleanStill("one");
  $("#btn-clean-all").onclick = () => {
    if (!confirm("Crop each still using the frame you set on that still?")) return;
    applyCleanStill("each");
  };
  $("#btn-clean-copy").onclick = () => {
    if (!confirm("Copy this still's crop onto every enabled still? This replaces the frames you set on the others.")) return;
    applyCleanStill("copy");
  };
  $("#btn-clean-disable").onclick = () => applyCleanStill("one", { disable: true });
  $("#btn-clean-ignore").onclick = () => applyCleanStill("one", { ignore: true });
  $("#btn-clean-undo").onclick = async () => {
    if (!confirm("Restore originals for every still that was cleaned?")) return;
    try {
      const out = await api(`/api/datasets/${encodeURIComponent(d.slug)}/clean/undo`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "{}",
      });
      state.dataset = out.dataset;
      await startClean();
    } catch (err) {
      alert(err.message);
    }
  };
  if (img) {
    img.onload = () => drawCleanOverlay();
    if (img.complete) drawCleanOverlay();
  }
  if (canvas) {
    canvas.addEventListener("mousedown", onCleanDragStart);
    canvas.addEventListener("mousemove", onCleanDragMove);
    canvas.addEventListener("mouseup", onCleanDragEnd);
    canvas.addEventListener("mouseleave", onCleanDragEnd);
  }
}

function keepFromBoxes(boxes) {
  let x0 = 0;
  let y0 = 0;
  let x1 = 1;
  let y1 = 1;
  (boxes || []).forEach((box) => {
    if (box.edge === "top") y0 = Math.max(y0, box.y1);
    else if (box.edge === "bottom") y1 = Math.min(y1, box.y0);
    else if (box.edge === "left") x0 = Math.max(x0, box.x1);
    else if (box.edge === "right") x1 = Math.min(x1, box.x0);
  });
  return clampKeep({ x0, y0, x1, y1 });
}

function keepOf(item) {
  if (!item) return { x0: 0, y0: 0, x1: 1, y1: 1 };
  if (item.keep) return clampKeep(item.keep);
  return keepFromBoxes(item.boxes || []);
}

function clampKeep(keep) {
  let x0 = Math.max(0, Math.min(1, keep.x0));
  let y0 = Math.max(0, Math.min(1, keep.y0));
  let x1 = Math.max(0, Math.min(1, keep.x1));
  let y1 = Math.max(0, Math.min(1, keep.y1));
  if (x1 < x0) {
    const t = x0;
    x0 = x1;
    x1 = t;
  }
  if (y1 < y0) {
    const t = y0;
    y0 = y1;
    y1 = t;
  }
  const min = 0.08;
  if (x1 - x0 < min) {
    const mid = (x0 + x1) / 2;
    x0 = Math.max(0, mid - min / 2);
    x1 = Math.min(1, x0 + min);
  }
  if (y1 - y0 < min) {
    const mid = (y0 + y1) / 2;
    y0 = Math.max(0, mid - min / 2);
    y1 = Math.min(1, y0 + min);
  }
  return { x0, y0, x1, y1 };
}

function keepIsFull(keep) {
  return keep.x0 <= 0.002 && keep.y0 <= 0.002 && keep.x1 >= 0.998 && keep.y1 >= 0.998;
}

function keepPixels(item, keep) {
  const srcW = Number(item && item.width) || 0;
  const srcH = Number(item && item.height) || 0;
  const w = Math.max(0, Math.round((keep.x1 - keep.x0) * srcW));
  const h = Math.max(0, Math.round((keep.y1 - keep.y0) * srcH));
  return { w, h, srcW, srcH, short: Math.min(w, h) };
}

function keepSizeHtml(item, keep) {
  if (!item) return "";
  const { w, h, srcW, srcH, short } = keepPixels(item, keep);
  if (!srcW || !srcH) return "";
  const cropped = w !== srcW || h !== srcH;
  const low = short > 0 && short < 1024;
  return `<b>${w}×${h}</b>${cropped ? `<span class="meta">from ${srcW}×${srcH}</span>` : `<span class="meta">full still</span>`}${low ? `<span class="badge dirty">short edge ${short}</span>` : ""}`;
}

function updateKeepSizeLabel(item, keep) {
  const el = $("#clean-keep-size");
  if (el) el.innerHTML = keepSizeHtml(item, keep);
}

function drawSizeChip(ctx, boxX, boxY, boxW, boxH, text, warn) {
  ctx.font = "12px Cascadia Mono, Consolas, ui-monospace, monospace";
  ctx.textBaseline = "top";
  const padX = 7;
  const padY = 5;
  const line = 12;
  const tw = ctx.measureText(text).width;
  const chipW = tw + padX * 2;
  const chipH = line + padY * 2;
  let x = boxX + 8;
  let y = boxY + 8;
  if (chipW + 12 > boxW) x = boxX + Math.max(4, (boxW - chipW) / 2);
  else if (x + chipW > boxX + boxW - 4) x = Math.max(boxX + 4, boxX + boxW - chipW - 4);
  if (y + chipH > boxY + boxH - 4 && boxY >= chipH + 6) y = boxY - chipH - 4;
  ctx.fillStyle = warn ? "rgba(196, 92, 58, 0.94)" : "rgba(215, 164, 74, 0.95)";
  ctx.fillRect(x, y, chipW, chipH);
  ctx.fillStyle = warn ? "#f2eadb" : "#1a1408";
  ctx.fillText(text, x + padX, y + padY);
}

function cleanCanvasPoint(ev) {
  const canvas = $("#clean-overlay");
  if (!canvas) return null;
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return null;
  return {
    x: ev.clientX - rect.left,
    y: ev.clientY - rect.top,
    nx: (ev.clientX - rect.left) / rect.width,
    ny: (ev.clientY - rect.top) / rect.height,
    cw: rect.width,
    ch: rect.height,
  };
}

function keepHandles(keep, cw, ch) {
  const x = keep.x0 * cw;
  const y = keep.y0 * ch;
  const r = (keep.x1 - keep.x0) * cw;
  const b = (keep.y1 - keep.y0) * ch;
  return [
    { id: "nw", x, y },
    { id: "n", x: x + r / 2, y },
    { id: "ne", x: x + r, y },
    { id: "e", x: x + r, y: y + b / 2 },
    { id: "se", x: x + r, y: y + b },
    { id: "s", x: x + r / 2, y: y + b },
    { id: "sw", x, y: y + b },
    { id: "w", x, y: y + b / 2 },
  ];
}

function hitKeepHandle(pt, keep) {
  const hs = 10;
  const handles = keepHandles(keep, pt.cw, pt.ch);
  for (let i = 0; i < handles.length; i += 1) {
    const h = handles[i];
    if (Math.abs(pt.x - h.x) <= hs && Math.abs(pt.y - h.y) <= hs) return h.id;
  }
  const x = keep.x0 * pt.cw;
  const y = keep.y0 * pt.ch;
  const r = keep.x1 * pt.cw;
  const b = keep.y1 * pt.ch;
  if (pt.x >= x && pt.x <= r) {
    if (Math.abs(pt.y - y) <= hs) return "n";
    if (Math.abs(pt.y - b) <= hs) return "s";
  }
  if (pt.y >= y && pt.y <= b) {
    if (Math.abs(pt.x - x) <= hs) return "w";
    if (Math.abs(pt.x - r) <= hs) return "e";
  }
  if (pt.nx > keep.x0 && pt.nx < keep.x1 && pt.ny > keep.y0 && pt.ny < keep.y1) return "move";
  return null;
}

function cursorForHandle(id) {
  if (!id) return "crosshair";
  if (id === "move") return "move";
  if (id === "n" || id === "s") return "ns-resize";
  if (id === "e" || id === "w") return "ew-resize";
  if (id === "nw" || id === "se") return "nwse-resize";
  return "nesw-resize";
}

function applyHandleDrag(orig, id, nx, ny, start) {
  const k = { ...orig };
  if (id === "move") {
    const dx = nx - start.nx;
    const dy = ny - start.ny;
    const bw = orig.x1 - orig.x0;
    const bh = orig.y1 - orig.y0;
    k.x0 = orig.x0 + dx;
    k.y0 = orig.y0 + dy;
    k.x1 = k.x0 + bw;
    k.y1 = k.y0 + bh;
    if (k.x0 < 0) {
      k.x0 = 0;
      k.x1 = bw;
    }
    if (k.y0 < 0) {
      k.y0 = 0;
      k.y1 = bh;
    }
    if (k.x1 > 1) {
      k.x1 = 1;
      k.x0 = 1 - bw;
    }
    if (k.y1 > 1) {
      k.y1 = 1;
      k.y0 = 1 - bh;
    }
    return clampKeep(k);
  }
  if (id.includes("n")) k.y0 = ny;
  if (id.includes("s")) k.y1 = ny;
  if (id.includes("w")) k.x0 = nx;
  if (id.includes("e")) k.x1 = nx;
  return clampKeep(k);
}

function onCleanDragStart(ev) {
  if (ev.button !== 0) return;
  const item = cleanCurrent();
  const pt = cleanCanvasPoint(ev);
  if (!item || !pt) return;
  const keep = keepOf(item);
  const id = hitKeepHandle(pt, keep);
  if (!id) return;
  state.clean.drag = { id, start: { nx: pt.nx, ny: pt.ny }, orig: { ...keep } };
  state.clean.drawing = true;
}

function onCleanDragMove(ev) {
  const canvas = $("#clean-overlay");
  const item = cleanCurrent();
  const pt = cleanCanvasPoint(ev);
  if (!item || !pt || !canvas) return;
  const keep = keepOf(item);
  if (state.clean && state.clean.drawing && state.clean.drag) {
    const next = applyHandleDrag(state.clean.drag.orig, state.clean.drag.id, pt.nx, pt.ny, state.clean.drag.start);
    item.keep = next;
    canvas.style.cursor = cursorForHandle(state.clean.drag.id);
    drawCleanOverlay();
    updateKeepSizeLabel(item, next);
    return;
  }
  canvas.style.cursor = cursorForHandle(hitKeepHandle(pt, keep));
}

function onCleanDragEnd() {
  const c = state.clean;
  if (!c) return;
  const was = c.drawing;
  c.drawing = false;
  c.drag = null;
  if (was) renderDataset();
}

function drawCleanOverlay() {
  const item = cleanCurrent();
  const canvas = $("#clean-overlay");
  const img = $("#clean-img");
  if (!item || !canvas || !img || !img.naturalWidth) return;
  const dpr = window.devicePixelRatio || 1;
  const cw = img.clientWidth;
  const ch = img.clientHeight;
  canvas.width = Math.max(1, Math.round(cw * dpr));
  canvas.height = Math.max(1, Math.round(ch * dpr));
  canvas.style.width = `${cw}px`;
  canvas.style.height = `${ch}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cw, ch);
  const keep = keepOf(item);
  const x = keep.x0 * cw;
  const y = keep.y0 * ch;
  const bw = (keep.x1 - keep.x0) * cw;
  const bh = (keep.y1 - keep.y0) * ch;
  ctx.fillStyle = "rgba(17, 15, 12, 0.55)";
  ctx.fillRect(0, 0, cw, y);
  ctx.fillRect(0, y, x, bh);
  ctx.fillRect(x + bw, y, cw - x - bw, bh);
  ctx.fillRect(0, y + bh, cw, ch - y - bh);
  ctx.strokeStyle = "#d7a44a";
  ctx.lineWidth = 2;
  ctx.strokeRect(x + 1, y + 1, Math.max(0, bw - 2), Math.max(0, bh - 2));
  ctx.fillStyle = "#d7a44a";
  keepHandles(keep, cw, ch).forEach((h) => {
    ctx.fillRect(h.x - 5, h.y - 5, 10, 10);
  });
  const px = keepPixels(item, keep);
  if (px.srcW && px.srcH) {
    const warn = px.short > 0 && px.short < 1024;
    const label = warn ? `${px.w}×${px.h}  short ${px.short}` : `${px.w}×${px.h}`;
    drawSizeChip(ctx, x, y, bw, bh, label, warn);
  }
}

async function applyCleanStill(mode, extra = {}) {
  const d = state.dataset;
  const item = cleanCurrent();
  if (!item) return;
  const keep = keepOf(item);
  const enabled = (state.clean.items || []).filter((it) => it.enabled !== false);
  let payload;
  if (extra.disable || extra.ignore) {
    payload = { items: [{ file: item.file, keep, disable: !!extra.disable, ignore: !!extra.ignore }] };
  } else if (mode === "copy") {
    if (keepIsFull(keep)) return;
    payload = { apply_keep: keep, files: enabled.map((it) => it.file) };
  } else if (mode === "each") {
    const items = enabled
      .map((it) => ({ file: it.file, keep: keepOf(it) }))
      .filter((it) => !keepIsFull(it.keep));
    if (!items.length) return;
    payload = { items };
  } else {
    if (keepIsFull(keep)) return;
    payload = { items: [{ file: item.file, keep }] };
  }
  try {
    state.clean.busy = true;
    renderDataset();
    const out = await api(`/api/datasets/${encodeURIComponent(d.slug)}/clean/apply`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    state.dataset = out.dataset;
    const failed = (out.results || []).filter((r) => !r.ok);
    if (failed.length) {
      alert(failed.map((r) => `${r.file}: ${r.note}`).join("\n"));
    }
    const keepFile = item.file;
    const scan = await api(`/api/datasets/${encodeURIComponent(d.slug)}/clean/scan`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    state.dataset = scan.dataset || state.dataset;
    const items = (scan.items || []).map((it) => ({
      ...it,
      keep: it.keep || keepFromBoxes(it.boxes || []),
    }));
    const idx = items.findIndex((it) => it.file === keepFile);
    state.clean = {
      items,
      index: idx >= 0 ? idx : 0,
      selected: -1,
      err: "",
      busy: false,
      linked: !!scan.linked,
      flagged: scan.flagged || 0,
    };
    renderDataset();
  } catch (err) {
    state.clean.busy = false;
    state.clean.err = err.message;
    renderDataset();
  }
}

async function runExtract() {
  const d = state.dataset;
  try {
    await saveMeta();
  } catch (err) {
    state.error = err.message;
    renderDataset();
    return;
  }
  const payload = {
    slug: d.slug,
    overwrite: $("#f-over").checked,
    bump_version: $("#f-bump").checked,
    ...gatherPatch(),
  };
  state.extracting = true;
  state.log = [];
  state.result = null;
  state.error = null;
  state.progress = { stage: "starting", label: "Starting…", percent: 1, encoded: 0, total: d.enabled_count || d.items.length, current: "" };
  state.extractFiles = (d.items || []).filter((it) => it.enabled).map((it) => it.file);
  renderDataset();
  try {
    await api("/api/extract", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch (err) {
    state.extracting = false;
    state.error = err.message;
    renderDataset();
    return;
  }
  const tick = async () => {
    const st = await api("/api/extract/status");
    state.log = st.log || [];
    state.progress = st.progress || state.progress;
    if (st.files && st.files.length) state.extractFiles = st.files;
    const box = $("#extract-log");
    if ($("#extract-viz")) {
      applyExtractProgress(state.progress);
      if (box) {
        box.textContent = state.log.join("\n");
        box.scrollTop = box.scrollHeight;
      }
    } else {
      renderDataset();
    }
    if (st.status === "running") {
      setTimeout(tick, 400);
      return;
    }
    state.extracting = false;
    if (st.status === "done") {
      state.result = st.result;
      state.error = null;
      state.dataset = (st.result && st.result.dataset) || await api(`/api/datasets/${encodeURIComponent(d.slug)}`);
    } else {
      state.error = st.error || "extract failed";
    }
    renderDataset();
  };
  setTimeout(tick, 300);
}

function renderLibrary() {
  const q = state.query.trim().toLowerCase();
  const filter = state.folderFilter || "";
  const mods = (state.library || []).filter((m) => {
    const folder = m.subfolder || "";
    if (filter === "__root__" && folder) return false;
    if (filter && filter !== "__root__" && folder !== filter) return false;
    const blob = `${m.id || ""} ${m.name || ""} ${m.concept_type || ""} ${m.subfolder || ""}`.toLowerCase();
    return !q || blob.includes(q);
  });
  const groups = new Map();
  for (const m of mods) {
    const key = m.subfolder || "";
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(m);
  }
  for (const folder of state.folders || []) {
    if (!groups.has(folder)) groups.set(folder, []);
  }
  const keys = [...groups.keys()].sort((a, b) => {
    if (!a) return -1;
    if (!b) return 1;
    return a.localeCompare(b);
  }).filter((folder) => {
    if (filter === "__root__") return !folder;
    if (filter) return folder === filter;
    return true;
  });
  const searching = Boolean(q);
  const body = keys.map((folder) => {
    const label = folder || "(root)";
    const foldKey = folder || "__root__";
    const items = groups.get(folder) || [];
    const collapsed = !searching && Boolean(state.libraryCollapsed[foldKey]);
    const rows = collapsed ? "" : items.map((m) => `<tr>
      <td><input type="checkbox" data-hub-id="${escapeHtml(m.id || "")}" ${state.hubSelected[m.id] ? "checked" : ""}></td>
      <td><b>${escapeHtml(m.name)}</b><div class="mono">${escapeHtml(m.id || m.name)}</div></td>
      <td>${escapeHtml(m.concept_type || "—")}<div class="meta">${escapeHtml(m.mode || m.kind || "")}</div></td>
      <td>${m.token_count ?? "—"}</td>
      <td>${fmtBytes(m.bytes)}${m.likely_empty ? ` <span class="badge dirty">tiny</span>` : ""}</td>
      <td>${hfBadge(m)}</td>
      <td>${m.dataset_slug ? `<a href="#/datasets/${escapeHtml(m.dataset_slug)}">${escapeHtml(m.dataset_slug)}</a>` : "—"}</td>
      <td>
        <button class="btn small" data-open="${escapeHtml(m.path)}">folder</button>
        <button class="btn small danger" data-del="${escapeHtml(m.id || m.name)}">delete</button>
      </td>
    </tr>`).join("");
    const groupIds = items.map((m) => m.id).filter(Boolean);
    const allOn = groupIds.length && groupIds.every((id) => state.hubSelected[id]);
    const actions = folder ? `
        ${folder ? `<button class="btn small" data-ren="${escapeHtml(folder)}">rename</button>` : ""}
        ${items.length === 0 ? `<button class="btn small danger" data-rmdir="${escapeHtml(folder)}">delete</button>` : ""}
      ` : "";
    return `<tr class="folder-row${collapsed ? " collapsed" : ""}">
      <td class="folder-toggle" data-fold="${escapeHtml(foldKey)}">
        <input type="checkbox" data-fold-select="${escapeHtml(foldKey)}" ${allOn ? "checked" : ""}>
      </td>
      <td colspan="6" class="folder-toggle" data-fold="${escapeHtml(foldKey)}" title="${collapsed ? "Expand" : "Collapse"}">
        <span class="caret">${collapsed ? "▶" : "▼"}</span>
        ${escapeHtml(label)} · ${items.length}
      </td>
      <td>${actions}</td>
    </tr>${rows}`;
  }).join("");
  const folderOpts = [`<option value="" ${!filter ? "selected" : ""}>all folders</option>`, `<option value="__root__" ${filter === "__root__" ? "selected" : ""}>(root)</option>`]
    .concat((state.folders || []).map((f) => `<option value="${escapeHtml(f)}" ${filter === f ? "selected" : ""}>${escapeHtml(f)}</option>`))
    .join("");
  $("#app").innerHTML = `
    <div class="toolbar">
      <h2>Library</h2>
      ${librarySwitchHtml()}
      ${busyHtml(state.libraryLoading, Boolean(keys.length))}
      <input class="search" id="search" placeholder="Search mods" value="${escapeHtml(state.query)}">
      <select id="folder-filter" class="search" style="max-width:220px">${folderOpts}</select>
      <button class="btn" id="btn-expand-all" ${searching ? "disabled" : ""}>Expand all</button>
      <button class="btn" id="btn-collapse-all" ${searching ? "disabled" : ""}>Collapse all</button>
      <button class="btn" id="btn-new-folder">New folder</button>
      <button class="btn" id="btn-sync-comfy">Sync from Studio</button>
      <button class="btn primary" id="btn-hub-upload" ${state.hubUploading ? "disabled" : ""}>Upload selected${selectedHubIds(mods).length ? ` (${selectedHubIds(mods).length})` : ""}</button>
    </div>
    <p class="meta">Files ComfyUI loads from <span class="mono">models/refmods/</span>. Sync from Studio rebuilds this folder tree to match dataset folders. Folder names here are the loader dropdown.</p>
    ${keys.length ? `<table class="table">
      <thead><tr><th></th><th>Mod</th><th>Type</th><th>Tokens</th><th>Size</th><th>Hub</th><th>Dataset</th><th></th></tr></thead>
      <tbody>${body}</tbody>
    </table>` : `<p class="empty">${state.libraryLoading ? "Loading RefMods…" : "No RefMods in the ComfyUI models/refmods folder yet. Create a folder or extract a dataset into <span class=\"mono\">identity/</span>."}</p>`}
    ${hubDockHtml()}
  `;
  const search = $("#search");
  if (search) search.addEventListener("input", (e) => { state.query = e.target.value; renderLibrary(); });
  const sel = $("#folder-filter");
  if (sel) sel.onchange = () => { state.folderFilter = sel.value; renderLibrary(); };
  $$("[data-fold]").forEach((el) => {
    el.onclick = (e) => {
      if (e.target.closest("button, a")) return;
      const key = el.dataset.fold;
      if (state.libraryCollapsed[key]) delete state.libraryCollapsed[key];
      else state.libraryCollapsed[key] = true;
      renderLibrary();
    };
  });
  const expandAll = $("#btn-expand-all");
  if (expandAll) expandAll.onclick = () => { state.libraryCollapsed = {}; renderLibrary(); };
  const collapseAll = $("#btn-collapse-all");
  if (collapseAll) collapseAll.onclick = () => {
    state.libraryCollapsed = Object.fromEntries(keys.map((folder) => [folder || "__root__", true]));
    renderLibrary();
  };
  const newFolder = $("#btn-new-folder");
  if (newFolder) newFolder.onclick = () => promptCreateFolder(async () => {
    await loadLibrary();
    renderLibrary();
  });
  const syncBtn = $("#btn-sync-comfy");
  if (syncBtn) syncBtn.onclick = () => syncComfyLayout(async () => {
    await loadLibrary();
    renderLibrary();
  });
  $$("[data-ren]").forEach((b) => b.onclick = () => promptRenameFolder(b.dataset.ren, async () => {
    await loadLibrary();
    renderLibrary();
  }));
  $$("[data-rmdir]").forEach((b) => b.onclick = async () => {
    if (!confirm(`Delete empty folder ${b.dataset.rmdir}?`)) return;
    await api(`/api/library/folders/${encodeURIComponent(b.dataset.rmdir)}`, { method: "DELETE" });
    await loadLibrary();
    renderLibrary();
  });
  $$("[data-open]").forEach((b) => b.onclick = () => api("/api/open-folder", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path: b.dataset.open }),
  }));
  $$("[data-del]").forEach((b) => b.onclick = async () => {
    if (!confirm(`Delete ${b.dataset.del}.safetensors?`)) return;
    await api(`/api/library/${encodeURIComponent(b.dataset.del)}`, { method: "DELETE" });
    await loadLibrary();
    renderLibrary();
  });
  bindHubChecks(mods, () => renderLibrary());
}

function studioStatus(m) {
  if (m.status === "in_sync") return `<span class="badge ok">in ComfyUI</span>`;
  if (m.status === "stale") return `<span class="badge dirty">out of date</span>`;
  if (m.status === "comfy_only") return `<span class="badge">ComfyUI only</span>`;
  return `<span class="badge dirty">not in ComfyUI</span>`;
}

function renderStudio() {
  const q = state.query.trim().toLowerCase();
  const filter = state.studioFolderFilter || "";
  const mods = (state.studio || []).filter((m) => {
    const folder = m.subfolder || "";
    if (filter === "__root__" && folder) return false;
    if (filter && filter !== "__root__" && folder !== filter && !folder.startsWith(`${filter}/`)) return false;
    const blob = `${m.id || ""} ${m.name || ""} ${m.display_name || ""} ${m.dataset_slug || ""} ${m.subfolder || ""}`.toLowerCase();
    return !q || blob.includes(q);
  });
  const groups = new Map();
  for (const m of mods) {
    const key = m.subfolder || "";
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(m);
  }
  for (const folder of state.studioFolders || []) {
    if (!groups.has(folder)) groups.set(folder, []);
  }
  const keys = [...groups.keys()].sort((a, b) => {
    if (!a) return -1;
    if (!b) return 1;
    return a.localeCompare(b);
  }).filter((folder) => {
    if (filter === "__root__") return !folder;
    if (filter) return folder === filter || folder.startsWith(`${filter}/`);
    return true;
  });
  const searching = Boolean(q);
  const body = keys.map((folder) => {
    const label = folder || "(root)";
    const foldKey = folder || "__root__";
    const items = groups.get(folder) || [];
    const collapsed = !searching && Boolean(state.studioCollapsed[foldKey]);
    const rows = collapsed ? "" : items.map((m) => {
      const install = m.status === "missing"
        ? `<button class="btn small" data-install="${escapeHtml(m.dataset_slug)}">Copy to ComfyUI</button>`
        : (m.status === "stale"
          ? `<button class="btn small" data-replace="${escapeHtml(m.dataset_slug)}">Replace in ComfyUI</button>`
          : "");
      return `<tr>
      <td><input type="checkbox" data-hub-id="${escapeHtml(m.id || "")}" ${state.hubSelected[m.id] ? "checked" : ""}></td>
      <td><b>${escapeHtml(m.display_name || m.name)}</b><div class="mono">${escapeHtml(m.mod_name || m.name)}</div></td>
      <td>${escapeHtml(m.concept_type || "—")}<div class="meta">${escapeHtml(m.mode || "")}</div></td>
      <td>${m.token_count ?? "—"}</td>
      <td>${fmtBytes(m.bytes)}${m.dirty ? ` <span class="badge dirty">dataset dirty</span>` : ""}</td>
      <td>${studioStatus(m)}</td>
      <td>${hfBadge(m)}</td>
      <td>
        ${m.dataset_slug ? `<a class="btn small" href="#/datasets/${escapeHtml(m.dataset_slug)}">dataset</a>` : ""}
        ${m.local_path ? `<button class="btn small" data-open="${escapeHtml(m.local_path)}">folder</button>` : ""}
        ${install}
      </td>
    </tr>`;
    }).join("");
    const groupIds = items.map((m) => m.id).filter(Boolean);
    const allOn = groupIds.length && groupIds.every((id) => state.hubSelected[id]);
    const actions = folder
      ? `<button class="btn small" data-ren="${escapeHtml(folder)}">rename</button>`
      : "";
    return `<tr class="folder-row${collapsed ? " collapsed" : ""}">
      <td class="folder-toggle" data-fold="${escapeHtml(foldKey)}">
        <input type="checkbox" data-fold-select="${escapeHtml(foldKey)}" ${allOn ? "checked" : ""}>
      </td>
      <td colspan="6" class="folder-toggle" data-fold="${escapeHtml(foldKey)}" title="${collapsed ? "Expand" : "Collapse"}">
        <span class="caret">${collapsed ? "▶" : "▼"}</span>
        ${escapeHtml(label)} · ${items.length}
      </td>
      <td>${actions}</td>
    </tr>${rows}`;
  }).join("");
  const folderOpts = [`<option value="" ${!filter ? "selected" : ""}>all folders</option>`, `<option value="__root__" ${filter === "__root__" ? "selected" : ""}>(root)</option>`]
    .concat((state.studioFolders || []).map((f) => `<option value="${escapeHtml(f)}" ${filter === f ? "selected" : ""}>${escapeHtml(f)}</option>`))
    .join("");
  $("#app").innerHTML = `
    <div class="toolbar">
      <h2>Library</h2>
      ${librarySwitchHtml()}
      ${busyHtml(state.studioLoading, Boolean(keys.length))}
      <input class="search" id="search" placeholder="Search studio mods" value="${escapeHtml(state.query)}">
      <select id="studio-folder-filter" class="search" style="max-width:220px">${folderOpts}</select>
      <button class="btn" id="btn-expand-all" ${searching ? "disabled" : ""}>Expand all</button>
      <button class="btn" id="btn-collapse-all" ${searching ? "disabled" : ""}>Collapse all</button>
      <button class="btn" id="btn-sync-comfy">Sync to ComfyUI</button>
      <button class="btn primary" id="btn-hub-upload" ${state.hubUploading ? "disabled" : ""}>Upload selected${selectedHubIds(mods).length ? ` (${selectedHubIds(mods).length})` : ""}</button>
    </div>
    <p class="meta">Copies kept next to each dataset. Rename a folder from its group row, then Sync to ComfyUI to rebuild <span class="mono">models/refmods/</span> to match. Extract writes here and into ComfyUI.</p>
    ${keys.length ? `<table class="table">
      <thead><tr><th></th><th>Mod</th><th>Type</th><th>Tokens</th><th>Size</th><th>ComfyUI</th><th>Hub</th><th></th></tr></thead>
      <tbody>${body}</tbody>
    </table>` : `<p class="empty">${state.studioLoading ? "Loading RefMods…" : "No extracted RefMods in the studio yet. Extract a dataset and a copy stays in that dataset folder."}</p>`}
    ${hubDockHtml()}
  `;
  const search = $("#search");
  if (search) search.addEventListener("input", (e) => { state.query = e.target.value; renderStudio(); });
  const sel = $("#studio-folder-filter");
  if (sel) sel.onchange = () => { state.studioFolderFilter = sel.value; renderStudio(); };
  $$("[data-fold]").forEach((el) => {
    el.onclick = (e) => {
      if (e.target.closest("button, a, input")) return;
      const key = el.dataset.fold;
      if (state.studioCollapsed[key]) delete state.studioCollapsed[key];
      else state.studioCollapsed[key] = true;
      renderStudio();
    };
  });
  $$("[data-ren]").forEach((b) => b.onclick = () => {
    const src = b.dataset.ren;
    promptRenameFolder(src, async (dst) => {
      if (state.studioFolderFilter === src) state.studioFolderFilter = dst || "";
      await loadStudio();
      renderStudio();
    });
  });
  const syncBtn = $("#btn-sync-comfy");
  if (syncBtn) syncBtn.onclick = () => syncComfyLayout(async () => {
    await loadStudio();
    renderStudio();
  });
  const expandAll = $("#btn-expand-all");
  if (expandAll) expandAll.onclick = () => { state.studioCollapsed = {}; renderStudio(); };
  const collapseAll = $("#btn-collapse-all");
  if (collapseAll) collapseAll.onclick = () => {
    state.studioCollapsed = Object.fromEntries(keys.map((folder) => [folder || "__root__", true]));
    renderStudio();
  };
  $$("[data-open]").forEach((b) => b.onclick = () => api("/api/open-folder", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path: b.dataset.open }),
  }));
  const push = async (slug, replace) => {
    const q = replace ? "?replace=true" : "";
    await api(`/api/studio/${encodeURIComponent(slug)}/install${q}`, { method: "POST" });
    await loadStudio();
    renderStudio();
  };
  $$("[data-install]").forEach((b) => b.onclick = async () => {
    try { await push(b.dataset.install, false); } catch (err) { alert(err.message); }
  });
  $$("[data-replace]").forEach((b) => b.onclick = async () => {
    if (!confirm(`Replace the ComfyUI copy of ${b.dataset.replace}?`)) return;
    try { await push(b.dataset.replace, true); } catch (err) { alert(err.message); }
  });
  bindHubChecks(mods, () => renderStudio());
}

const SAMPLE_PROMPTS = window.SAMPLE_PROMPTS || [];

function bindBackToTop() {
  $$(".guide-section").forEach((sec) => {
    if (!sec.querySelector(":scope > .back-top")) {
      sec.insertAdjacentHTML("beforeend", `<button type="button" class="back-top" data-top>Back to top</button>`);
    }
  });
  $$("[data-top]").forEach((btn) => {
    btn.onclick = () => {
      const main = document.querySelector("main");
      if (main) main.scrollTo({ top: 0, behavior: "smooth" });
    };
  });
}

function copyPromptText(text, btn) {
  const done = () => {
    const prev = btn.textContent;
    btn.textContent = "Copied";
    setTimeout(() => { btn.textContent = prev; }, 1200);
  };
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(done).catch(done);
    return;
  }
  const ta = document.createElement("textarea");
  ta.value = text;
  document.body.appendChild(ta);
  ta.select();
  try { document.execCommand("copy"); } catch { /* ignore */ }
  ta.remove();
  done();
}

function renderPrompts() {
  const q = (state.promptQuery || "").trim().toLowerCase();
  const indexed = SAMPLE_PROMPTS.map((p, i) => ({ p, i })).filter(({ p }) => {
    if (!q) return true;
    return `${p.group} ${p.title} ${p.tests} ${p.text}`.toLowerCase().includes(q);
  });
  const groups = [];
  for (const { p } of indexed) {
    if (!groups.includes(p.group)) groups.push(p.group);
  }
  const toc = groups.map((g, i) => {
    const n = indexed.filter(({ p }) => p.group === g).length;
    return `<button type="button" data-jump="pg-${i}">${escapeHtml(g)} · ${n}</button>`;
  }).join("");
  const sections = groups.map((g, gi) => {
    const cards = indexed.map(({ p, i }) => {
      if (p.group !== g) return "";
      return `<article class="prompt-card">
        <div class="prompt-head">
          <div>
            <h3>${escapeHtml(p.title)}</h3>
            <div class="meta">Tests ${escapeHtml(p.tests)}</div>
          </div>
          <button type="button" class="btn small" data-copy="${i}">Copy</button>
        </div>
        <pre class="prompt-text">${escapeHtml(p.text)}</pre>
      </article>`;
    }).join("");
    return `<section class="guide-section" id="pg-${gi}">
      <div class="guide-head">
        <h3>${escapeHtml(g)}</h3>
        <button type="button" class="back-top" data-top>Back to top</button>
      </div>
      <div class="prompt-list">${cards}</div>
    </section>`;
  }).join("");
  const empty = indexed.length ? sections : `<p class="empty">No prompts match that search.</p>`;
  $("#app").innerHTML = `
    <div class="guide prompts-page">
      <div class="toolbar">
        <h2>Sample Prompts</h2>
        <span class="meta">${indexed.length} of ${SAMPLE_PROMPTS.length}</span>
        <input class="search" id="prompt-search" placeholder="Search 5s, 10s, spicy, rooftop…" value="${escapeHtml(state.promptQuery || "")}">
      </div>
      <p class="lede">5s, 10s, and 15s T2VA blocks, one main subject. Match that duration in ComfyUI. Edit the Subject 1 line to the person’s name and wardrobe. Spicy is R-rated, not explicit.</p>
      <nav class="guide-toc" id="prompt-toc">${toc}</nav>
      ${empty}
    </div>
  `;
  const search = $("#prompt-search");
  if (search) {
    search.addEventListener("input", (e) => {
      state.promptQuery = e.target.value;
      const pos = search.selectionStart;
      renderPrompts();
      const next = $("#prompt-search");
      if (next) {
        next.focus();
        try { next.setSelectionRange(pos, pos); } catch { /* ignore */ }
      }
    });
  }
  $$("#prompt-toc [data-jump]").forEach((btn) => {
    btn.onclick = () => {
      const el = document.getElementById(btn.dataset.jump);
      if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
    };
  });
  $$("[data-copy]").forEach((btn) => {
    btn.onclick = () => {
      const p = SAMPLE_PROMPTS[Number(btn.dataset.copy)];
      if (p) copyPromptText(p.text, btn);
    };
  });
  bindBackToTop();
}

function renderInstructions() {
  $("#app").innerHTML = `
    <div class="guide">
      <div class="toolbar">
        <h2>Instructions</h2>
      </div>
      <p class="lede">This studio encodes stills and optional video clips into MiniMax-H3 RefMods — small identity adapters for ComfyUI. It does not generate video. Curate a set, extract once, then re-extract when the set changes.</p>
      <nav class="guide-toc" id="guide-toc">
        <button type="button" data-jump="g-loop">The loop</button>
        <button type="button" data-jump="g-pages">Pages</button>
        <button type="button" data-jump="g-setup">Setup</button>
        <button type="button" data-jump="g-stills">Stills &amp; clips</button>
        <button type="button" data-jump="g-extract">Extract</button>
        <button type="button" data-jump="g-folders">Folders</button>
        <button type="button" data-jump="g-comfy">In ComfyUI</button>
        <button type="button" data-jump="g-hub">Hugging Face</button>
        <button type="button" data-jump="g-tools">Tools</button>
      </nav>

      <section class="guide-section" id="g-loop">
        <h3>The loop</h3>
        <ol>
          <li>Point <a href="#/setup">Setup</a> at your ComfyUI (needed for extract; you can still curate stills without it).</li>
          <li>Create or import a dataset. Drop <b>8–20 stills</b> for an identity set. You can add video clips too.</li>
          <li>Pick a RefMod folder (new identity sets default to <span class="mono">identity/</span>). Extract.</li>
          <li>The mod lands in <span class="mono">ComfyUI/models/refmods/&lt;folder&gt;/</span>. A copy stays next to the dataset.</li>
          <li>Swap a weak still, then <b>Re-extract</b>. The ComfyUI loader name stays the same unless you bump the version or change the folder.</li>
        </ol>
      </section>

      <section class="guide-section" id="g-pages">
        <h3>Pages</h3>
        <div class="guide-tiles">
          <article class="guide-tile">
            <h3>Datasets</h3>
            <p>Your stills and clips. Open a set to drop files, set the cover, clean watermarks, and extract.</p>
          </article>
          <article class="guide-tile">
            <h3>Library</h3>
            <p>Switch between <a href="#/library">ComfyUI</a> (<span class="mono">models/refmods/</span>) and <a href="#/library/studio">Studio</a> (copies next to each dataset). Rename folders on Studio, then Sync to ComfyUI. Upload to Hugging Face from either pane.</p>
          </article>
          <article class="guide-tile">
            <h3>Sample Prompts</h3>
            <p>Copy-paste 5s, 10s, and 15s T2VA tests with one subject. Spicy is R-rated, not explicit.</p>
          </article>
          <article class="guide-tile">
            <h3>Setup</h3>
            <p>ComfyUI root, Python, VAE, datasets folder, mod-name prefix, and Hugging Face token.</p>
          </article>
        </div>
      </section>

      <section class="guide-section" id="g-setup">
        <h3>Setup</h3>
        <p>Extract needs a ComfyUI install that already has:</p>
        <ul>
          <li><span class="mono">custom_nodes/ComfyUI-MiniMaxH3Mod</span></li>
          <li><span class="mono">models/vae/minimax_h3_video_vae_fp16.safetensors</span></li>
        </ul>
        <p>Works with <b>any local ComfyUI</b>: Desktop, Windows portable, Easy-Install, Stability Matrix, or a git clone. Paste the folder with <span class="mono">main.py</span>, or the wrapper that contains a <span class="mono">ComfyUI</span> folder. For Desktop, pick the user data folder (often <span class="mono">Documents\\ComfyUI</span>), not the app under <span class="mono">AppData\\Local\\Programs</span>.</p>
        <p>Leave Python blank and Save — portable / Easy-Install fill <span class="mono">python_embeded\\python.exe</span>; Desktop / venv fills <span class="mono">.venv</span> or <span class="mono">venv</span>.</p>
        <p>The health chip in the header turns green when extract is ready. Until then you can still build datasets.</p>
      </section>

      <section class="guide-section" id="g-stills">
        <h3>Stills and clips</h3>
        <p>Identity sets work best at <b>8–20</b> stills. Boosters / concept sets usually want <b>6–12</b> targeted shots. Low-res frames (short edge under 1024) are marked on the sheet.</p>
        <p>You can also drop video clips onto the same sheet. Extract sends stills as <span class="mono">--image</span> and clips as <span class="mono">--video</span>. A set can be stills only, clips only, or mixed. Identity work is still usually still-led; clips add motion and extra angles, they do not replace a good still mix.</p>
        <table>
          <thead><tr><th>Mix</th><th>Aim</th></tr></thead>
          <tbody>
            <tr><td>Front</td><td>About 40% — clear face, eye contact, a couple of expressions</td></tr>
            <tr><td>Three-quarter</td><td>About 30% — left and right</td></tr>
            <tr><td>Profile</td><td>About 20%</td></tr>
            <tr><td>Body</td><td>About 10% — waist-up or full-body</td></tr>
          </tbody>
        </table>
        <ul>
          <li>Vary lighting and backgrounds so the encoder locks on the face, not the room.</li>
          <li>Prefer sharp, high-res originals. PNG / JPG / WebP are fine.</li>
          <li>Uncheck <b>use</b> on a still to leave it in the folder without encoding it.</li>
          <li>Tick <b>cover</b> on the still that should appear on the Datasets list. That does not mark the set dirty.</li>
          <li>Pixel size is shown on each thumbnail. Short edge under 1024 is also marked <b>low-res</b>.</li>
        </ul>
        <p>Clips: <span class="mono">.mp4</span>, <span class="mono">.webm</span>, <span class="mono">.mov</span>, <span class="mono">.mkv</span>, <span class="mono">.avi</span>, <span class="mono">.m4v</span>. They show as a <b>CLIP</b> tile (no thumbnail), they cannot be the list cover, and watermark clean is stills-only. Leave <b>use</b> checked to encode them. Audio in the file is ignored.</p>
        <table>
          <thead><tr><th>Limit</th><th>What actually happens</th></tr></thead>
          <tbody>
            <tr><td>File size</td><td>No studio cap. The clip just has to decode (OpenCV, then imageio).</td></tr>
            <tr><td>Duration</td><td>Any length. Extract uniformly samples at most <b>60 frames</b> from the whole clip, then Full Reference keeps <b>16</b> of those and snaps to 13 (H3’s 4k+1 frame grid). A 2-second clip and a 2-minute clip both collapse to that same budget.</td></tr>
            <tr><td>Resolution</td><td>Longest edge is capped around <b>2048</b> on load (at the default 1024 short-edge extract). Then downscaled to extract resolution. Tiny frames are upscaled to a 320px floor.</td></tr>
            <tr><td>Token budget</td><td>Stills + clips share one cap (default <b>8192</b>). Extra latent frames get dropped if the stack does not fit. Encode is ~0.2–1&nbsp;MB per latent frame before that trim.</td></tr>
            <tr><td>Canvas</td><td>The first still anchors the canvas; clips are cover-cropped to it. Video-only sets use the first clip.</td></tr>
          </tbody>
        </table>
        <div class="guide-note">
          <b>Skip</b> heavy makeup changes, occluded faces, and near-duplicates. One strong angle beats three blurry copies of it. Short, clear clips beat long busy ones — extra minutes are not extra identity.
        </div>
      </section>

      <section class="guide-section" id="g-extract">
        <h3>Extract and “updated”</h3>
        <p>Identity defaults: Full Reference (<span class="mono">encode</span>), concept <span class="mono">identity</span>, short edge <span class="mono">1024</span>, max tokens <span class="mono">8192</span>.</p>
        <p>A dataset is <b>updated</b> when the extract fingerprint no longer matches the last successful extract. That includes enabled stills and clips (add / remove / edit / reorder / use-toggle), description, concept, mod name, folder, and extract settings (mode, resolution, tokens, pool, refinement, multiplier).</p>
        <p>Cover, display name, captions, device, and disabled files do <b>not</b> mark it dirty. Extracting (or installing a matching copy into ComfyUI) clears it.</p>
        <ul>
          <li><b>Overwrite</b> replaces the same filename in ComfyUI.</li>
          <li><b>Bump version</b> writes <span class="mono">_v2_refmod</span> (and so on) so the old loader entry stays.</li>
        </ul>
      </section>

      <section class="guide-section" id="g-folders">
        <h3>Folders</h3>
        <p>Folder names are the ComfyUI loader path: <span class="mono">identity/name</span>, <span class="mono">celebs/june/name</span>. Nested names use <span class="mono">/</span>.</p>
        <ol>
          <li>Set the folder on the dataset (sidebar dropdown), or create one from that dropdown.</li>
          <li>Rename from the folder row on <a href="#/library/studio">Library → Studio</a> (also on ComfyUI). That moves every RefMod in the folder and updates datasets that used it.</li>
          <li>Click <b>Sync to ComfyUI</b> so <span class="mono">models/refmods/</span> matches Studio: missing folders are created, misplaced mods are moved, missing mods are copied, empty leftovers are removed. Unrelated ComfyUI-only files stay put.</li>
        </ol>
        <p>After a sync or extract, use <b>Refresh RefMods</b> on the ComfyUI loader if the dropdown looks stale.</p>
      </section>

      <section class="guide-section" id="g-comfy">
        <h3>Use in ComfyUI</h3>
        <p>In a MiniMax-H3 workflow:</p>
        <p class="mono">Load H3 RefMods → Apply H3 RefMod (between conditioning and the guider)</p>
        <p>Copy ready-made 5s / 10s / 15s shots from the <a href="#/prompts">Sample Prompts</a> tab and set the same duration on the H3 sampler.</p>
        <ul>
          <li>Pick the extracted mod, e.g. <span class="mono">identity/sdprompts_minimaxh3_&lt;name&gt;_v1_refmod</span>.</li>
          <li>Strength <span class="mono">1.0</span>, copies <span class="mono">1</span>.</li>
          <li>On MiniMaxH3Mod 0.2.x+, leave Apply on <span class="mono">constant</span> / <span class="mono">linear</span> / <span class="mono">1.0</span>.</li>
          <li>T2VA prompts: <span class="mono">subject_definitions</span>, then <span class="mono">integrated_multimodal_description</span>, <span class="mono">overall_soundscape</span>, <span class="mono">non_diegetic_music</span>. Use <span class="mono">&lt;Subject 1&gt;</span>, not <span class="mono">&lt;Picture 1&gt;</span>. Copy tests from <a href="#/prompts">Sample Prompts</a>. Match clip length to the prompt group (5, 10, or 15 seconds). Later shot timestamps stay inside that length.</li>
        </ul>
        <p>A healthy RefMod is about <b>1.1–1.6 MB</b>. Under ~100 KB usually means encode failed.</p>
      </section>

      <section class="guide-section" id="g-hub">
        <h3>Hugging Face</h3>
        <p>On Setup, save a write token and a repo (<span class="mono">user/name</span>). On Library, select mods and <b>Upload selected</b>. Uploads keep the ComfyUI folder path. After you re-extract, Hub stale means the remote file no longer matches.</p>
      </section>

      <section class="guide-section" id="g-tools">
        <h3>Other tools</h3>
        <ul>
          <li><b>Import folder</b> — copy or link an existing stills (and clips) directory as a dataset.</li>
          <li><b>Split sheet</b> — cut a contact sheet into panels and make one dataset per sheet.</li>
          <li><b>Clean watermarks</b> — flag edge marks, crop or keep an original under <span class="mono">_originals/</span>.</li>
          <li><b>Open folder</b> — reveal the dataset directory on disk.</li>
          <li><b>Notes</b> — private comments on the dataset. They are not written into the RefMod and do not mark it updated.</li>
          <li><b>Check quality</b> — ranks stills 0–100 from face sharpness, size, lighting, and pose, and labels close-up / medium / full-body plus head angle. It flags crowded poses, marks a diverse recommended 10, and lists what to shoot next (missing 3/4s, a medium, a full-body, sharper close-ups). Clips are skipped. Rank only until you click Use recommended, which unchecks the rest.</li>
        </ul>
        <p>Community guides by <a href="https://huggingface.co/malcolmrey" target="_blank" rel="noopener">malcolmrey</a>: <a href="https://huggingface.co/datasets/malcolmrey/various/blob/main/h3-center/docs/MINIMAX_H3_REFMOD_CREATION_GUIDE.md" target="_blank" rel="noopener">creation guide</a> · <a href="https://huggingface.co/datasets/malcolmrey/various/blob/main/h3-center/docs/MINIMAX_H3_REFMODS_INSTALLATION_AND_USAGE_GUIDE.md" target="_blank" rel="noopener">install &amp; use in ComfyUI</a>. Custom node: <a href="https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod" target="_blank" rel="noopener">ComfyUI-MiniMaxH3Mod</a>.</p>
      </section>
    </div>
  `;
  $$("#guide-toc [data-jump]").forEach((btn) => {
    btn.onclick = () => {
      const el = document.getElementById(btn.dataset.jump);
      if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
    };
  });
  bindBackToTop();
}

function renderSetup() {
  const h = state.health || {};
  const c = (h.config || {});
  $("#app").innerHTML = `
    <div class="toolbar"><h2>Setup</h2></div>
    <div class="layout">
      <section class="side" style="position:static">
        <p class="meta">Point this at <b>any</b> local ComfyUI. Paste the folder with <span class="mono">main.py</span>, or a wrapper such as <span class="mono">ComfyUI_windows_portable</span> / <span class="mono">ComfyUI-Easy-Install</span>. Desktop: pick the user data folder you chose at install (often <span class="mono">Documents\\ComfyUI</span>), not the electron app folder. Leave Python blank and Save to auto-fill <span class="mono">python_embeded</span>, <span class="mono">.venv</span>, or <span class="mono">venv</span>.</p>
        <label>ComfyUI root${pickRow("s-comfy", c.comfy_root || h.comfy_root || "", "folder", { title: "ComfyUI folder", placeholder: "ComfyUI, portable, Easy-Install, or Desktop data folder" })}</label>
        <label>Python${pickRow("s-py", c.python_path || h.python_path || "", "file", { title: "Python that launches ComfyUI", filetypes: [["python.exe", "python.exe"], ["python", "python"], ["All files", "*.*"]] })}</label>
        <label>H3 video VAE${pickRow("s-vae", c.vae_path || h.vae_path || "", "file", { title: "MiniMax H3 video VAE", filetypes: [["Safetensors", "*.safetensors"], ["All files", "*.*"]] })}</label>
        <label>extract_mod.py${pickRow("s-script", c.extract_script || h.extract_script || "", "file", { title: "extract_mod.py", filetypes: [["extract_mod.py", "extract_mod.py"], ["Python", "*.py"], ["All files", "*.*"]] })}</label>
        <label>RefMods output${pickRow("s-out", c.output_dir || h.output_dir || "", "folder", { title: "RefMods output folder" })}</label>
        <label>Datasets root${pickRow("s-ds", c.datasets_root || h.datasets_root || "", "folder", { title: "Datasets folder" })}</label>
        <label>Mod name prefix<input id="s-prefix" type="text" value="${escapeHtml(c.mod_prefix || h.mod_prefix || "sdprompts")}"></label>
        <p class="meta">New mods are named <span class="mono">{prefix}_minimaxh3_{dataset}_v1_refmod</span>. The RefMod folder on each dataset is prepended, so ComfyUI lists <span class="mono">identity/name</span>.</p>
        <h3>Hugging Face</h3>
        <p class="meta">Write token from <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noopener">huggingface.co/settings/tokens</a>. Repo is <span class="mono">user/name</span>. Uploads keep the same folder path as ComfyUI.</p>
        <label>Access token<input id="s-hf-token" type="password" placeholder="${c.hf_token_set ? "saved — leave blank to keep" : "hf_…"}" autocomplete="off"></label>
        <label>Repo<input id="s-hf-repo" type="text" value="${escapeHtml(c.hf_repo || h.hub_repo || "")}" placeholder="username/h3-refmods"></label>
        <label>Repo type
          <select id="s-hf-type">
            <option value="dataset" ${(c.hf_repo_type || "dataset") === "dataset" ? "selected" : ""}>dataset</option>
            <option value="model" ${c.hf_repo_type === "model" ? "selected" : ""}>model</option>
          </select>
        </label>
        <label class="check"><input id="s-hf-private" type="checkbox" ${(c.hf_private || "true") !== "false" ? "checked" : ""}> Private repo</label>
        <button class="btn primary" id="s-save">Save & probe</button>
      </section>
      <aside class="side">
        <h3>Health</h3>
        <div>${h.ready ? `<span class="badge ok">ready</span>` : `<span class="badge dirty">blocked</span>`}</div>
        <div class="meta">Install: ${escapeHtml(h.install_label || h.install_kind || "unknown")} · Device: ${escapeHtml(h.device || "unknown")}${h.pack_version ? ` · MiniMaxH3Mod ${escapeHtml(h.pack_version)}` : ""}</div>
        ${(h.missing || []).map((m) => `<div class="warn">${escapeHtml(m)}</div>`).join("")}
        <div class="meta">Hugging Face: ${h.hub_ready ? `<span class="badge ok">ready</span> ${escapeHtml(h.hub_repo || "")}` : `<span class="badge dirty">not set</span>`}</div>
        ${(h.hub_missing || []).map((m) => `<div class="warn">${escapeHtml(m)}</div>`).join("")}
        <div class="mono">${escapeHtml((h.probe_detail && h.probe_detail.comfy) || "")}</div>
        <p class="meta">Install the pack with <span class="mono">git clone https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod</span> into that install’s <span class="mono">custom_nodes</span>. Place <span class="mono">minimax_h3_video_vae_fp16.safetensors</span> in <span class="mono">models/vae</span> (or another VAE folder listed in <span class="mono">extra_model_paths.yaml</span>).</p>
      </aside>
    </div>
  `;
  bindPickers();
  $("#s-save").onclick = async () => {
    state.health = await api("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        comfy_root: $("#s-comfy").value.trim(),
        python_path: $("#s-py").value.trim(),
        vae_path: $("#s-vae").value.trim(),
        extract_script: $("#s-script").value.trim(),
        output_dir: $("#s-out").value.trim(),
        datasets_root: $("#s-ds").value.trim(),
        mod_prefix: $("#s-prefix").value.trim(),
        hf_token: $("#s-hf-token").value.trim(),
        hf_repo: $("#s-hf-repo").value.trim(),
        hf_repo_type: $("#s-hf-type").value,
        hf_private: $("#s-hf-private").checked,
      }),
    });
    setChip();
    renderSetup();
  };
}

async function showDatasets(alive) {
  state.datasetsLoading = true;
  renderDatasets();
  try {
    await loadDatasets();
  } catch (err) {
    if (!alive()) return;
    state.datasetsLoading = false;
    if ((state.datasets || []).length) {
      renderDatasets();
      return;
    }
    throw err;
  }
  if (!alive()) return;
  state.datasetsLoading = false;
  renderDatasets();
}

async function showLibrary(alive) {
  const pane = libraryPane();
  if (pane === "studio") {
    state.studioLoading = true;
    renderStudio();
    try {
      await loadStudio();
    } catch (err) {
      if (!alive()) return;
      state.studioLoading = false;
      throw err;
    }
    if (!alive()) return;
    state.studioLoading = false;
    renderStudio();
    return;
  }
  state.libraryLoading = true;
  renderLibrary();
  try {
    const cached = (state.library || []).length || (state.folders || []).length;
    if (!cached) {
      const preview = await loadLibrary(true);
      if (!alive()) return;
      renderLibrary();
      if (!preview.lite) {
        state.libraryLoading = false;
        renderLibrary();
        return;
      }
    }
    await loadLibrary(false);
  } catch (err) {
    if (!alive()) return;
    state.libraryLoading = false;
    if ((state.library || []).length || (state.folders || []).length) {
      renderLibrary();
      return;
    }
    throw err;
  }
  if (!alive()) return;
  state.libraryLoading = false;
  renderLibrary();
}

async function render() {
  const gen = ++renderGen;
  setTab();
  const { view, slug } = route();
  const alive = () => gen === renderGen && route().view === view && route().slug === slug;
  try {
    if (view === "studio") {
      location.hash = "#/library/studio";
      return;
    } else if (view === "library") {
      await showLibrary(alive);
    } else if (view === "design" || view === "voice" || view === "voices") {
      location.hash = "#/datasets";
      return;
    } else if (view === "instructions") {
      renderInstructions();
    } else if (view === "prompts") {
      renderPrompts();
    } else if (view === "setup") {
      renderSetup();
    } else if (view === "datasets" && slug) {
      await loadDataset(slug);
      state.log = [];
      state.result = null;
      state.error = null;
      state.progress = null;
      state.extractFiles = [];
      state.clean = null;
      renderDataset();
    } else {
      await showDatasets(alive);
    }
  } catch (err) {
    if (!alive()) return;
    $("#app").innerHTML = `<p class="empty">${escapeHtml(err.message)}</p>`;
  }
}

window.addEventListener("hashchange", render);
window.addEventListener("load", () => {
  loadHealth().catch(() => {
    $("#health-chip").textContent = "api down";
    $("#health-chip").className = "health-chip bad";
  });
  render();
});
