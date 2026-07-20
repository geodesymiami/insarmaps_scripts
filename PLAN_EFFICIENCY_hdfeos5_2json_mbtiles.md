# Plan: Efficiency of `hdfeos5_2json_mbtiles.py`

**Status:** Analysis / suggestions only — implement later.  
**Scope:** `tools/insarmaps_scripts/hdfeos5_2json_mbtiles.py` (primary). Behavior is shared in large part with `hdfeos5_or_csv_2json_mbtiles.py` for the `.he5` path; CSV-specific notes called out where relevant.  
**Related:** `MemoryIssues.md` (earlier notes; this doc supersedes terminology and expands tables).

---

## 1. Terminology: geocoded vs radar-coded (not “high-res”)

Do **not** call products “high-res” in this pipeline. Both **MintPy** and **MiaplPy** can be either:

| Mode | Typical meaning | How the converter detects it today | Insarmaps / tippecanoe effect |
|------|-----------------|------------------------------------|-------------------------------|
| **Geocoded** | Values on a regular geographic **xy** grid (`X_FIRST`/`Y_FIRST` + `X_STEP`/`Y_STEP`) | Attributes include `X_STEP` and `Y_STEP` | Tippecanoe with drop/detail flags (`-Bg -d9 -D12 -g12 -r0`); UI can draw rectangular pixels from step |
| **Radar-coded** | Values on a radar (az/rg) index grid; lat/lon are **per-pixel** geometry layers, not a regular lon/lat step | Missing `X_STEP` and/or `Y_STEP` → current `high_res_mode()` | Tippecanoe **without** those detail flags (`-P -l chunk_1 -x d -pf -pk` only); UI treats features as points |

**Important:** In both cases the `.he5` on disk is still stored as **2D slices** (`LENGTH` × `WIDTH`) times `n_dates`. Radar-coded does **not** mean “already a 1D point list” inside HDFEOS5 — it means there is no regular geographic step metadata. Many cells may be NaN (especially sparse PS-like products).

Suggested rename in a future code cleanup: `high_res_mode` → `is_radar_coded` / `lacks_geo_step` (behavior unchanged).

---

## 2. Is `--num-workers` properly implemented?

### What it does today

```text
Pool(num_workers).starmap(create_json, worker_args)
```

- Grid flattened conceptually to `N = WIDTH × LENGTH` indices.
- Split into chunks of **`CHUNK_SIZE = 20000`** cells → hundreds of tasks on large products.
- **Each task argument tuple includes the full `timeseries_datasets` dict** (one 2D array per date), plus full `lats`/`lons`, dates, and the index range.

Shared memory is allocated for the 3D displacement cube in `main()`, and per-date entries in `timeseries_datasets` are views into that buffer — **but `starmap` still pickles those arrays into the task queue**. SharedMemory therefore does **not** prevent large IPC copies.

### Why `--num-workers=1` often “works better”

Your impression is well-founded:

1. **Peak RAM rises with workers** via concurrent unpickle + worker heaps (see table below).
2. **Pickle overhead** can dominate or cancel CPU gains when the payload is huge (pixels × dates).
3. **Chunk walk is inefficient:** each `create_json` uses `np.ndenumerate` from the start of the 2D array and `continue` until `start_idx`. Later chunks re-scan earlier indices → wasted CPU that **does not parallelize cleanly**.
4. On memory-constrained nodes (Jetstream, tight SLURM `--mem`), workers >1 → `MemoryError` in `multiprocessing/queues` during unpickle; workers=1 avoids that path’s worst spikes.

So: the option is **wired correctly** (Pool really uses N processes), but the **parallel design is not memory-safe or always faster**. It is only “proper” in the narrow sense that the flag is honored.

### Advantages and drawbacks of higher `--num-workers`

| | Advantages | Drawbacks |
|--|------------|-----------|
| **`--num-workers=1`** | Lowest peak RAM; simplest; often finishes when larger N OOMs; less pickle churn | Wall time for JSON phase is serial; underuses multi-core CPUs |
| **`--num-workers=2–4`** | Can speed JSON phase if RAM is abundant and pickle cost is tolerable | Higher peak RSS; more concurrent unpickles; OOM risk; speedup often **sublinear** |
| **`--num-workers` large (≫4)** | Rarely helpful for this script | Pickle/queue pressure grows; COW pages dirtied across many processes; tippecanoe still serial afterward so **end-to-end** gain shrinks |

**Practical rule:** use the **fewest workers that finish without OOM**. Prefer `1` near memory limits; try `2` only after measuring RSS on a representative product.

---

## 3. Tippecanoe: can it be made more efficient?

Tippecanoe runs **once**, after all `chunk_*.json` exist, via `os.system`, on **all** `*.json` in the output dir. It is **not** controlled by `--num-workers`.

### Current commands

- **Radar-coded** (no geo step):  
  `tippecanoe *.json -P -l chunk_1 -x d -pf -pk -o <name>.mbtiles`
- **Geocoded** (has geo step):  
  same plus `-Bg -d9 -D12 -g12 -r0`

Flags of note:

| Flag | Role |
|------|------|
| `-P` | Parallel readers inside tippecanoe (already on) |
| `-x d` | Drop property `d` from **tiles** (displacements fetched via DB/API later — good for tile size) |
| `-pf` / `-pk` | Keep all points / don’t drop dense points |
| `-Bg -d9 -D12 -g12 -r0` | Geocoded path: base zoom / detail / drop rate — **more tile work** |

### Efficiency ideas (future)

| Idea | Benefit | Risk / note |
|------|---------|-------------|
| **`--skip-tippecanoe`** + separate SLURM step | Isolate RAM: JSON job vs tippecanoe job; retry tippecanoe without reloading `.he5` | Needs small CLI change |
| Run tippecanoe on **fast local scratch**, then copy `.mbtiles` | Less I/O wait | Operational |
| Avoid `*.json` shell glob; pass an **explicit file list** or `@list` | Safer for huge chunk counts | Portability of tippecanoe version |
| **Concatenate NDJSON** to fewer/larger inputs before tippecanoe | Fewer open files; sometimes faster | Extra disk pass; test carefully |
| Tune geocoded detail flags | Faster / smaller tiles | Can change map appearance — needs visual QA |
| Ensure only `chunk_*.json` are inputs (not stray JSON) | Correctness + less work | Globs are brittle today |
| Tippecanoe version / `--read-parallel` equivalents | Possible speedups | Version-dependent |

Tippecanoe often **dominates wall time** on large products even when JSON used multiple workers — so fixing JSON alone may not fix end-to-end time.

---

## 4. Effect of 250 vs 500 acquisitions (dates)

Let:

- \(N_{xy} = \mathrm{LENGTH} \times \mathrm{WIDTH}\) (cells in the 2D layout)
- \(N_t\) = number of dates (acquisitions)
- \(N_{\mathrm{valid}}\) = non-NaN points actually written to JSON (≤ \(N_{xy}\))

### Memory (order of magnitude)

| Stage | Dependence on \(N_t\) | 250 → 500 |
|-------|----------------------|-----------|
| Loaded displacement cube | \(\propto N_t \times N_{xy}\) | **~2×** cube RAM |
| `timeseries_datasets` dict | \(\propto N_t\) entries, each size \(N_{xy}\) | **~2×** |
| Lat/lon grids | Independent of \(N_t\) | unchanged |
| Per GeoJSON feature property `d` | \(\propto N_t\) floats per valid point | **~2×** JSON text size |
| Pickle payload per chunk task | Includes all date slices → \(\propto N_t \times N_{xy}\) | **~2×** IPC weight |
| Tippecanoe input | Features still drop `d` from tiles (`-x d`), but must **read** JSON that contains full `d` | **~2×** JSON read volume |

### Processing time

| Stage | Dependence on \(N_t\) | 250 → 500 |
|-------|----------------------|-----------|
| Per-point gather of displacements | Loop over all dates → \(\propto N_t\) | **~2×** per valid point |
| Linear regression (`lstsq`) | Cost grows with \(N_t\) (roughly linear to mild superlinear) | **~2×** or a bit more |
| JSON serialize/write | Longer `d` arrays → \(\propto N_t\) | **~2×** write volume |
| Chunk index scan (current `ndenumerate`) | Mostly \(\propto N_{xy}\), weakly coupled to \(N_t\) | little change |
| Tippecanoe | Reads larger JSON; tile build still \(\propto N_{\mathrm{valid}}\) geometry | **noticeably slower** from I/O of larger `d` even though tiles drop `d` |

**Summary:** Doubling dates roughly **doubles** in-memory cube size, **doubles** per-point work and JSON size, and **worsens** multiprocessing pickle cost. Spatial size \(N_{xy}\) still sets how many points you visit; \(N_t\) multiplies cost **per point**.

---

## 5. Summary: what drives memory and time vs dimensions

Interpret dimensions as:

| Symbol | Meaning in `.he5` |
|--------|-------------------|
| **X** | `WIDTH` (columns) |
| **Y** | `LENGTH` (rows) |
| **Z** | \(N_t\) dates / acquisitions |
| **V** | \(N_{\mathrm{valid}}\) non-NaN points written (≤ X×Y) |

| Factor | Memory | JSON CPU time | Tippecanoe time | Notes |
|--------|--------|---------------|-----------------|-------|
| **X × Y** | Dominant for cube + lat/lon; pickle size | Dominant for cell visits; bad chunk scan \(\sim O((XY)^2 / \mathrm{chunk})\) worst case | Scales with **V** (and file count) | Radar-coded sparse: V ≪ XY → still pay XY in RAM today |
| **Z (dates)** | Multiplies cube & pickle & `d` length | Multiplies per-point gather + regression + JSON size | Larger JSON to parse | 250→500 ≈ 2× on Z-driven terms |
| **V (valid points)** | Little until JSON/features built | Features emitted ∝ V | **Main** tippecanoe geometric cost | Compacting to V early would help |
| **`--num-workers`** | Peak RSS ↑ (pickle + heaps) | Can ↓ wall time if RAM OK | No effect | Often `1` is safest |
| **Geocoded vs radar-coded** | Similar if same X,Y,Z | Similar JSON cost | Geocoded flags do **more** detail work | Naming only for tippecanoe/UI step attrs |
| **CHUNK_SIZE** | More chunks → more pickled tasks | Affects task granularity | More input files if many chunks | Fixed at 20000 today |
| **Quality layers** (or_csv only) | Extra arrays ∝ XY or V | Extra props per point | Slightly larger features | N/A in pure `hdfeos5_2json_mbtiles.py` |

Rough base cube memory (float32):

\[
\mathrm{RAM}_{cube} \approx 4 \cdot X \cdot Y \cdot Z \quad \text{(bytes)}
\]

Example: \(X{=}3000,\; Y{=}2000,\; Z{=}250\) → \(\approx 6\,\mathrm{GB}\) cube alone; \(Z{=}500\) → \(\approx 12\,\mathrm{GB}\), before lat/lon, Python overhead, JSON, and workers.

---

## 6. Suggestions to make it more efficient / memory-friendly

Ordered roughly by impact vs risk. **Implement later** (this file is the plan).

### A. Fix parallelization (highest leverage for `--num-workers`)

1. **Stop pickling full timeseries per task**  
   - Use `Pool` **initializer** + globals, or pass only SharedMemory **name** + shape/dtype + date list + index range.  
   - Goal: task payload = small metadata only.  
   - Then higher `--num-workers` can help without OOM.

2. **Replace `ndenumerate` + continue** with flat index loops  
   `for i in range(start, end+1): row, col = divmod(i, WIDTH)`  
   or `array.reshape(-1)` views.  
   Cuts wasted CPU especially on later chunks.

3. **Optional: `--num-workers 1` fast path**  
   Call `create_json` in-process (no Pool) to avoid pickle entirely.

### B. Radar-coded / sparse products (MintPy or MiaplPy)

4. **Compact to valid points early**  
   After mask: keep 1D `lats[V]`, `lons[V]`, `data[Z, V]` (or dict of 1D length V).  
   Memory and JSON time scale with **V×Z**, not **X×Y×Z**, when V ≪ X×Y.  
   Preserve stable `p` mapping (document whether `p` = original flat index or compact index — Insarmaps needs unique `p`).

5. **Geocoded dense products**  
   Compaction helps less if almost all cells are valid; still do (2)+(1).

### C. Tippecanoe / I/O

6. Add **`--skip-tippecanoe`** (and document the exact tippecanoe command).  
7. Write chunks to **local scratch**; run tippecanoe there; copy `.mbtiles` out.  
8. Prefer explicit chunk file list over `*.json`.  
9. Optionally merge NDJSON chunks before tippecanoe (benchmark).

### D. Dates / streaming (large Z)

10. **Avoid holding all date 2D arrays as a Python dict of separate arrays** if a single 3D SharedMemory + indexing suffices.  
11. Longer term: stream dates when building each point’s `d` (slower CPU, lower RAM) — only if RAM is the hard limit.  
12. Compress or quantize `d` in JSON (behavior change — needs Insarmaps QA).

### E. Operational defaults

13. Document: default recommendation **`--num-workers 1`** for large X×Y×Z; raise only after RSS checks.  
14. Request SLURM mem from estimate \(c \cdot 4 XYZ\) with \(c \gtrsim 2\)–3 for JSON+overhead (workers=1), higher if workers>1 **until** pickle fix lands.  
15. Rename `high_res_mode` → radar-coded detection in user-facing logs.

### F. Out of scope / do not confuse

- Changing Insarmaps DB schema.  
- Modifying tippecanoe flags without visual QA on geocoded layers.  
- Assuming radar-coded data are already 1D in the `.he5` — they are not until you compact.

---

## 7. Suggested implementation phases (later)

| Phase | Work | Expected win |
|-------|------|--------------|
| **P0** | Flat index loop; in-process path for `num_workers=1` | CPU + stability |
| **P1** | SharedMemory / initializer; tiny task payloads | Makes workers>1 actually useful |
| **P2** | Compact valid points for radar-coded (and optional geocoded mask) | RAM + time when sparse |
| **P3** | `--skip-tippecanoe`; scratch I/O; safer tippecanoe inputs | Wall time / ops |
| **P4** | Terminology rename; docs; default worker guidance | Clarity |

---

## 8. Quick answers checklist

| Question | Answer |
|----------|--------|
| Geocoded vs radar-coded? | Both MintPy/MiaplPy; distinguished by presence of geographic step attrs, not “high-res.” |
| Is `--num-workers` proper? | Flag works; design pickles full data per chunk → often worse with N>1. |
| Tippecanoe efficiency? | Already `-P` and `-x d`; split job, scratch disk, safer inputs, optional merge; geocoded flags cost more. |
| 250 vs 500 dates? | ~2× cube RAM, ~2× per-point/JSON cost; pickle and tippecanoe input I/O suffer similarly. |
| What scales with X,Y,Z? | See §5 table. |

---

*Document created for later implementation. Does not change `hdfeos5_2json_mbtiles.py`.*
