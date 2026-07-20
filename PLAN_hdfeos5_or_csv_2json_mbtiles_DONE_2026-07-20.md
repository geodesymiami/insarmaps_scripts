# Plan: Point-native refactor of `hdfeos5_or_csv_2json_mbtiles.py`

**Status:** Implemented 2026-07-20 (archived plan; not an active TODO list).  
**Open work:** see `PLAN_EFFICIENCY_hdfeos5_2json_mbtiles.md` (separate) and EGMS ingest (deferred).

## Summary

Refactor the converter so CSV/point data no longer pads into a fake √N×⌈N/√N⌉ grid. Insarmaps consumes GeoJSON **Point** features (`d`, `m`, `p` + lat/lon); the grid was only a legacy adapter for HDFEOS-style `create_json`. Keep CLI, tippecanoe flags, metadata.pickle shape, and HDFEOS behavior equivalent.

**Out of scope for this PR:** EGMS lat/lon candidates, XML enrichment, CLI metadata overrides (separate follow-up).  
**Do not modify** `hdfeos5_2json_mbtiles.py` (reference only).

## Will removing the fake grid be faster / more efficient?

**Yes for CSV — mainly CPU and code path simplicity; pad memory savings are small.**

| Effect | Magnitude | Why |
|--------|-----------|-----|
| **CPU in JSON workers** | **Large** | Today each chunk does `ndenumerate` from the start of the 2D array and `continue` until `start_idx`, then `break` after `end_idx`. Later chunks re-scan all earlier indices → roughly **O(n² / CHUNK_SIZE)** index checks. Point-native `for i in range(start, end)` is **O(n)** total. |
| **Pad RAM** | **Small** | Fake square pad waste is typically **&lt; ~2%** (often ≪1% for large N). Not why EGMS OOMs. |
| **Timeseries layout** | **Moderate clarity / pickling** | 1D `(n_dates, n_points)` is simpler; same order of data size. Multiprocessing still pickles full `timeseries` per task unless we change that separately (`MemoryIssues.md`). |
| **Tippecanoe / MBTiles** | **None** | Same number of real Point features written. |
| **HDFEOS** | **Same CPU win** if flattened then ranged by index instead of ndenumerate+continue | Real grid still exists on disk; we only change the walk. |

So: expect **noticeably faster JSON generation** on large CSVs; **not** a cure for multi-GB RAM from loading the full series into workers.

## Attributes: `hdfeos5_2json_mbtiles.py` vs `hdfeos5_or_csv_…` (reference only)

Git history: **attribute/CSV work landed in `hdfeos5_or_csv_2json_mbtiles.py`** (2025–2026). `hdfeos5_2json_mbtiles.py` is older (last notable touch ~Nov 2025 tippecanoe stderr) and still grid-only. Use it as the **simpler upload filter model**, not as “more recently updated attributes.”

| Topic | `hdfeos5_2json_mbtiles.py` | `hdfeos5_or_csv_2json_mbtiles.py` |
|-------|----------------------------|-------------------------------------|
| Upload filter | `k in attributes` ∩ `needed_attributes` | Same, **plus** forced `attribute_keys.append("dem")` |
| Missing keys | Omitted (not filled) | CSV path fills via `add_dummy_attribute` (Galapagos) + calculated REF/dates/footprint |
| High-res | Drop `X_STEP`,`Y_STEP`,`X_FIRST`,`Y_FIRST` from needed set | Same |
| `WIDTH`/`LENGTH` | Real he5 dims; stay in needed set | Fake √N dims today |
| Bug | **Missing comma** after `"min_baseline_perp"` → set contains `"min_baseline_perpunwrap_method"`, loses both real keys | Comma fixed |

**Attribute policy for this refactor (or_csv only):**

1. Keep upload rule: only keys present in both `attributes` and `needed_attributes` (match he5-only philosophy).
2. High-res (CSV): also drop `X_STEP`,`Y_STEP`,`X_FIRST`,`Y_FIRST`; **also drop `WIDTH`/`LENGTH`** from the needed copy so fake/point dims are not uploaded (cleaner than inventing √N or `LENGTH=1`).
3. Keep CSV **real** enrichment: `add_calculated_attributes`, footprint from points, optional standardized filename enrich — not Galapagos for those.
4. Keep `add_dummy_attribute` for now so existing CSV ingest still satisfies Insarmaps’ long attribute list (behavior preserve); do **not** expand dummies. Optional later PR: shrink `needed_attributes` like the he5 comment (“TODO: … extra_attributes”).
5. Keep forced `"dem"` popup key in or_csv (he5-only lacks it; or_csv added it for elevation UI) unless we confirm it is unused for pure point clouds — default **keep**.
6. Do not “fix” the missing-comma bug in `hdfeos5_2json_mbtiles.py` (user: do not modify that file).

## Key Components Affected

- `tools/insarmaps_scripts/hdfeos5_or_csv_2json_mbtiles.py` — main refactor
- `minsar/insarmaps_utils/insarmaps_csv_geo.py` — keep LAT/LON candidate lists in sync if extracted/shared
- New unit tests under `tools/insarmaps_scripts/tests/` (or colocated if repo convention differs)
- `tools/insarmaps_scripts/MemoryIssues.md` — update grid wording to point-chunk model for CSV
- Optionally thin split into helpers in same directory (only if file stays readable; prefer one file first unless split is clearly cleaner)

## Action Items

- [x] Add unit tests for CSV→feature behavior (tiny fixture)
- [x] Introduce point-native JSON writer + chunking by point index
- [x] Change `read_from_csv_file` to return 1D arrays — no fake grid pad
- [x] Flatten HDFEOS 2D slices once and reuse the same point writer (`p` = row-major index)
- [x] Localize `needed_attributes` mutations (copy per run); drop WIDTH/LENGTH in no-geo-step mode
- [x] Update MemoryIssues.md
- [x] Run unit tests (`python -m unittest tests.test_hdfeos5_or_csv_points`)
- [ ] EGMS lat/lon / XML — **out of scope** (explicitly deferred)

## Execution Plan (Detailed Change Instructions)

### 1. Stabilize contract: `ConvertInput`-like return

Both readers feed the same downstream shape:

```text
attributes: dict
dates: list[str]
decimal_dates: list[float]
timeseries: dict[str, np.ndarray]   # date -> 1D length n_points  (OR keep 2D for he5 and flatten at convert)
lats, lons: 1D float arrays length n_points
folder_name: str
quality: dict[str, 1D array] | None
shm: SharedMemory | None
```

Prefer **1D everywhere after load** so `create_json` never needs `num_rows`/`num_columns`.

### 2. Replace `create_json` grid walk with point loop

**Before:** `ndenumerate` over 2D first slice; skip by flattened index range; index quality as `arr[row][col]`.

**After:**

```python
for i in range(start_idx, end_idx + 1):
    if math.isnan(first_values[i]):
        continue
    # displacements from timeseries[date][i]
    # quality from quality[key][i]
    # properties p = i   # same as today's point_num for non-padded CSV
```

Chunking: split `0 .. n_points-1` into chunks of 20000 (same `CHUNK_SIZE`).

### 3. CSV reader: drop fake grid

Remove:

- `num_rows = int(np.sqrt(...))` / pad / reshape to 2D
- Fake `WIDTH`/`LENGTH` as √N dimensions (optional: still set `WIDTH`/`LENGTH` to `n_points` and `1`, or omit from uploaded keys for high-res — **prefer keep keys present with `WIDTH=n_points`, `LENGTH=1` or drop from `needed_attributes` only if high-res already drops step keys; do not invent √N**)

Recommended for CSV attributes:

- Do **not** invent √N `WIDTH`/`LENGTH`. In high-res mode, drop `WIDTH`/`LENGTH` from the per-run `needed_attributes` copy (same idea as dropping `X_STEP`). Optionally still store internal counts for logging only — they need not be uploaded.

Preserve:

- mm→m scaling
- SARvey `DYYYYMMDD` vs bare `YYYYMMDD`
- quality field names/behavior (`dem_error`, `elevation`, `coherence`, `omega`, `st_consist`, `point_ID`)
- `folder_name` from CSV stem
- `high_res_mode` (no `X_STEP`/`Y_STEP`) → same tippecanoe command

**`p` indexing for CSV:** Today padded cells are skipped and `point_num` advances only for valid first-date points, starting from chunk `start_idx` (which includes pad holes). After removing padding, `p` should be the **original CSV row index** `0 .. n_points-1` for valid points (skip NaN first-date rows without renumbering). Document this carefully in tests — slight `p` shifts vs old padded layout are acceptable for CSV because pad holes never produced features; contiguous valid points should keep the same `p` as old code when there was no pad gap before the point (i.e. when `n_points` was a perfect rectangle). When `n_points` was not rectangular, old `p` values jumped over pad indices — **new `p` = CSV row index is cleaner and preferred**; call out in summary as intentional CSV `p` cleanup (Insarmaps only requires unique `p` within a dataset).

### 4. HDFEOS reader: flatten then share writer

After building per-date 2D arrays (and lat/lon 2D):

```python
n_rows, n_cols = lats.shape
lats_1d = lats.reshape(-1)
lons_1d = lons.reshape(-1)
timeseries_1d = {d: arr.reshape(-1) for d, arr in timeseries_datasets.items()}
# p = i = row * n_cols + col  → identical to current point_num for valid pixels
```

Keep shared memory lifetime/cleanup in `main` unchanged.

### 5. `convert_data` / metadata cleanup (no behavior change)

- Copy `needed_attributes` before removing high-res keys
- Remove unused `x_step`/`y_first` locals (or keep only if still needed — they are not)
- Midpoint via `nanmean` on 1D lat/lon
- Same pickle keys/values structure
- Same tippecanoe high-res vs low-res command strings
- Prefer `subprocess.run` over `os.system` **only if** command string is identical; otherwise leave `os.system` for this PR to avoid shell-glob differences

### 6. Light structural cleanup (same file OK)

Within one file (or minimal helpers module if preferred):

- `detect_lat_lon_columns`
- `extract_time_series_from_df`
- `build_quality_arrays` (1D)
- `create_json_points` / `generate_point_worker_args`
- Keep `enrich_attributes_from_csv_filename`, `add_dummy_attribute`, `add_calculated_attributes` logic identical

Sync LAT/LON candidate lists with `minsar/insarmaps_utils/insarmaps_csv_geo.py` (still no EGMS `latitude`/`longitude` in this PR unless trivial case-insensitive add is approved — **default: no EGMS change here**).

### 7. Tests

Add small unittest fixtures (few points, few dates):

1. CSV SARvey-style (`X`,`Y`,`D20200101`,…) → features have correct coords, `d` in meters, unique `p`
2. CSV bare `YYYYMMDD` + `Latitude`/`Longitude` → same (existing candidates)
3. Point chunking: `n_points > CHUNK_SIZE` split (use small CHUNK_SIZE override in test if needed)
4. HDFEOS flatten equivalence: synthetic 2×3 grid → `p` = row-major indices for non-NaN cells
5. `high_res_mode` True when `X_STEP` missing

No tippecanoe / full ingest in unit tests.

### 8. Docs

Update `MemoryIssues.md`: CSV/path uses point chunks, not WIDTH×LENGTH fake grid; HDFEOS flattens then chunks.

## Key Commands & Flows

```bash
# After refactor, same CLI:
hdfeos5_or_csv_2json_mbtiles.py file.csv ./JSON --num-workers 2
hdfeos5_or_csv_2json_mbtiles.py file.he5 ./JSON --num-workers 2

# Tests (to be wired):
python -m unittest tools.insarmaps_scripts.tests.test_hdfeos5_or_csv_points
# or path-local discovery once tests land
```

Data flow:

```text
.csv → 1D points → JSON chunks → tippecanoe (high-res) → .mbtiles + metadata.pickle
.he5 → 2D → flatten 1D → same JSON/tippecanoe path (low-res tippecanoe if X_STEP present)
```

## TODO List

- [ ] Write tests for existing / intended point behavior
- [ ] Implement point-native convert + CSV/HDFEOS loaders
- [ ] Add tests for new 1D path and HDFEOS `p` equivalence
- [ ] Update MemoryIssues.md
- [ ] Run unit tests; smoke CSV if a small sample is available

## Risks / intentional deltas

| Risk | Mitigation |
|------|------------|
| CSV `p` differs when old pad holes existed | Prefer CSV-row `p`; document; Insarmaps needs uniqueness not absolute values |
| HDFEOS `p` drift | Flatten row-major; test on synthetic grid |
| Multiprocessing pickling still heavy | Same pattern for now; optional follow-up: shared arrays / fork-only |
| Nested repo `tools/insarmaps_scripts` | Commit there; MinSAR only if `insarmaps_csv_geo` synced |

## Approval gate

**Stop here.** Do not implement until this plan is explicitly approved (or amended).
