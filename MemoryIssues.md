# Memory and performance notes: `hdfeos5_or_csv_2json_mbtiles.py`

This document summarizes how parallelization works in `hdfeos5_or_csv_2json_mbtiles.py` (and the closely related `hdfeos5_2json_mbtiles.py`), why `--num-workers` can trigger OOM on large products, how to choose worker counts, and how the tippecanoe step fits in.

See also `PLAN_EFFICIENCY_hdfeos5_2json_mbtiles.md` for a fuller efficiency plan (geocoded vs radar-coded terminology, tippecanoe, date scaling).

**CSV / point path (updated):** CSV no longer pads into a fake √N grid. Points are 1D arrays; JSON workers iterate `range(start, end)` over point indices. HDFEOS5 is flattened row-major to the same 1D writer (`p` = flat index).

---

## How parallelization is implemented

1. **Input load (HDFEOS5 `.he5`)**  
   The full displacement cube is read from the file, masked, then copied into **`multiprocessing.shared_memory.SharedMemory`**.  
   `timeseries_datasets` maps each date to a **1D** array of length `WIDTH × LENGTH` (row-major flatten).

2. **Work splitting**  
   Points are split into chunks of **20,000 indices** (`CHUNK_SIZE_DEFAULT`).  
   Example: `N = 6669 × 2599 ≈ 1.73×10⁷` → `ceil(N / 20000) ≈ 867` tasks.

3. **Execution**  
   A `multiprocessing.Pool` with `num_workers` runs **`Pool.starmap(create_json, worker_args)`**.  
   Each task still receives the full `timeseries_datasets` / `lats` / `lons` in the argument tuple (pickling cost unchanged; see below).

4. **Tippecanoe**  
   After JSON chunks and `metadata.pickle` are written, the script **`os.chdir`**s to the output directory and runs **`tippecanoe`** via `os.system` on all `*.json` files, producing a single `.mbtiles` file. This is **not** parallelized inside Python; tippecanoe is its own process.

---

## Why `--num-workers 3` can fail with `MemoryError` (queue / unpickle)

The traceback you saw:

```text
File ".../multiprocessing/queues.py", line 367, in get
    return _ForkingPickler.loads(res)
MemoryError
```

occurs during **`multiprocessing` IPC**: worker processes pull tasks from a queue and **unpickle** the task payload.

Important detail: **`starmap` sends each task’s arguments through that pickling path.** Every chunk task includes **references to the full `timeseries_datasets` dict and large `lats` / `lons` arrays** (and the rest of the tuple), not just the small index range. So for hundreds of chunks, the **same huge argument bundle is serialized repeatedly** for the worker queue. That can inflate memory use **far beyond** a single in-memory copy of the dataset and makes **higher `--num-workers`** more likely to hit OOM, because more workers can mean more concurrent unpickling and larger queue pressure.

Shared memory for the 3D cube **does not** remove this duplication for the task payloads: the per-date 2D arrays in `timeseries_datasets` are still part of what gets passed to each task.

So the failure mode is **not** simply “3× the data because of 3 workers”; it is **multiprocessing’s serialization + queue depth + worker concurrency** interacting with **very large pickled arguments per task**.

---

## How `--num-workers` affects memory requirements

**Short answer: yes—using more workers often increases peak memory in this script, but not as a simple “N workers ⇒ N × dataset size.”** Several mechanisms stack:

### 1. Base memory (mostly independent of worker count)

The parent process loads the cube into **shared memory**, builds **`timeseries_datasets`**, **`lats`**, **`lons`**, and the list of **~hundreds of chunk tasks**. That footprint is driven by **pixels × dates** (and lat/lon grids), not primarily by whether you pick 1 or 4 workers.

### 2. Worker processes and copy-on-write (can grow with more workers)

Worker processes are created with **`fork`** (typical on Linux). At fork time they **share** read-only pages with the parent (copy-on-write). When each worker runs **`create_json`**, it reads numpy slices and builds Python/JSON structures, which **touches** memory pages. That can **private duplicate** parts of what each worker accesses. **More workers ⇒ more processes doing that at once**, so **resident set size (RSS)** can rise compared to a single worker, even though the *logical* dataset is not replicated N full times in advance.

### 3. Task queue + unpickling (usually the dominant extra cost when workers > 1)

**`Pool.starmap`** must send **each chunk’s arguments** to a worker through **pickle**. Every task repeats the **same large tuple** (including the full **`timeseries_datasets`** dict and **`lats`/`lons`**). That means:

- **Temporary buffers** during **serialize** (parent) and **unpickle** (workers).
- **Several workers** may **unpickle different tasks at the same time**, so peak memory can spike with **higher `--num-workers`** even if the underlying arrays are shared in the parent.

This is why **increasing workers can push a job from “fits” to OOM** without changing the input file.

### 4. What does *not* happen

- Memory is **not** guaranteed to stay **flat** when you add workers: the **IPC path** and **multiple active worker heaps** work against that.
- Memory is **not** a clean **linear** function of worker count (e.g. exactly 2× for 2 workers); it depends on **queue backlog**, **timing**, and the **OS**.

### Practical takeaway

Treat **higher `--num-workers` as likely increasing peak RAM** for this script, especially from **pickling/unpickling** and **concurrent workers**. If you are near the memory limit, **prefer fewer workers** (often `1` or `2`) even if that is slower.

---

## Choosing `--num-workers`

Practical guidance:

1. **Start conservative** (`1` or `2`) on large grids or long time series. If you see `MemoryError` in `multiprocessing/queues` or `pickle`, **reduce workers** before anything else.

2. **Increasing workers does not reliably speed things up** here: each task still scans the relevant slice of the grid; the dominant cost may be **CPU per point**, **I/O writing JSON**, or **pickling overhead**, not parallelizable speedup.

3. **There is no single formula** in the script that ties `--num-workers` to pixels, dates, or file size. Because of the **per-task pickling** behavior, **peak memory does not scale as a simple function** of `(pixels, dates, workers)`; it depends on Python’s multiprocessing implementation and how the OS handles the queue.

**Rule of thumb:** use the **fewest workers that finish without OOM** and stay within your SLURM memory request; if `2` works and `3` fails, **`2` is the safe setting** for that dataset on that node.

---

## What drives memory primarily: pixels or dates?

**Both**, at different stages:

| Factor | Role |
|--------|------|
| **Pixels** (`WIDTH × LENGTH`) | Sets the size of each 2D slice and the total number of grid points to iterate. Dominates **base** array size: roughly `n_dates × n_rows × n_cols × itemsize` for the 3D cube plus lat/lon grids. |
| **Dates** (`n_dates`) | Multiplies the size of the 3D stack and **the length of the `d` array** stored in every GeoJSON feature. More dates → larger JSON and more work per point. |
| **Chunk count** | `ceil(num_points / 20000)` tasks; **more chunks → more duplicate task arguments** through the multiprocessing queue. |

So **pixels** dominate **spatial** memory; **dates** multiply **temporal** depth and **per-feature JSON size**. **Neither alone** explains the OOM in the queue; **task serialization** is the aggravating factor for `--num-workers > 1`.

---

## Can an “optimal” `--num-workers` be calculated from pixels/dates/file size?

**Not in a reliable closed form** from those numbers alone, without measuring on your cluster, because:

- Peak usage depends on **pickle size per task**, queue buffering, and **how many workers** unpickle at once.
- **File size** on disk (compressed HDF5) does not equal **RAM** after loading and dict unpacking.

**Empirical approach:** run with `--num-workers 1` on a representative node, note runtime and max RSS (e.g. `time`/`/usr/bin/time -v`, or SLURM `MaxRSS`). Then try `2`, then `3`, only if memory allows.

---

## Other ways to make the workflow more efficient (without changing this script)

- **SLURM memory**: Request enough RAM for the loaded cube + lat/lon + JSON + multiprocessing overhead; if you must use multiple workers, **increase mem** rather than only increasing workers.
- **Avoid unnecessary reruns**: the script prints if the output directory already exists but still **regenerates JSON and reruns tippecanoe**; keep outputs separate or archive completed tiles so you do not repeat work.
- **I/O**: JSON chunk output is many small-ish writes; **fast local scratch** (`$SCRATCH`, node-local SSD) helps if the job is I/O bound after JSON creation.
- **Long-term fix (would require code changes elsewhere)**: use a multiprocessing pattern that **does not pickle full `timeseries_datasets` per task** (e.g. initializer + globals, or `fork` with read-only shared buffers only, or `imap` with a single shared memory block). The current script does not do that.

---

## Is tippecanoe a bottleneck?

Often **yes**, for large datasets:

- It runs **after** all JSON chunks exist and must read **all** `*.json`, build vector tiles, and write **one** `.mbtiles`.
- It can be **CPU- and single-process heavy** (depending on tippecanoe version and flags), so wall time may be dominated by this step even when JSON conversion was parallel.

The script uses **`-P`** (parallel input readers in tippecanoe) and, for non–high-res mode, extra flags like `-Bg -d9 -D12 -g12 -r0` that affect **tile detail and work**.

---

## Running tippecanoe as a second job / SLURM step

**As shipped, the script always runs tippecanoe in the same process after conversion** (`os.system(cmd)` in `main()`). There is **no** `--skip-tippecanoe` flag in the current script.

**If you could split the pipeline** (e.g. a future flag or a small wrapper), the second step would be:

1. `cd` to the **output directory** that contains `chunk_*.json` and `metadata.pickle`.
2. Run the same `tippecanoe` command the script builds:
   - **High-res mode** (no `X_STEP`/`Y_STEP` in attributes):  
     `tippecanoe *.json -P -l chunk_1 -x d -pf -pk -o <folder_name>.mbtiles 2> tippecanoe_stderr.log`
   - **Else**:  
     `tippecanoe *.json -P -l chunk_1 -x d -pf -pk -Bg -d9 -D12 -g12 -r0 -o <folder_name>.mbtiles 2> tippecanoe_stderr.log`  
   where `<folder_name>` is the basename of the input file (without extension), matching what the script uses.

**Without modifying the script**, you cannot skip the built-in tippecanoe run; the practical two-job workflow would require **either** a small code change (skip tippecanoe) **or** accepting that tippecanoe runs twice (not recommended). **Documenting the command above** is still useful if you later add a skip flag or maintain a fork.

---

## Summary

- Parallelism is **per-chunk JSON generation** via `Pool.starmap`, **fixed chunk size 20,000 grid cells**, **not** parallel tippecanoe.
- **`--num-workers` often increases peak memory** (not a fixed multiple of dataset size): **more workers** ⇒ more **concurrent unpickling** and **worker heaps**, on top of **per-task pickling** of large arguments for every chunk.
- OOM with more workers is consistent with **large pickled task arguments** repeated for every chunk.
- **Pixels and dates** both matter for base memory and output size; **worker count** interacts badly with **multiprocessing’s serialization**.
- **Tippecanoe** is often the slow final step; **running it separately** is feasible only if the conversion step can be told not to invoke it (not available in the unmodified script).

---

*Generated for analysis of `hdfeos5_or_csv_2json_mbtiles.py` / `hdfeos5_2json_mbtiles.py` behavior.*
