# SPAC-seq ELRR Data Readiness and Seal Activation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fail-closed SPAC-seq data layer that validates manifests, aligns RNA/guide observations, assigns perturbations conservatively, constructs development-only W/C episode metadata, and activates the prospective seal without reading external, final, or reserve values.

**Architecture:** A frozen section manifest and typed role policy sit in front of every raw loader. Platform-specific adapters emit two AnnData modalities with identical observations; a manifest-based assignment state machine, NTC-only expression transform, context/anchor builder, and episode generator then produce auditable development artifacts. The readiness runner returns either `data_ready` or a precise `data_not_ready` reason and creates `seal_activation.json` only after access-policy tests pass.

**Tech Stack:** Python 3.10+, NumPy, pandas, SciPy sparse/spatial, h5py, anndata/scanpy, pyarrow for Parquet metadata, standard-library zipfile/hashlib/json, assertion-based executable tests, CSV/JSON/H5AD artifacts.

---

## Scope and prerequisite

This is Plan A of two. Plan B (`docs/superpowers/plans/2026-07-21-spacseq-elrr-model-evaluation.md`) must not start until this plan writes `results/data_readiness/data_readiness_summary.json` with `status="data_ready"`.

Frozen scientific source:

```text
docs/superpowers/specs/2026-07-21-spacseq-elrr-design.md
design commit = 1535ab1b645ceec9644730d961507e45dc8b41cf
```

This planning turn creates no git commit. During execution, each listed commit step requires the execution session to have explicit commit authorization.

Raw inputs remain read-only:

```text
/mnt/172/wh/25-12/spatial/raw/SPAC-seq
/root/code/SPAC-seq
```

All writes stay under:

```text
repro_runs/2026-07-21-spacseq-elrr/
```

## File map

- `src/elrr/config.py`: frozen roles, thresholds, platform names, seeds, and source paths.
- `src/elrr/schema.py`: enums and dataclasses for sections, observations, assignments, episodes, and status values.
- `src/elrr/access.py`: role-aware value guards, pre-freeze log, seal activation, and content hashes.
- `src/elrr/manifest.py`: section/library manifest readers and cross-field validation.
- `src/elrr/io_visium.py`: exact-barcode Visium HD ingestion and spatial metadata validation.
- `src/elrr/io_stereo.py`: integer-coordinate Stereo ingestion and RNA/guide alignment.
- `src/elrr/assignment.py`: raw-UMI guide-to-target state machine.
- `src/elrr/preprocessing.py`: frozen gene selection and section-NTC transforms.
- `src/elrr/context.py`: physical-coordinate context, NTC density, anchors, and D4 transforms.
- `src/elrr/episodes.py`: W splits, C directions, support/readiness gates, and deduplication.
- `src/elrr/readiness.py`: orchestration and decision-oriented QC artifacts.
- `run_data_readiness.py`: development-only CLI; cannot access external/final/reserve values.
- `verify_data_artifacts.py`: artifact/schema/hash/readback verifier.
- `tests/run_data_tests.py`: fresh-process data-plan test aggregator.

### Task 1: Package scaffold, frozen roles, and typed schema

**Files:**
- Create: `repro_runs/2026-07-21-spacseq-elrr/src/elrr/__init__.py`
- Create: `repro_runs/2026-07-21-spacseq-elrr/src/elrr/config.py`
- Create: `repro_runs/2026-07-21-spacseq-elrr/src/elrr/schema.py`
- Create: `repro_runs/2026-07-21-spacseq-elrr/tests/test_schema.py`

- [ ] **Step 1: Write the failing schema test**

```python
from elrr.config import (
    DEVELOPMENT_SECTIONS,
    EXTERNAL_SECTIONS,
    FINAL_SECTIONS,
    RESERVE_SECTIONS,
    QUARANTINE_SECTIONS,
    GUIDE_MIN_RAW_UMI,
    TARGET_MIN_SCREEN_BINS,
    TARGET_MIN_SECTION_BINS,
    NTC_MIN_SECTION_BINS,
)
from elrr.schema import AssignmentStatus, DataRole, StageDecision

assert DEVELOPMENT_SECTIONS == (
    "lung_992",
    "mc38_libHD274E6",
    "mc38_libHD275E7",
    "mc38_libHD276E8",
    "mc38_libHD277E9",
    "ot1_day4_rep1",
    "ot1_day4_rep2",
    "ot1_day7_rep1",
    "ot1_day10_rep1",
)
assert EXTERNAL_SECTIONS == ("lung_994", "ot1_day7_rep2")
assert FINAL_SECTIONS == ("lung_993",)
assert RESERVE_SECTIONS == ("ot1_day10_rep2",)
assert QUARANTINE_SECTIONS == ("lung_subq",)
assert (GUIDE_MIN_RAW_UMI, TARGET_MIN_SCREEN_BINS) == (2, 100)
assert (TARGET_MIN_SECTION_BINS, NTC_MIN_SECTION_BINS) == (40, 50)
assert DataRole.RESERVE.value == "reserve"
assert AssignmentStatus.ACCEPTED_SGCTRL.value == "accepted_sgctrl"
assert StageDecision.DATA_NOT_READY.value == "data_not_ready"
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```text
rtk uv run python repro_runs/2026-07-21-spacseq-elrr/tests/test_schema.py
```

Expected: `ModuleNotFoundError: No module named 'elrr'`.

- [ ] **Step 3: Implement the minimal frozen types and constants**

```python
# src/elrr/schema.py
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class DataRole(str, Enum):
    DEVELOPMENT = "development"
    EXTERNAL = "external"
    FINAL = "final"
    RESERVE = "reserve"
    QUARANTINE = "quarantine"


class AssignmentStatus(str, Enum):
    ACCEPTED_TARGET = "accepted_target"
    ACCEPTED_NON_TARGETING = "accepted_non_targeting"
    ACCEPTED_INTERGENIC = "accepted_intergenic"
    ACCEPTED_SGCTRL = "accepted_sgctrl"
    UNASSIGNED_LOW_SIGNAL = "unassigned_low_signal"
    AMBIGUOUS_TIE = "ambiguous_tie"
    CONFLICTING_MULTI_TARGET = "conflicting_multi_target"


class StageDecision(str, Enum):
    DATA_NOT_READY = "data_not_ready"
    ADVANCE_TO_EXTERNAL_GATE = "advance_to_external_gate"
    DO_NOT_UNSEAL_LUNG_993 = "do_not_unseal_lung_993"
    FINAL_SUCCESS = "final_success"
    FINAL_GATE_FAILED = "final_gate_failed"
    EXTERNAL_ABORTED_NO_RESULT = "external_aborted_no_result"
    FINAL_ABORTED_NO_RESULT = "final_aborted_no_result"


@dataclass(frozen=True)
class SectionSpec:
    section_id: str
    screen_id: str
    biological_specimen_id: str
    timepoint: str
    platform: str
    capture_mode: str
    role: DataRole
    rna_path: Path
    guide_path: Path
    spatial_source: str
```

```python
# src/elrr/config.py
GUIDE_MIN_RAW_UMI = 2
TARGET_MIN_SCREEN_BINS = 100
TARGET_MIN_SECTION_BINS = 40
NTC_MIN_SECTION_BINS = 50
COMMON_GENE_LIMIT = 2000
COMMON_GENE_MINIMUM = 500
STUDY_SEED = 20260721

DEVELOPMENT_SECTIONS = (
    "lung_992",
    "mc38_libHD274E6",
    "mc38_libHD275E7",
    "mc38_libHD276E8",
    "mc38_libHD277E9",
    "ot1_day4_rep1",
    "ot1_day4_rep2",
    "ot1_day7_rep1",
    "ot1_day10_rep1",
)
EXTERNAL_SECTIONS = ("lung_994", "ot1_day7_rep2")
FINAL_SECTIONS = ("lung_993",)
RESERVE_SECTIONS = ("ot1_day10_rep2",)
QUARANTINE_SECTIONS = ("lung_subq",)
```

- [ ] **Step 4: Run the focused test and verify GREEN**

Expected: the script exits 0 and prints `ok`.

- [ ] **Step 5: Commit the scaffold**

```text
rtk git add repro_runs/2026-07-21-spacseq-elrr/src/elrr repro_runs/2026-07-21-spacseq-elrr/tests/test_schema.py
rtk git commit -m "feat: freeze ELRR data roles and schemas"
```

### Task 2: Frozen section manifest and library-manifest validation

**Files:**
- Create: `repro_runs/2026-07-21-spacseq-elrr/manifests/section_manifest.csv`
- Create: `repro_runs/2026-07-21-spacseq-elrr/src/elrr/manifest.py`
- Create: `repro_runs/2026-07-21-spacseq-elrr/tests/test_manifest.py`
- Input required at execution: `repro_runs/2026-07-21-spacseq-elrr/manifests/library_manifest.csv`

- [ ] **Step 1: Create the exact section manifest**

Use these 14 rows; do not substitute processed `Lung_Metastasis_M00*.h5` files for the raw Space Ranger matrix paths.

```csv
section_id,screen_id,biological_specimen_id,timepoint,platform,capture_mode,role,rna_path,guide_path,spatial_source
lung_992,lung_metastasis,M001,na,visium_hd,direct_capture,development,/mnt/172/wh/25-12/spatial/raw/SPAC-seq/Lung_Metastasis/992_raw_output/filtered_feature_bc_matrix.h5,/mnt/172/wh/25-12/spatial/raw/SPAC-seq/Lung_Metastasis/992_filtered_guide_bc_matrix.h5,/mnt/172/wh/25-12/spatial/raw/SPAC-seq/Lung_Metastasis/992_raw_output
lung_994,lung_metastasis,M002,na,visium_hd,direct_capture,external,/mnt/172/wh/25-12/spatial/raw/SPAC-seq/Lung_Metastasis/994_raw_output/filtered_feature_bc_matrix.h5,/mnt/172/wh/25-12/spatial/raw/SPAC-seq/Lung_Metastasis/994_filtered_guide_bc_matrix.h5,/mnt/172/wh/25-12/spatial/raw/SPAC-seq/Lung_Metastasis/994_raw_output
lung_993,lung_metastasis,M003,na,visium_hd,direct_capture,final,/mnt/172/wh/25-12/spatial/raw/SPAC-seq/Lung_Metastasis/993_raw_output/filtered_feature_bc_matrix.h5,/mnt/172/wh/25-12/spatial/raw/SPAC-seq/Lung_Metastasis/993_filtered_guide_bc_matrix.h5,/mnt/172/wh/25-12/spatial/raw/SPAC-seq/Lung_Metastasis/993_raw_output
lung_subq,lung_metastasis,subq,na,visium_hd,direct_capture,quarantine,/mnt/172/wh/25-12/spatial/raw/SPAC-seq/Lung_Metastasis/subq_raw_output/filtered_feature_bc_matrix.h5,/mnt/172/wh/25-12/spatial/raw/SPAC-seq/Lung_Metastasis/subq_filtered_guide_bc_matrix.h5,/mnt/172/wh/25-12/spatial/raw/SPAC-seq/Lung_Metastasis/subq_raw_output
mc38_libHD274E6,mc38_serial_tumor,MC38_serial_tumor_1,na,visium_hd,direct_capture,development,/mnt/172/wh/25-12/spatial/raw/SPAC-seq/MC38_tumor/Transcriptome/libHD274E6_filtered_gene_bc_matrix.h5,/mnt/172/wh/25-12/spatial/raw/SPAC-seq/MC38_tumor/Perturbation/libHD274E6_filtered_guide_bc_matrix.h5,/mnt/172/wh/25-12/spatial/raw/SPAC-seq/MC38_tumor/libHD274E6_raw_output.zip
mc38_libHD275E7,mc38_serial_tumor,MC38_serial_tumor_1,na,visium_hd,direct_capture,development,/mnt/172/wh/25-12/spatial/raw/SPAC-seq/MC38_tumor/Transcriptome/libHD275E7_filtered_gene_bc_matrix.h5,/mnt/172/wh/25-12/spatial/raw/SPAC-seq/MC38_tumor/Perturbation/libHD275E7_filtered_guide_bc_matrix.h5,/mnt/172/wh/25-12/spatial/raw/SPAC-seq/MC38_tumor/libHD275E7_raw_output.zip
mc38_libHD276E8,mc38_serial_tumor,MC38_serial_tumor_1,na,visium_hd,direct_capture,development,/mnt/172/wh/25-12/spatial/raw/SPAC-seq/MC38_tumor/Transcriptome/libHD276E8_filtered_gene_bc_matrix.h5,/mnt/172/wh/25-12/spatial/raw/SPAC-seq/MC38_tumor/Perturbation/libHD276E8_filtered_guide_bc_matrix.h5,/mnt/172/wh/25-12/spatial/raw/SPAC-seq/MC38_tumor/libHD276E8_raw_output.zip
mc38_libHD277E9,mc38_serial_tumor,MC38_serial_tumor_1,na,visium_hd,direct_capture,development,/mnt/172/wh/25-12/spatial/raw/SPAC-seq/MC38_tumor/Transcriptome/libHD277E9_filtered_gene_bc_matrix.h5,/mnt/172/wh/25-12/spatial/raw/SPAC-seq/MC38_tumor/Perturbation/libHD277E9_filtered_guide_bc_matrix.h5,/mnt/172/wh/25-12/spatial/raw/SPAC-seq/MC38_tumor/libHD277E9_raw_output.zip
ot1_day4_rep1,ot1_day,OT1_D4_R1,day4,stereo,mrna_embedding,development,/mnt/172/wh/25-12/spatial/raw/SPAC-seq/SPACseq_OT1_day/Day4_rep1.h5,/mnt/172/wh/25-12/spatial/raw/SPAC-seq/SPACseq_OT1_day/Day4_rep1.guide.h5,embedded_obs
ot1_day4_rep2,ot1_day,OT1_D4_R2,day4,stereo,mrna_embedding,development,/mnt/172/wh/25-12/spatial/raw/SPAC-seq/SPACseq_OT1_day/Day4_rep2.h5,/mnt/172/wh/25-12/spatial/raw/SPAC-seq/SPACseq_OT1_day/Day4_rep2.guide.h5,embedded_obs
ot1_day7_rep1,ot1_day,OT1_D7_R1,day7,stereo,mrna_embedding,development,/mnt/172/wh/25-12/spatial/raw/SPAC-seq/SPACseq_OT1_day/Day7_rep1.h5,/mnt/172/wh/25-12/spatial/raw/SPAC-seq/SPACseq_OT1_day/Day7_rep1.guide.h5,embedded_obs
ot1_day7_rep2,ot1_day,OT1_D7_R2,day7,stereo,mrna_embedding,external,/mnt/172/wh/25-12/spatial/raw/SPAC-seq/SPACseq_OT1_day/Day7_rep2.h5,/mnt/172/wh/25-12/spatial/raw/SPAC-seq/SPACseq_OT1_day/Day7_rep2.guide.h5,embedded_obs
ot1_day10_rep1,ot1_day,OT1_D10_R1,day10,stereo,mrna_embedding,development,/mnt/172/wh/25-12/spatial/raw/SPAC-seq/SPACseq_OT1_day/Day10_rep1.h5,/mnt/172/wh/25-12/spatial/raw/SPAC-seq/SPACseq_OT1_day/Day10_rep1.guide.h5,embedded_obs
ot1_day10_rep2,ot1_day,OT1_D10_R2,day10,stereo,mrna_embedding,reserve,/mnt/172/wh/25-12/spatial/raw/SPAC-seq/SPACseq_OT1_day/Day10_rep2.h5,/mnt/172/wh/25-12/spatial/raw/SPAC-seq/SPACseq_OT1_day/Day10_rep2.guide.h5,embedded_obs
```

- [ ] **Step 2: Write failing manifest tests**

```python
from pathlib import Path
import pandas as pd
import pytest

from elrr.manifest import ManifestError, load_library_manifest, load_section_manifest

sections = load_section_manifest(Path("repro_runs/2026-07-21-spacseq-elrr/manifests/section_manifest.csv"))
assert sections["section_id"].is_unique
assert sections.loc[sections.section_id.eq("lung_993"), "role"].item() == "final"
assert sections.loc[sections.section_id.eq("ot1_day10_rep2"), "role"].item() == "reserve"
assert sections.loc[sections.section_id.str.startswith("mc38_"), "biological_specimen_id"].nunique() == 1

bad = pd.DataFrame(
    {"guide_id": ["sgA", "sgA"], "target_gene": ["A", "B"], "control_type": ["target", "target"]}
)
with pytest.raises(ManifestError):
    load_library_manifest(bad)
```

Require library columns exactly:

```text
library_reference_id, guide_id, target_gene, control_type, source_reference, source_sha256
```

Accepted `control_type` values are `target`, `non_targeting`, `intergenic`, and `sgctrl`. Missing files, blank target mappings, duplicate guide mappings, or a guide mapped to two identities must return `manifest_incomplete`; no underscore-based inference is allowed.

- [ ] **Step 3: Run RED**

Expected: import failure for `elrr.manifest`.

- [ ] **Step 4: Implement manifest parsing and fail-closed validation**

```python
REQUIRED_LIBRARY_COLUMNS = (
    "library_reference_id",
    "guide_id",
    "target_gene",
    "control_type",
    "source_reference",
    "source_sha256",
)
CONTROL_TYPES = {"target", "non_targeting", "intergenic", "sgctrl"}

def load_library_manifest(source):
    frame = source.copy() if hasattr(source, "columns") else pd.read_csv(source, dtype=str)
    if tuple(frame.columns) != REQUIRED_LIBRARY_COLUMNS:
        raise ManifestError("library manifest columns do not match frozen schema")
    if frame[list(REQUIRED_LIBRARY_COLUMNS)].isna().any().any():
        raise ManifestError("library manifest contains blank required values")
    if frame["guide_id"].duplicated().any():
        raise ManifestError("guide_id must map exactly once")
    if not set(frame["control_type"]).issubset(CONTROL_TYPES):
        raise ManifestError("unknown control_type")
    return frame
```

If the actual library reference/Table S1 is unavailable, do not create a guessed `library_manifest.csv`; the readiness runner must emit `manifest_incomplete` and stop before assignment.

- [ ] **Step 5: Run GREEN and commit**

```text
rtk uv run python repro_runs/2026-07-21-spacseq-elrr/tests/test_manifest.py
rtk git add repro_runs/2026-07-21-spacseq-elrr/manifests/section_manifest.csv repro_runs/2026-07-21-spacseq-elrr/src/elrr/manifest.py repro_runs/2026-07-21-spacseq-elrr/tests/test_manifest.py
rtk git commit -m "feat: add frozen SPAC-seq manifests"
```

### Task 3: Role-aware access guard and seal activation

**Files:**
- Create: `repro_runs/2026-07-21-spacseq-elrr/src/elrr/access.py`
- Create: `repro_runs/2026-07-21-spacseq-elrr/tests/test_access.py`

- [ ] **Step 1: Write failing injected-loader tests**

```python
from elrr.access import AccessDenied, AccessKind, guarded_load
from elrr.schema import DataRole

calls = []
def loader(path):
    calls.append(path)
    return object()

guarded_load("dev.h5", DataRole.DEVELOPMENT, AccessKind.VALUE, "development", loader)
assert calls == ["dev.h5"]

for role in (DataRole.EXTERNAL, DataRole.FINAL, DataRole.RESERVE, DataRole.QUARANTINE):
    calls.clear()
    try:
        guarded_load("sealed.h5", role, AccessKind.VALUE, "development", loader)
    except AccessDenied:
        pass
    else:
        raise AssertionError("sealed role was not rejected")
    assert calls == []
```

Also test that checksum/stat access is allowed for all roles, value access is denied before the injected loader runs, and `activate_seal` refuses to write unless the design commit is `1535ab1b645ceec9644730d961507e45dc8b41cf`, all source checksums exist, the pre-freeze log hash is present, and every access-policy test is true.

- [ ] **Step 2: Run RED**

Expected: import failure for `elrr.access`.

- [ ] **Step 3: Implement the guard and activation artifact**

```python
class AccessKind(str, Enum):
    STAT = "stat"
    CHECKSUM = "checksum"
    SCHEMA = "schema"
    VALUE = "value"

VALUE_ROLES_BY_MODE = {
    "development": {DataRole.DEVELOPMENT},
    "external": {DataRole.DEVELOPMENT, DataRole.EXTERNAL},
    "final": {DataRole.DEVELOPMENT, DataRole.EXTERNAL, DataRole.FINAL},
}

def guarded_load(path, role, kind, mode, loader):
    if kind is AccessKind.VALUE and role not in VALUE_ROLES_BY_MODE.get(mode, set()):
        raise AccessDenied(f"{mode} cannot read {role.value} values")
    return loader(path)
```

`activate_seal` writes atomically via a temporary file and `Path.replace`, with keys:

```text
design_commit
activation_timestamp
data_role_mapping
source_checksums
pre_freeze_access_log_sha256
access_policy_sha256
access_test_artifact_sha256
```

The function must never accept a reserve value-access test as true.

- [ ] **Step 4: Run GREEN and commit**

```text
rtk uv run python repro_runs/2026-07-21-spacseq-elrr/tests/test_access.py
rtk git add repro_runs/2026-07-21-spacseq-elrr/src/elrr/access.py repro_runs/2026-07-21-spacseq-elrr/tests/test_access.py
rtk git commit -m "feat: guard ELRR data roles"
```

### Task 4: Exact-barcode Visium HD adapter

**Files:**
- Create: `repro_runs/2026-07-21-spacseq-elrr/src/elrr/io_visium.py`
- Create: `repro_runs/2026-07-21-spacseq-elrr/tests/test_io_visium.py`

- [ ] **Step 1: Write failing synthetic alignment tests**

```python
import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from elrr.io_visium import JoinMismatch, align_visium_modalities

rna = ad.AnnData(
    X=sparse.csr_matrix([[1, 0], [0, 2]]),
    obs=pd.DataFrame(index=["AA-1", "BB-1"]),
    var=pd.DataFrame(index=["G1", "G2"]),
)
guide = ad.AnnData(
    X=sparse.csr_matrix([[3], [4]]),
    obs=pd.DataFrame(index=["BB-1", "AA-1"]),
    var=pd.DataFrame(index=["sgA"]),
)
r, g, audit = align_visium_modalities(rna, guide, "lung_992")
assert r.obs_names.tolist() == ["AA-1", "BB-1"]
assert g.obs_names.tolist() == ["AA-1", "BB-1"]
assert g.X.toarray().ravel().tolist() == [4, 3]
assert audit["exact_barcode_match"] is True

with pytest.raises(JoinMismatch):
    align_visium_modalities(rna, guide[["BB-1"]].copy(), "lung_992")
```

Also require duplicate barcodes, absent tissue positions, nonunique `(section_id, barcode)`, and inconsistent pixel/array metadata to fail.

- [ ] **Step 2: Run RED**

Expected: import failure for `elrr.io_visium`.

- [ ] **Step 3: Implement the adapter**

Detect H5 structure from root groups rather than the `.h5` suffix:

```python
def detect_h5_kind(path):
    with h5py.File(path, "r") as handle:
        keys = set(handle.keys())
    if "matrix" in keys:
        return "10x"
    if {"X", "obs", "var"}.issubset(keys):
        return "h5ad"
    raise FeatureSchemaError(f"unknown H5 schema: {sorted(keys)}")
```

`read_visium_section(spec, library_manifest)` must:

1. read standard Space Ranger RNA (`matrix` root) with `sc.read_10x_h5(spec.rna_path, gex_only=True)`;
2. read guide H5AD (`X/obs/var` roots) with `ad.read_h5ad`, while also supporting a standard 10x guide matrix if schema detection finds `matrix`;
3. treat the entire separate guide file as guide namespace because its `feature_types` are known to be mislabeled;
4. load `tissue_positions.parquet` and `scalefactors_json.json` from a directory or `zipfile.ZipFile`;
5. assert unique and exactly equal barcode sets;
6. reorder guide and spatial rows to RNA order;
7. store raw sparse counts in `.layers["counts"]`;
8. attach nullable-union observation fields and source hashes;
9. return `(rna, guide, join_audit)` without dense conversion.

Use exact joins:

```python
missing = rna.obs_names.difference(guide.obs_names)
extra = guide.obs_names.difference(rna.obs_names)
if len(missing) or len(extra):
    raise JoinMismatch(f"missing_guide={len(missing)} guide_only={len(extra)}")
guide = guide[rna.obs_names].copy()
assert guide.obs_names.equals(rna.obs_names)
```

- [ ] **Step 4: Run the focused test and verify GREEN**

```text
rtk uv run python repro_runs/2026-07-21-spacseq-elrr/tests/test_io_visium.py
```

Expected: all synthetic barcode, spatial-metadata, sparse-count, and row-order tests pass. The development-only real adapter is exercised after `run_data_readiness.py` exists in Task 9.

- [ ] **Step 5: Commit**

```text
rtk git add repro_runs/2026-07-21-spacseq-elrr/src/elrr/io_visium.py repro_runs/2026-07-21-spacseq-elrr/tests/test_io_visium.py
rtk git commit -m "feat: add strict Visium HD alignment"
```

### Task 5: Integer-coordinate Stereo adapter

**Files:**
- Create: `repro_runs/2026-07-21-spacseq-elrr/src/elrr/io_stereo.py`
- Create: `repro_runs/2026-07-21-spacseq-elrr/tests/test_io_stereo.py`

- [ ] **Step 1: Write failing coordinate-contract tests**

```python
import pandas as pd
import pytest
from elrr.io_stereo import CoordinateContractError, align_stereo_observations

rna = pd.DataFrame({"section_id": ["s", "s"], "x_raw": [10, 20], "y_raw": [5, 5]})
guide = pd.DataFrame({"section_id": ["s", "s", "s"], "x_raw": [20, 10, 30], "y_raw": [5, 5, 5]})
aligned, audit = align_stereo_observations(rna, guide)
assert aligned[["x_raw", "y_raw"]].values.tolist() == [[10, 5], [20, 5]]
assert audit["guide_only_coordinates"] == 1

with pytest.raises(CoordinateContractError):
    align_stereo_observations(rna.assign(x_raw=[10.0, 20.5]), guide)
```

Require duplicate coordinate keys to fail, RNA coordinates missing from guide to fail, guide-only coordinates to be counted, and a row permutation on either side to produce identical aligned IDs.

- [ ] **Step 2: Run RED**

Expected: import failure for `elrr.io_stereo`.

- [ ] **Step 3: Implement the adapter**

`read_stereo_section` must use `anndata.read_h5ad` only after verifying the input is AnnData H5, preserve raw integer chip coordinates, and create:

```text
observation_id = section_id + ":" + str(x_raw) + ":" + str(y_raw)
```

It must never subtract per-file minima or use a `max(x)`-derived bin ID. Before joining:

```python
key = ["section_id", "x_raw", "y_raw"]
for frame in (rna_obs, guide_obs):
    frame[["x_raw", "y_raw"]] = frame[["x_raw", "y_raw"]].astype("int64")
    if frame.duplicated(key).any():
        raise CoordinateContractError("duplicate coordinate key")
```

Join guide to RNA order by the explicit key, keep guide-only coordinates only in the audit, and require `in_tissue`, physical coordinate scale, axis order, and tissue-mask provenance before a section can become primary-ready.

- [ ] **Step 4: Run the focused test and verify GREEN**

```text
rtk uv run python repro_runs/2026-07-21-spacseq-elrr/tests/test_io_stereo.py
```

Expected: all integer-coordinate, duplicate-key, missing-coordinate, guide-only, and row-order tests pass. The development-only real adapter is exercised after `run_data_readiness.py` exists in Task 9.

- [ ] **Step 5: Commit**

```text
rtk git add repro_runs/2026-07-21-spacseq-elrr/src/elrr/io_stereo.py repro_runs/2026-07-21-spacseq-elrr/tests/test_io_stereo.py
rtk git commit -m "feat: add strict Stereo coordinate alignment"
```

### Task 6: Conservative guide assignment state machine

**Files:**
- Create: `repro_runs/2026-07-21-spacseq-elrr/src/elrr/assignment.py`
- Create: `repro_runs/2026-07-21-spacseq-elrr/tests/test_assignment.py`

- [ ] **Step 1: Write failing assignment table tests**

```python
import pandas as pd
from elrr.assignment import assign_bins

library = pd.DataFrame(
    {
        "guide_id": ["sgA_1", "sgA_2", "sgB_1", "sgNTC", "sgIG", "sgCtrl"],
        "target_gene": ["A", "A", "B", "NTC", "IG", "CTRL"],
        "control_type": ["target", "target", "target", "non_targeting", "intergenic", "sgctrl"],
    }
)
raw = pd.DataFrame(
    {
        "bin_id": ["low", "a", "a", "conflict", "conflict", "tie", "tie", "ntc", "ctrl"],
        "guide_id": ["sgA_1", "sgA_1", "sgA_2", "sgA_1", "sgB_1", "sgA_1", "sgB_1", "sgNTC", "sgCtrl"],
        "raw_umi": [1, 2, 3, 9, 2, 4, 4, 2, 3],
    }
)
assigned, target_counts = assign_bins(raw, library)
status = assigned.set_index("bin_id")["assignment_status"].to_dict()
assert status["low"] == "unassigned_low_signal"
assert status["a"] == "accepted_target"
assert assigned.set_index("bin_id").loc["a", "top_count"] == 5
assert status["conflict"] == "conflicting_multi_target"
assert status["tie"] == "ambiguous_tie"
assert status["ntc"] == "accepted_non_targeting"
assert status["ctrl"] == "accepted_sgctrl"
```

Add explicit sgCtrl+target, sgCtrl+NTC, target+NTC, target+intergenic, raw row order, and sparse guide matrix tests. Require raw guide counts to remain unchanged after assignment.

- [ ] **Step 2: Run RED**

Expected: import failure for `elrr.assignment`.

- [ ] **Step 3: Implement assignment without argmax rescue**

Implementation order:

```python
eligible = raw.loc[raw["raw_umi"].ge(2)].merge(
    library[["guide_id", "target_gene", "control_type"]],
    on="guide_id",
    how="left",
    validate="many_to_one",
)
if eligible[["target_gene", "control_type"]].isna().any().any():
    raise ManifestError("unmapped guide_id")

identity_counts = (
    eligible.groupby(["bin_id", "target_gene", "control_type"], sort=True)["raw_umi"]
    .sum()
    .rename("target_sum")
    .reset_index()
)
```

Convert sparse guide counts without dense materialization:

```python
def guide_matrix_to_long(guide, section_id):
    counts = guide.layers["counts"].tocoo()
    return pd.DataFrame({
        "section_id": section_id,
        "bin_id": guide.obs_names.to_numpy()[counts.row],
        "guide_id": guide.var_names.to_numpy()[counts.col],
        "raw_umi": counts.data.astype(np.int64),
    })
```

For each bin: zero eligible identities gives low signal; one gives the matching accepted status; multiple identities with equal maximum target sums gives `ambiguous_tie`; all other multiple identities give `conflicting_multi_target`. Store `top_guide_count` separately from `top_count`. Do not implement DBSCAN, KDE, top-ratio rescue, or underscore parsing.

- [ ] **Step 4: Run GREEN and commit**

```text
rtk uv run python repro_runs/2026-07-21-spacseq-elrr/tests/test_assignment.py
rtk git add repro_runs/2026-07-21-spacseq-elrr/src/elrr/assignment.py repro_runs/2026-07-21-spacseq-elrr/tests/test_assignment.py
rtk git commit -m "feat: add conservative guide assignment"
```

### Task 7: NTC-only gene panel and section transform

**Files:**
- Create: `repro_runs/2026-07-21-spacseq-elrr/src/elrr/preprocessing.py`
- Create: `repro_runs/2026-07-21-spacseq-elrr/tests/test_preprocessing.py`

- [ ] **Step 1: Write failing preprocessing tests**

Create three synthetic development specimens with known NTC counts. Require:

```python
state = fit_gene_panel(sections, limit=2000, minimum=2)
assert state.genes == ("g2", "g1")
assert state.ddof == 0
assert state.selection_source == "development_direct_ntc_only"

section = sections[0]
z = transform_section(section.rna, section.assignments, state)
assert z.shape[1] == len(state.genes)
assert np.all(np.isfinite(z))
```

Test stable `feature_id` tie-breaking, at least 10 nonzero NTC bins per platform, exclusion annotations, `log1p(count/sum*10000)`, specimen-balanced variance, section NTC mean/std with `ddof=0`, frozen global sigma fallback when section sigma `<1e-6`, and `data_not_ready` when fewer than 500 real common genes remain. Mutating development target expression must not change the panel or scales.

- [ ] **Step 2: Run RED**

Expected: import failure for `elrr.preprocessing`.

- [ ] **Step 3: Implement exact fitting and transform APIs**

```python
@dataclass(frozen=True)
class GenePanelState:
    genes: Sequence[str]
    global_sigma: np.ndarray
    ddof: int
    selection_source: str
    source_hashes: dict[str, str]

def normalize_log1p(counts):
    totals = np.asarray(counts.sum(axis=1)).ravel()
    if np.any(totals <= 0):
        raise PreprocessingError("zero-library observation")
    scaled = counts.multiply(10000.0 / totals[:, None]).tocsr(copy=True)
    scaled.data = np.log1p(scaled.data)
    return scaled
```

Compute each specimen NTC variance with `ddof=0`, average variances equally over specimens, sort by `(-variance, feature_id)`, cap at 2,000, and freeze order. Use only `accepted_non_targeting`; intergenic and sgCtrl never enter panel fitting or section scaling.

- [ ] **Step 4: Run GREEN and commit**

```text
rtk uv run python repro_runs/2026-07-21-spacseq-elrr/tests/test_preprocessing.py
rtk git add repro_runs/2026-07-21-spacseq-elrr/src/elrr/preprocessing.py repro_runs/2026-07-21-spacseq-elrr/tests/test_preprocessing.py
rtk git commit -m "feat: add NTC-only expression transform"
```

### Task 8: Physical context, NTC anchors, W splits, and C directions

**Files:**
- Create: `repro_runs/2026-07-21-spacseq-elrr/src/elrr/context.py`
- Create: `repro_runs/2026-07-21-spacseq-elrr/src/elrr/episodes.py`
- Create: `repro_runs/2026-07-21-spacseq-elrr/tests/test_context.py`
- Create: `repro_runs/2026-07-21-spacseq-elrr/tests/test_episodes.py`

- [ ] **Step 1: Write failing context/anchor tests**

```python
context, state = build_context(obs, ntc_mask)
assert context.shape == (len(obs), 4)
assert np.all((context[:, :2] >= 0) & (context[:, :2] <= 1))
assert state.k == 16
assert state.variance_ddof == 0

anchor, audit = build_ntc_anchor(context, expression, ntc_mask)
assert anchor.shape == expression.shape
assert np.allclose(audit["weight_sum"], 1.0)
```

Require physical bounding-box normalization, Euclidean mask distance, leave-one-out NTC KDE, zero density feature when variance `<1e-12`, stable `(distance, observation_id)` tie-breaking, exactly 16 NTC anchors, row-permutation invariance, and failure on missing tissue-mask/physical-unit provenance.

- [ ] **Step 2: Write failing W/C episode tests**

Require deterministic 70/30 W splits by 2x2 quadrants, `sparse`-stratum merging, spatial block closest to 30% only within 20–40%, no expression access, and minimums 100 screen bins/40 section-target/50 NTC.

Require exact C directions:

```python
assert external_directions == (
    ("lung_992", "lung_994"),
    ("ot1_day7_rep1", "ot1_day7_rep2"),
)
assert final_directions == (("lung_992", "lung_993"),)
```

MC38 serial sections must never count as independent C specimens.

- [ ] **Step 3: Run both tests and verify RED**

Expected: import failures for `elrr.context` and `elrr.episodes`.

- [ ] **Step 4: Implement the exact contracts**

Implement the split primitives directly from observation IDs and context:

```python
class EpisodeNotReady(RuntimeError):
    pass

def stable_hash(*parts):
    payload = chr(124).join(str(part) for part in parts).encode()
    return hashlib.sha256(payload).hexdigest()

def stable_order(ids, seed, unit_id):
    return tuple(sorted(ids, key=lambda value: stable_hash(seed, unit_id, value)))

def quadrant_labels(context):
    return ((context[:, 0] >= 0.5).astype(int) * 2 + (context[:, 1] >= 0.5).astype(int))

def random_stratified_split(ids, context, seed, unit_id):
    labels = quadrant_labels(context)
    groups = []
    sparse = []
    for label in range(4):
        members = [value for value, group in zip(ids, labels) if group == label]
        if len(members) < 4:
            sparse.extend(members)
        else:
            groups.append(members)
    if sparse:
        groups.append(sparse)
    support, query = [], []
    for members in groups:
        ordered = stable_order(members, seed, unit_id)
        n_query = max(1, int(np.ceil(0.30 * len(ordered))))
        query.extend(ordered[:n_query])
        support.extend(ordered[n_query:])
    return tuple(sorted(support)), tuple(sorted(query))

def spatial_block_split(ids, context, seed, unit_id):
    labels = quadrant_labels(context)
    candidates = []
    for label in range(4):
        query = tuple(value for value, group in zip(ids, labels) if group == label)
        fraction = len(query) / len(ids)
        if 0.20 <= fraction <= 0.40:
            candidates.append((abs(fraction - 0.30), stable_hash(seed, unit_id, label), label, query))
    if not candidates:
        raise EpisodeNotReady("no spatial quadrant in the frozen 20-40% range")
    _, _, selected, query = min(candidates)
    support = tuple(value for value, group in zip(ids, labels) if group != selected)
    return tuple(sorted(support)), tuple(sorted(query))
```

Define immutable episode records in `episodes.py`:

```python
@dataclass(frozen=True)
class WUnit:
    unit_id: str
    section_id: str
    target_gene: str
    view: str
    support_ids: tuple
    query_ids: tuple

@dataclass(frozen=True)
class CUnit:
    unit_id: str
    support_section_id: str
    query_section_id: str
    target_gene: str
```

`build_w_units(assignments_by_section, contexts_by_section, section_manifest, seed)` iterates accepted development `(section_id,target_gene)` groups with at least 40 bins, calls both split functions, and stores only immutable observation-ID records. `build_c_units(assignments_by_section, section_manifest)` uses this exact table:

```python
C_DIRECTIONS = {
    "external": (
        ("lung_992", "lung_994"),
        ("ot1_day7_rep1", "ot1_day7_rep2"),
    ),
    "final": (("lung_992", "lung_993"),),
}
```

Development C directions are every timepoint/platform/capture-matched ordered pair of different biological specimens; MC38 serial sections are excluded because their specimen IDs are identical. Both builders accept assignments, coordinates, manifest roles, and observation IDs; neither accepts target expression arrays. Context D4 transforms operate around `(0.5,0.5)` on x/y only.

- [ ] **Step 5: Run GREEN and commit**

```text
rtk uv run python repro_runs/2026-07-21-spacseq-elrr/tests/test_context.py
rtk uv run python repro_runs/2026-07-21-spacseq-elrr/tests/test_episodes.py
rtk git add repro_runs/2026-07-21-spacseq-elrr/src/elrr/context.py repro_runs/2026-07-21-spacseq-elrr/src/elrr/episodes.py repro_runs/2026-07-21-spacseq-elrr/tests/test_context.py repro_runs/2026-07-21-spacseq-elrr/tests/test_episodes.py
rtk git commit -m "feat: build ELRR contexts and episodes"
```

### Task 9: Development-only readiness runner and seal artifact

**Files:**
- Create: `repro_runs/2026-07-21-spacseq-elrr/src/elrr/readiness.py`
- Create: `repro_runs/2026-07-21-spacseq-elrr/run_data_readiness.py`
- Create: `repro_runs/2026-07-21-spacseq-elrr/tests/test_readiness_runner.py`
- Create after run: `repro_runs/2026-07-21-spacseq-elrr/results/data_readiness/data_readiness_summary.json`
- Create after run: `repro_runs/2026-07-21-spacseq-elrr/results/data_readiness/join_audit.csv`
- Create after run: `repro_runs/2026-07-21-spacseq-elrr/results/data_readiness/assignment_audit.csv`
- Create after run: `repro_runs/2026-07-21-spacseq-elrr/results/data_readiness/episode_readiness.csv`
- Create after run: `repro_runs/2026-07-21-spacseq-elrr/results/data_readiness/development_splits.csv`
- Create after run: `repro_runs/2026-07-21-spacseq-elrr/results/data_readiness/pre_freeze_access_log.json`
- Create after run: `repro_runs/2026-07-21-spacseq-elrr/results/data_readiness/seal_activation.json`
- Create after run: `repro_runs/2026-07-21-spacseq-elrr/results/data_readiness/gene_panel.json`
- Create after run: `repro_runs/2026-07-21-spacseq-elrr/results/data_readiness/section_transforms.npz`
- Create after run: `repro_runs/2026-07-21-spacseq-elrr/results/data_readiness/processed/development/{section_id}.rna.h5ad`
- Create after run: `repro_runs/2026-07-21-spacseq-elrr/results/data_readiness/processed/development/{section_id}.guide.h5ad`

- [ ] **Step 1: Write a failing injected-runner test**

Inject manifest, platform loaders, and access callbacks. Require the runner to:

1. call value loaders only for development section IDs;
2. return `manifest_incomplete` before raw assignment if the library manifest is missing;
3. write all audits atomically;
4. produce `data_ready` only with at least 3 specimen groups, 8 W episodes, 3 C targets, and 500 genes;
5. write `seal_activation.json` only after all access tests pass;
6. keep Lung 994, Lung 993, OT1 Day7 rep2, and OT1 Day10 rep2 value-loader call counts at zero.

- [ ] **Step 2: Run RED**

Expected: import failure for `elrr.readiness`.

- [ ] **Step 3: Implement readiness orchestration**

Define the new Task 9 types and helpers with exact signatures:

```python
@dataclass(frozen=True)
class PreparedSection:
    section_id: str
    specimen_id: str
    expression: np.ndarray
    context: np.ndarray
    anchor: np.ndarray
    observation_ids: Sequence[str]
    assignments: pd.DataFrame

def evaluate_data_readiness(prepared, w_units, c_units, gene_count):
    reasons = []
    if len({x.specimen_id for x in prepared.values()}) < 3:
        reasons.append("insufficient_specimen_groups")
    if len(w_units) < 8:
        reasons.append("insufficient_w_units")
    if len({x.target_gene for x in c_units}) < 3:
        reasons.append("insufficient_c_targets")
    if gene_count < 500:
        reasons.append("insufficient_common_genes")
    return ("data_ready" if not reasons else "data_not_ready", tuple(sorted(reasons)))
```

`run_development_readiness` then performs the already-defined calls in this fixed order:

```python
development = load_section_manifest(section_manifest).query("role == 'development'")
loaded = {}
for spec in iter_section_specs(development):
    reader = read_visium_section if spec.platform == "visium_hd" else read_stereo_section
    rna, guide, join_audit = reader(spec, library_manifest)
    raw_long = guide_matrix_to_long(guide, spec.section_id)
    assignments, target_counts = assign_bins(raw_long, library_manifest)
    loaded[spec.section_id] = (spec, rna, guide, assignments, join_audit)

panel = fit_gene_panel(tuple(loaded.values()), limit=2000, minimum=500)
prepared = {}
for section_id, (spec, rna, guide, assignments, join_audit) in loaded.items():
    expression = transform_section(rna, assignments, panel)
    assignment_by_id = assignments.set_index("bin_id").reindex(rna.obs_names)
    ntc_mask = assignment_by_id["assignment_status"].eq("accepted_non_targeting").to_numpy()
    context, context_state = build_context(rna.obs, ntc_mask)
    anchor, anchor_audit = build_ntc_anchor(context, expression, ntc_mask)
    prepared[section_id] = PreparedSection(
        section_id=section_id,
        specimen_id=spec.biological_specimen_id,
        expression=expression,
        context=context,
        anchor=anchor,
        observation_ids=tuple(rna.obs_names),
        assignments=assignments,
    )

assignments_by_section = {
    section_id: item.assignments for section_id, item in prepared.items()
}
contexts_by_section = {
    section_id: item.context for section_id, item in prepared.items()
}
w_units = build_w_units(assignments_by_section, contexts_by_section, development, seed=20260721)
c_units = build_c_units(assignments_by_section, development)
status, reason_codes = evaluate_data_readiness(prepared, w_units, c_units, len(panel.genes))
```

Task 9 imports `guide_matrix_to_long` from Task 6 and defines `iter_section_specs` plus the atomic artifact writers used above; each is covered by the injected runner test. No exception may become an empty/zero dataset. Map every failure to a typed reason and preserve raw counts in audit rows.

- [ ] **Step 4: Run GREEN and commit**

```text
rtk uv run python repro_runs/2026-07-21-spacseq-elrr/tests/test_readiness_runner.py
rtk git add repro_runs/2026-07-21-spacseq-elrr/src/elrr/readiness.py repro_runs/2026-07-21-spacseq-elrr/run_data_readiness.py repro_runs/2026-07-21-spacseq-elrr/tests/test_readiness_runner.py
rtk git commit -m "feat: add ELRR data readiness runner"
```

### Task 10: Data artifact verifier, test aggregator, and execution checkpoint

**Files:**
- Create: `repro_runs/2026-07-21-spacseq-elrr/verify_data_artifacts.py`
- Create: `repro_runs/2026-07-21-spacseq-elrr/tests/test_data_artifact_verifier.py`
- Create: `repro_runs/2026-07-21-spacseq-elrr/tests/run_data_tests.py`
- Create: `repro_runs/2026-07-21-spacseq-elrr/README.md`

- [ ] **Step 1: Write failing artifact-verifier tests**

Require verifier failure for:

- absent manifest hash;
- a development summary naming external/final/reserve sections as value-read;
- duplicated observation IDs;
- missing guide-level raw evidence;
- episode counts inconsistent with assignment audit;
- a `data_ready` summary with fewer than 500 genes or three C targets;
- a seal activation hash not tied to design commit `1535ab1b645ceec9644730d961507e45dc8b41cf`.

- [ ] **Step 2: Run RED**

Expected: import failure for `verify_data_artifacts`.

- [ ] **Step 3: Implement verifier and fresh-process aggregator**

`run_data_tests.py` must run, in order:

```python
TEST_FILES = (
    "test_schema.py",
    "test_manifest.py",
    "test_access.py",
    "test_io_visium.py",
    "test_io_stereo.py",
    "test_assignment.py",
    "test_preprocessing.py",
    "test_context.py",
    "test_episodes.py",
    "test_readiness_runner.py",
    "test_data_artifact_verifier.py",
)
```

Use `subprocess.run([sys.executable, test], check=False)` and stop on the first nonzero return code. The verifier prints one compact JSON summary and exits nonzero on any contract failure.

- [ ] **Step 4: Run the full synthetic/injected suite**

```text
rtk uv run python repro_runs/2026-07-21-spacseq-elrr/tests/run_data_tests.py
```

Expected: `PASS 11 data test files`.

- [ ] **Step 5: Run development readiness only**

```text
rtk uv run python repro_runs/2026-07-21-spacseq-elrr/run_data_readiness.py
rtk uv run python repro_runs/2026-07-21-spacseq-elrr/verify_data_artifacts.py
```

Expected outcome is exactly one of:

```text
data_ready
data_not_ready with explicit immutable reason_codes
```

If `library_manifest.csv`, tissue-mask provenance, or physical units are unresolved, preserve `data_not_ready`; do not infer mappings or relax thresholds. Do not start Plan B unless the verifier confirms `data_ready`.

- [ ] **Step 6: Commit documentation and verifier**

```text
rtk git add repro_runs/2026-07-21-spacseq-elrr/README.md repro_runs/2026-07-21-spacseq-elrr/verify_data_artifacts.py repro_runs/2026-07-21-spacseq-elrr/tests/test_data_artifact_verifier.py repro_runs/2026-07-21-spacseq-elrr/tests/run_data_tests.py
rtk git commit -m "docs: document ELRR data readiness gate"
```

## Plan A completion evidence

Before reporting Plan A complete, run:

```text
rtk uv run python repro_runs/2026-07-21-spacseq-elrr/tests/run_data_tests.py
rtk uv run python repro_runs/2026-07-21-spacseq-elrr/verify_data_artifacts.py
rtk git status --short repro_runs/2026-07-21-spacseq-elrr
```

The handoff to Plan B must include:

- `data_readiness_summary.json`;
- frozen gene list and section transforms;
- exact development W/C unit IDs;
- `seal_activation.json`;
- source/config/code hashes;
- a statement that external/final/reserve value-read counts remain zero.

## Spec coverage for Plan A

| Design sections | Implementation tasks |
| --- | --- |
| §5 data roles and prospective seal | Tasks 1–3, 9–10 |
| §6 author-code reuse boundary | Tasks 4–6 |
| §7 manifest, nullable schema, Visium/Stereo joins | Tasks 1–5 |
| §8 conservative assignment and support gates | Tasks 2, 6, 8–9 |
| §9 NTC-only expression preprocessing | Task 7 |
| §10 W/C episodes and leakage-safe splits | Task 8 |
| §11 physical context and NTC anchors | Task 8 |
| §17 access, hashes, and seal activation | Tasks 3, 9–10 |
| §18 typed fail-closed statuses | Tasks 1–3, 6, 9 |
| §20 data contract tests | Tasks 1–10 |
| §21 data artifacts | Tasks 9–10 |

