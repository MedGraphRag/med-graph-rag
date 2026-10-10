"""
Download and normalize ICD-10 (МКБ-10) from the Russian Ministry of Health.

Source:
    https://github.com/ak4nv/mkb10

The repository states that the source data comes from
the Russian Ministry of Health.

Source CSV structure:

    ID
    REC_CODE
    MKB_CODE
    MKB_NAME
    ID_PARENT
    ADDL_CODE
    ACTUAL
    DATE

Important:
    We DO NOT calculate parent_code from MKB_CODE.

    The source already contains the exact hierarchy in ID_PARENT.

Example:

    ID=1   MKB_CODE=I       ID_PARENT=NULL
    ID=2   MKB_CODE=A00-A09 ID_PARENT=1
    ID=3   MKB_CODE=A00     ID_PARENT=2
    ID=4   MKB_CODE=A00.0   ID_PARENT=3

Result:

    I
    └── A00-A09
        └── A00
            └── A00.0

Output:

    data/icd10.json
"""

from __future__ import annotations

import csv
import json
import urllib.request
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

# ============================================================================
# Configuration
# ============================================================================

SOURCE_URL = (
    "https://raw.githubusercontent.com/"
    "ak4nv/mkb10/master/resources/"
    "1.2.643.5.1.13.13.11.1005_2.27.csv"
)

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")

RAW_FILE = RAW_DIR / "icd10.csv"
OUTPUT_FILE = OUTPUT_DIR / "icd10.json"

ICD10_VERSION = "2.27"
ICD10_LANGUAGE = "ru"


# ============================================================================
# Pydantic models
# ============================================================================


class ICD10Level(str, Enum):
    CLASS = "class"
    BLOCK = "block"
    CATEGORY = "category"
    SUBCATEGORY = "subcategory"


class ICD10Node(BaseModel):
    """Normalized ICD-10 node."""

    code: str = Field(
        description="Код по классификации МКБ-10",
    )

    name: str = Field(
        description="Официальное наименование",
    )

    parent_code: str | None = Field(
        default=None,
        description="Код родительского узла",
    )

    level: ICD10Level = Field(
        description="Уровень узла",
    )


class ICD10Dataset(BaseModel):
    """Complete normalized ICD-10 dataset."""

    version: str
    language: str
    source: str
    nodes: list[ICD10Node]


# ============================================================================
# Download
# ============================================================================


def download_icd10() -> Path:
    """Download original ICD-10 CSV."""

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Downloading ICD-10...")
    print(SOURCE_URL)

    request = urllib.request.Request(
        SOURCE_URL,
        headers={
            "User-Agent": "icd10-downloader/1.0",
        },
    )

    with urllib.request.urlopen(request) as response:
        content = response.read()

    RAW_FILE.write_bytes(content)

    print(f"Saved raw file: {RAW_FILE}")
    print(f"Size: {len(content):,} bytes")

    return RAW_FILE


# ============================================================================
# CSV
# ============================================================================


def read_icd10_csv(path: Path) -> list[dict[str, str]]:
    """
    Read source CSV.

    The source uses:
        ;
    as delimiter and UTF-8 encoding.
    """

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        reader = csv.DictReader(
            file,
            delimiter=";",
        )

        rows = []

        for row in reader:
            normalized = {
                key.strip(): (
                    value.strip()
                    if value is not None
                    else ""
                )
                for key, value in row.items()
            }

            rows.append(normalized)

    return rows


# ============================================================================
# Level detection
# ============================================================================


def detect_level(
    code: str,
    parent_code: str | None,
) -> ICD10Level:
    """
    Determine ICD-10 hierarchy level.

    The source hierarchy is:

        class
            ↓
        block
            ↓
        category
            ↓
        subcategory

    Examples:

        I       -> class
        A00-A09 -> block
        A00     -> category
        A00.0   -> subcategory

    We use the actual source code structure rather than
    trying to infer parent relationships.
    """

    code = code.strip()

    ROMAN_CLASSES = {
    "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
    "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX", "XXI", "XXII"
}   

    # ICD-10 class:
    #
    # Example:
    #   I
    #
    if code in ROMAN_CLASSES:
        return ICD10Level.CLASS

    # ICD-10 block:
    #
    # Examples:
    #   A00-A09
    #   A15-A19
    #   E00-E07
    #
    if "-" in code:
        return ICD10Level.BLOCK

    # ICD-10 subcategory:
    #
    # Examples:
    #   A00.0
    #   A00.1
    #   E11.9
    #
    if "." in code:
        return ICD10Level.SUBCATEGORY

    # ICD-10 category:
    #
    # Examples:
    #   A00
    #   E11
    #   I10
    #
    return ICD10Level.CATEGORY


# ============================================================================
# Conversion
# ============================================================================


def convert_rows(
    rows: list[dict[str, str]],
) -> list[ICD10Node]:
    """
    Convert source rows into normalized ICD10Node objects.

    Parent relationships are resolved through:

        ID_PARENT -> ID -> MKB_CODE

    This is important.

    We do NOT assume that:

        A00.0 -> A00

    is always enough to reconstruct the hierarchy.

    The source explicitly provides the parent ID,
    so we use the authoritative relationship from the source.
    """

    required_columns = {
        "ID",
        "MKB_CODE",
        "MKB_NAME",
        "ID_PARENT",
        "ACTUAL",
    }

    if not rows:
        raise RuntimeError(
            "ICD-10 CSV is empty."
        )

    missing = required_columns - set(rows[0])

    if missing:
        raise RuntimeError(
            "ICD-10 CSV has unexpected structure.\n"
            f"Missing columns: {sorted(missing)}\n"
            f"Available columns: {sorted(rows[0])}"
        )

    # ----------------------------------------------------------------------
    # First pass:
    #
    # ID -> MKB_CODE
    #
    # This allows us to resolve ID_PARENT later.
    # ----------------------------------------------------------------------

    id_to_code: dict[str, str] = {}

    for row in rows:
        node_id = row["ID"].strip()
        code = row["MKB_CODE"].strip()

        if not node_id or not code:
            continue

        id_to_code[node_id] = code

    # ----------------------------------------------------------------------
    # Second pass:
    #
    # Build Pydantic nodes.
    # ----------------------------------------------------------------------

    nodes: list[ICD10Node] = []

    seen_codes: set[str] = set()

    skipped_inactive = 0
    skipped_empty = 0

    for row in rows:
        code = row["MKB_CODE"].strip()
        name = row["MKB_NAME"].strip()

        # ------------------------------------------------------------------
        # Empty records
        # ------------------------------------------------------------------

        if not code or not name:
            skipped_empty += 1
            continue

        # ------------------------------------------------------------------
        # Only actual/current records.
        #
        # ACTUAL == 1 means the record is current.
        # ------------------------------------------------------------------

        actual = row["ACTUAL"].strip()

        if actual != "1":
            skipped_inactive += 1
            continue

        # ------------------------------------------------------------------
        # Avoid duplicate MKB codes.
        # ------------------------------------------------------------------

        if code in seen_codes:
            continue

        # ------------------------------------------------------------------
        # Resolve parent.
        #
        # Example:
        #
        # A00.0
        # ID = 4
        # ID_PARENT = 3
        #
        # 3 -> A00
        #
        # Therefore:
        #
        # parent_code = "A00"
        # ------------------------------------------------------------------

        parent_id = row["ID_PARENT"].strip()

        parent_code = (
            id_to_code.get(parent_id)
            if parent_id
            else None
        )

        # ------------------------------------------------------------------
        # Determine level from MKB_CODE.
        # ------------------------------------------------------------------

        level = detect_level(
            code=code,
            parent_code=parent_code,
        )

        node = ICD10Node(
            code=code,
            name=name,
            parent_code=parent_code,
            level=level,
        )

        nodes.append(node)

        seen_codes.add(code)

    print()
    print("Conversion result:")
    print(f"  nodes:             {len(nodes):,}")
    print(f"  skipped inactive:  {skipped_inactive:,}")
    print(f"  skipped empty:     {skipped_empty:,}")

    return nodes


# ============================================================================
# Validation
# ============================================================================


def validate_hierarchy(
    nodes: list[ICD10Node],
) -> None:
    """
    Validate the resulting hierarchy.

    Every parent_code must point to an existing node.
    """

    codes = {
        node.code
        for node in nodes
    }

    errors: list[str] = []

    for node in nodes:

        if node.parent_code is None:
            continue

        if node.parent_code not in codes:
            errors.append(
                f"{node.code} -> missing parent "
                f"{node.parent_code}"
            )

    if errors:
        print()
        print("Hierarchy errors:")

        for error in errors[:20]:
            print(f"  ERROR: {error}")

        if len(errors) > 20:
            print(
                f"  ... and {len(errors) - 20:,} more"
            )

        raise RuntimeError(
            f"Invalid ICD-10 hierarchy: "
            f"{len(errors)} missing parents."
        )

    print("Hierarchy validation: OK")


# ============================================================================
# Statistics
# ============================================================================


def print_statistics(
    nodes: list[ICD10Node],
) -> None:
    """Print useful information about the generated tree."""

    counts = {
        level: 0
        for level in ICD10Level
    }

    roots = 0

    for node in nodes:
        counts[node.level] += 1

        if node.parent_code is None:
            roots += 1

    print()
    print("ICD-10 statistics:")
    print(f"  classes:       {counts[ICD10Level.CLASS]:,}")
    print(f"  blocks:        {counts[ICD10Level.BLOCK]:,}")
    print(f"  categories:    {counts[ICD10Level.CATEGORY]:,}")
    print(f"  subcategories: {counts[ICD10Level.SUBCATEGORY]:,}")
    print(f"  root nodes:    {roots:,}")


# ============================================================================
# JSON
# ============================================================================


def save_json(
    nodes: list[ICD10Node],
) -> Path:
    """Save normalized ICD-10 dataset."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset = ICD10Dataset(
        version=ICD10_VERSION,
        language=ICD10_LANGUAGE,
        source="Минздрав РФ",
        nodes=nodes,
    )

    OUTPUT_FILE.write_text(
        json.dumps(
            dataset.model_dump(
                mode="json",
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(f"Saved: {OUTPUT_FILE}")

    return OUTPUT_FILE


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    """Download, normalize and validate ICD-10."""

    # 1. Download original CSV.
    raw_file = download_icd10()

    # 2. Read CSV.
    rows = read_icd10_csv(raw_file)

    print(f"Source rows: {len(rows):,}")

    # 3. Convert to Pydantic models.
    nodes = convert_rows(rows)

    # 4. Validate parent relationships.
    validate_hierarchy(nodes)

    # 5. Print statistics.
    print_statistics(nodes)

    # 6. Save normalized JSON.
    save_json(nodes)

    print()
    print("ICD-10 download completed successfully.")


if __name__ == "__main__":
    main()

