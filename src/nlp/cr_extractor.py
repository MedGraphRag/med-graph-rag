from pathlib import Path

import pymupdf
import re


def extract_pages(pdf_path: str | Path) -> list[dict[str, int | str]]:
    """Extract text from a PDF, preserving the original page numbers.

    Args:
        pdf_path: Path to a clinical guideline PDF.

    Returns:
        A list of dictionaries containing the 1-based page number and text.
        Pages with no extractable text are preserved as empty strings.

    Raises:
        FileNotFoundError: If the PDF does not exist.
        ValueError: If the path is not a PDF or the document is empty.
        RuntimeError: If the PDF cannot be opened or read.
    """
    path = Path(pdf_path)

    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {path}")

    try:
        with pymupdf.open(path) as document:
            if document.is_encrypted:
                raise ValueError(f"Encrypted PDF is not supported: {path}")

            if len(document) == 0:
                raise ValueError(f"PDF contains no pages: {path}")

            pages = []

            for page_number, page in enumerate(document, start=1):
                text = page.get_text("text")
                pages.append(
                    {
                        "page": page_number,
                        "text": text.strip(),
                    }
                )

            return pages

    except ValueError:
        raise
    except Exception as error:
        raise RuntimeError(f"Failed to extract text from {path}") from error


SECTION_HEADING_PATTERN = re.compile(
    r"^\s*\d+(?:\.\d+)*\.?\s+[А-ЯЁA-Z]\S*.*$"
)


def extract_section_headings(text: str) -> list[str]:
    headings = []
    lines = text.splitlines()
    index = 0

    if "Оглавление" in text and "1. Краткая информация" not in text:
        return []

    while index < len(lines):
        line = " ".join(lines[index].split())

        if not line or not SECTION_HEADING_PATTERN.match(line):
            index += 1
            continue

        heading = line
        index += 1

        while index < len(lines):
            next_line = " ".join(lines[index].split())

            if not next_line:
                break

            if SECTION_HEADING_PATTERN.match(next_line):
                break

            if next_line[0].islower():
                heading += " " + next_line
                index += 1
            else:
                break

        headings.append(heading)

    return headings