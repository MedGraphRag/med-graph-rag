
from pathlib import Path

import pymupdf
import re


def extract_pages(
    pdf_path: str | Path,
) -> list[dict[str, int | str | bool | dict]]:
    """Extract text from a PDF, preserving page numbers and table presence.

    Args:
        pdf_path: Path to a clinical guideline PDF.

    Returns:
        A list of dictionaries containing:
        - page: 1-based page number;
        - text: extracted page text;
        - formatted_text: formatted page content;
        - has_tables: whether tables were detected on the page.

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
                formatted_text = page.get_text("dict")

                pages.append(
                    {
                        "page": page_number,
                        "text": text.strip(),
                        "formatted_text": formatted_text,
                    }
                )

            return pages

    except ValueError:
        raise
    except Exception as error:
        raise RuntimeError(
            f"Failed to extract text from {path}"
        ) from error


SECTION_HEADING_PATTERN = re.compile(
    r"^\s*\d+(?:\.\d+)*\.?\s+[А-ЯЁA-Z]\S*.*$"
)


def is_valid_section_heading(heading: str) -> bool:
    if heading.startswith((
        "1. Диагноз ПГПТ устанавливается",
        "2. Применение визуализирующих методов исследования обязательно",
    )):
        return False

    if len(heading) > 180:
        return False

    if re.search(r"\bet\s+al\.|\bdoi\s*:", heading, re.IGNORECASE):
        return False

    if re.search(r"\b\d{4}\s*;\s*\d+", heading):
        return False

    if heading.endswith((".", ";", ",")):
        return False

    # Строки оглавления с точками-лидерами и номером страницы.
    if re.search(r"\.{3,}\s*\d+\s*$", heading):
        return False

    # Библиографические записи с фамилиями и инициалами.
    if re.search(r"\b[A-ZА-ЯЁ]\.\s*[A-ZА-ЯЁ]?\w*[, ]", heading):
        if re.search(
            r"\b(et al\.|doi:|19\d{2}|20\d{2})\b",
            heading,
            re.IGNORECASE,
        ):
            return False

    # Библиография: название публикации начинается после авторов.
    if re.match(
        r"^\d+\.\s+[A-ZА-ЯЁ][a-zа-яё]+(?:\s+[A-ZА-ЯЁ]\.)",
        heading,
    ):
        return False

    # Список авторов или членов рабочей группы.
    if re.search(
        r"\b(Research Group|Study Group|д\.м\.н\.|к\.м\.н\.|"
        r"академик РАН|директор ФГБУ)\b",
        heading,
        re.IGNORECASE,
    ):
        return False

        # Отсекаем обозначения уровня доказательности, например "5 С" или "2 В".
    if re.fullmatch(r"\d+\s+[А-ЯЁA-Z]", heading):
        return False

    return True


def extract_section_headings(formatted_text: dict) -> list[str]:
    lines = []

    for block in formatted_text.get("blocks", []):
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = " ".join(
                "".join(span["text"] for span in spans).split()
            )

            if not text:
                continue

            visible_spans = [
                span for span in spans if span["text"].strip()
            ]

            is_bold = bool(visible_spans) and all(
                span["flags"] & 16 for span in visible_spans
            )
            font_size = (
                max(span["size"] for span in visible_spans)
                if visible_spans
                else 0
            )

            lines.append((text, is_bold, font_size))

    headings = []
    index = 0

    while index < len(lines):
        text, is_bold, font_size = lines[index]

        if not is_bold:
            index += 1
            continue

        # Номер раздела может находиться на отдельной строке.
        if re.fullmatch(r"\d+(?:\.\d+)*\.?", text):
            parts = [text]
            next_index = index + 1

            while next_index < len(lines):
                next_text, next_is_bold, next_font_size = lines[next_index]

                if (
                    not next_is_bold
                    or abs(next_font_size - font_size) > 0.5
                ):
                    break

                # Не захватываем следующий самостоятельный заголовок.
                if (
                    next_index > index + 1
                    and SECTION_HEADING_PATTERN.match(next_text)
                ):
                    break

                parts.append(next_text)
                next_index += 1

            heading = " ".join(parts)
            heading = " ".join(heading.split())

            if (
                SECTION_HEADING_PATTERN.match(heading)
                and is_valid_section_heading(heading)
            ):
                headings.append(heading)

            index = next_index
            continue

        if not SECTION_HEADING_PATTERN.match(text):
            index += 1
            continue

        heading = text
        index += 1

        while index < len(lines):
            next_text, next_is_bold, next_font_size = lines[index]

            if (
                not next_is_bold
                or abs(next_font_size - font_size) > 0.5
                or SECTION_HEADING_PATTERN.match(next_text)
            ):
                break

            unfinished = (
                heading.count("(") > heading.count(")")
                or heading.count("«") > heading.count("»")
                or heading.rstrip().endswith(
                    ("или", "и", "для", "по", "на", "с", "в", "из", "при")
                )
            )

            if not unfinished:
                break

            heading += " " + next_text
            index += 1

        heading = " ".join(heading.split())

        if is_valid_section_heading(heading):
            headings.append(heading)

    return headings
