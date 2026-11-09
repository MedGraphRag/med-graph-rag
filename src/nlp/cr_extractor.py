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



# --------------------------------------------------------------------------
# Извлечение таблиц
# --------------------------------------------------------------------------

# Максимальное расстояние от подписи до таблицы (в пунктах PDF).
CAPTION_MAX_GAP = 120

# Минимальное количество векторных объектов для запуска find_tables()
# на странице без подписи и возможного продолжения таблицы.
MIN_DRAWINGS_FOR_TABLE = 4

CAPTION_PATTERN = re.compile(
    r"^\s*таблица\s*(\d+)\b",
    re.IGNORECASE,
)

CAPTION_QUICK_PATTERN = re.compile(
    r"^\s*таблица\s*\d+",
    re.IGNORECASE | re.MULTILINE,
)


def _normalize_cell(text: str | None) -> str | None:
    """Нормализует содержимое ячейки, сохраняя переносы строк."""
    if not isinstance(text, str):
        return None

    lines = [
        re.sub(r"[ \t\u00a0]+", " ", line).strip()
        for line in text.splitlines()
    ]
    text = "\n".join(line for line in lines if line)

    text = re.sub(r" +([,.;:)])", r"\1", text)
    text = re.sub(r"\( +", "(", text)

    return text or None


def _find_captions(
    page: "pymupdf.Page",
) -> list[tuple[int, "pymupdf.Rect"]]:
    """Находит подписи вида «Таблица N» и возвращает их координаты."""
    captions: dict[int, pymupdf.Rect] = {}

    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            line_text = "".join(
                span["text"] for span in line.get("spans", [])
            )
            match = CAPTION_PATTERN.match(line_text)

            if match:
                number = int(match.group(1))
                captions.setdefault(number, pymupdf.Rect(line["bbox"]))

    return sorted(captions.items(), key=lambda item: item[1].y0)


def _may_contain_table(
    page: "pymupdf.Page",
    has_caption: bool,
    carry: bool,
) -> bool:
    """Решает, нужно ли запускать затратный поиск таблиц на странице."""
    if has_caption or carry:
        return True

    return len(page.get_cdrawings()) >= MIN_DRAWINGS_FOR_TABLE


def _cell_text_from_words(
    words: list,
    bbox: tuple | None,
    fallback: str | None,
) -> str | None:
    """Восстанавливает текст ячейки по словам страницы.

    Слова группируются по строкам сверху вниз, внутри строки —
    слева направо. Если координаты ячейки недоступны или в ней
    не найдены слова, используется результат table.extract().
    """
    if bbox is None:
        return fallback

    x0, y0, x1, y1 = bbox

    cell_words = [
        word
        for word in words
        if x0 <= (word[0] + word[2]) / 2 <= x1
        and y0 <= (word[1] + word[3]) / 2 <= y1
    ]

    if not cell_words:
        return fallback

    cell_words.sort(
        key=lambda word: ((word[1] + word[3]) / 2, word[0])
    )

    lines = [[cell_words[0]]]

    for word in cell_words[1:]:
        reference = lines[-1][0]
        tolerance = (reference[3] - reference[1]) * 0.5
        center_diff = abs(
            (word[1] + word[3]) / 2
            - (reference[1] + reference[3]) / 2
        )

        if center_diff <= tolerance:
            lines[-1].append(word)
        else:
            lines.append([word])

    return "\n".join(
        " ".join(
            word[4] for word in sorted(line, key=lambda item: item[0])
        )
        for line in lines
    )


def extract_tables(
    pdf_path: str | Path,
) -> list[dict[str, int | list | str | None]]:
    """Извлекает таблицы из PDF и нормализует содержимое ячеек.

    Статусы:
    - extracted: таблица обнаружена и содержимое извлечено;
    - unprocessed: обнаружена подпись или таблица, но извлечение
      не удалось, либо find_tables() завершился ошибкой.

    На страницах без подписи поиск выполняется, если предыдущая таблица
    дошла до нижней части страницы или на текущей странице достаточно
    векторных объектов. Это ускоряет обработку, но является эвристикой:
    таблица без границ и без подписи может быть пропущена.
    """
    path = Path(pdf_path)

    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {path}")

    results = []

    try:
        with pymupdf.open(path) as document:
            if document.is_encrypted:
                raise ValueError(
                    f"Encrypted PDF is not supported: {path}"
                )

            carry = False

            for page_number, page in enumerate(document, start=1):
                text = page.get_text("text")
                has_caption = bool(CAPTION_QUICK_PATTERN.search(text))
                captions = _find_captions(page) if has_caption else []

                if not _may_contain_table(
                    page,
                    has_caption or bool(captions),
                    carry,
                ):
                    carry = False
                    continue

                try:
                    tables = page.find_tables().tables
                except Exception as error:
                    # Не теряем ошибку даже на странице без подписи:
                    # она могла проверяться из-за carry или drawings.
                    results.append({
                        "page": page_number,
                        "table_index": None,
                        "rows": None,
                        "status": "unprocessed",
                        "message": f"find_tables failed: {error}",
                    })
                    carry = False
                    continue

                # Извлекаем слова один раз на страницу, а не для каждой ячейки.
                words = page.get_text("words") if tables else []

                # Сопоставляем подписи и таблицы по расстоянию между ними.
                captioned: set[int] = set()
                unmatched_captions: list[int] = []

                for number, caption_rect in captions:
                    candidates = [
                        (
                            tables[index].bbox[1] - caption_rect.y1,
                            index,
                        )
                        for index in range(len(tables))
                        if index not in captioned
                        and -5
                        <= tables[index].bbox[1] - caption_rect.y1
                        <= CAPTION_MAX_GAP
                    ]

                    if candidates:
                        captioned.add(min(candidates)[1])
                    else:
                        unmatched_captions.append(number)

                table_reaches_bottom = False

                for index, table in enumerate(tables):
                    table_index = index + 1
                    has_table_caption = index in captioned

                    try:
                        # Если найденная область доходит до нижнего края,
                        # проверим следующую страницу на продолжение.
                        if table.bbox[3] > page.rect.height * 0.85:
                            table_reaches_bottom = True

                        extracted_rows = table.extract()
                        table_rows = getattr(table, "rows", None)

                        use_bboxes = (
                            table_rows is not None
                            and len(table_rows) == len(extracted_rows)
                        )

                        rows = []

                        for row_index, row in enumerate(extracted_rows):
                            bboxes = (
                                table_rows[row_index].cells
                                if use_bboxes
                                else [None] * len(row)
                            )

                            if len(bboxes) != len(row):
                                bboxes = [None] * len(row)

                            normalized_row = [
                                _normalize_cell(
                                    _cell_text_from_words(
                                        words,
                                        bbox,
                                        cell,
                                    )
                                )
                                for bbox, cell in zip(bboxes, row)
                            ]
                            rows.append(normalized_row)

                        has_content = any(
                            any(cell for cell in row)
                            for row in rows
                        )

                        # Без подписи отбрасываем объекты, которые
                        # недостаточно похожи на настоящие таблицы.
                        if not has_table_caption and (
                            not has_content
                            or len(rows) < 2
                            or max(len(row) for row in rows) < 2
                        ):
                            continue

                        if not has_content:
                            message = "Empty table"
                        elif has_table_caption:
                            message = None
                        else:
                            message = (
                                "No caption "
                                "(possibly a continuation of a table)"
                            )

                        results.append({
                            "page": page_number,
                            "table_index": table_index,
                            "rows": rows if has_content else None,
                            "status": (
                                "extracted"
                                if has_content
                                else "unprocessed"
                            ),
                            "message": message,
                        })

                    except Exception as error:
                        results.append({
                            "page": page_number,
                            "table_index": table_index,
                            "rows": None,
                            "status": "unprocessed",
                            "message": str(error),
                        })

                for number in unmatched_captions:
                    results.append({
                        "page": page_number,
                        "table_index": None,
                        "rows": None,
                        "status": "unprocessed",
                        "message": (
                            f"Caption «Таблица {number}» found, "
                            "but no table detected"
                        ),
                    })

                carry = table_reaches_bottom

            return results

    except ValueError:
        raise
    except Exception as error:
        raise RuntimeError(
            f"Failed to extract tables from {path}"
        ) from error
