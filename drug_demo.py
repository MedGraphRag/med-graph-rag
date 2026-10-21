
from pathlib import Path

from src.nlp.grls_resolver import DrugResolver


def main():
    parquet_path = Path("data/processed/grls_normalized.parquet")

    if not parquet_path.exists():
        print(f"Файл реестра не найден: {parquet_path}")
        return

    resolver = DrugResolver(parquet_path)

    print("=== Демонстрация поиска лекарств ===")
    print("Введи exit для выхода.\n")

    while True:
        query = input("Введите название препарата: ").strip()

        if query.lower() == "exit":
            break

        if not query:
            print("Введите название препарата.\n")
            continue

        result = resolver.search(query)

        print(f"\nЗапрос: {result.query}")
        print(f"Найдено записей: {len(result.matches)}")
        print(
            "Неоднозначный результат: "
            + ("да" if result.ambiguous else "нет")
        )

        if not result.matches:
            print("Совпадений не найдено.\n")
            continue

        for index, drug in enumerate(result.matches, start=1):
            print(f"\n{index}. {drug.trade_name}")
            print(f"   МНН: {drug.inn or 'не указано'}")
            print(
                "   АТХ-коды: "
                + (", ".join(drug.atx_codes) or "не указаны")
            )

        print()


if __name__ == "__main__":
    main()
