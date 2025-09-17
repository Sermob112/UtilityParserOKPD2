# CSVMerger.py — объединяет CSV и оставляет только уникальные purchase_number

import csv
from pathlib import Path
from typing import List, Dict, Tuple

# ======= НАСТРОЙКИ =======
INPUT_DIR = r"C:\Users\Sergey\MAGA\UtilityParser\NMCK\Files 223"
ADD_SOURCE_COLUMN = True        # добавить колонку source_file
RECURSIVE = False               # True — собирать CSV и из подпапок
KEEP_EMPTY_PURCHASE = False     # брать строки без purchase_number (обычно False)

# Группы для мерджа: по подстроке в имени файла -> имя итогового файла
GROUPS = [
    ("found_any",     "found_any_full.csv"),
    ("not_found_any", "not_found_any_full.csv"),
    ("only_639",      "only_639_full.csv"),
]

# приоритет колонок в сводном отчёте
PREFERRED_ORDER = [
    "purchase_number", "articles_found", "orders_found", "status",
    "file_name", "file_path",
    "article_snippets", "order_snippets", "order_639_snippets",
    "source_file",
]

# ранжирование статусов для выбора лучшей строки при дублях
STATUS_RANK = {"orders+article": 3, "orders": 2, "article": 1, "missing": 0}
# =========================

def sniff_delimiter(sample: str, default=";") -> str:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",",";","\t","|"])
        return dialect.delimiter
    except Exception:
        return default

def read_csv_as_dicts(path: Path) -> Tuple[List[str], List[Dict[str,str]]]:
    with path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        delimiter = sniff_delimiter(sample, default=";")
        reader = csv.DictReader(f, delimiter=delimiter)
        headers = [h.strip("\ufeff") for h in (reader.fieldnames or [])]
        rows: List[Dict[str,str]] = []
        for row in reader:
            rows.append({(k or "").strip(): (v if v is not None else "") for k, v in row.items()})
        return headers, rows

def list_csv_files(root: Path, recursive: bool) -> List[Path]:
    pattern = "**/*.csv" if recursive else "*.csv"
    return sorted(root.glob(pattern))

def union_headers(list_of_headers: List[List[str]]) -> List[str]:
    seen = {}
    for headers in list_of_headers:
        for h in headers:
            if h not in seen:
                seen[h] = True
    result: List[str] = []
    for h in PREFERRED_ORDER:
        if h in seen:
            result.append(h); seen.pop(h, None)
    result.extend(sorted(seen.keys()))
    return result

def row_nonempty_count(row: Dict[str,str]) -> int:
    return sum(1 for v in row.values() if str(v).strip())

def choose_better_row(a: Dict[str,str], b: Dict[str,str]) -> Dict[str,str]:
    """Выбрать лучшую строку по статусу, затем по числу непустых полей."""
    ra = STATUS_RANK.get(a.get("status","").strip().lower(), -1)
    rb = STATUS_RANK.get(b.get("status","").strip().lower(), -1)
    if ra != rb:
        return a if ra > rb else b
    ca, cb = row_nonempty_count(a), row_nonempty_count(b)
    if ca != cb:
        return a if ca > cb else b
    return a  # стабильность по первому

def merge_group(input_dir: Path, keyword: str, output_name: str) -> Tuple[int,int,int,int,List[Path]]:
    # кандидаты (исключаем уже собранные full)
    candidates = [
        p for p in list_csv_files(input_dir, RECURSIVE)
        if keyword.lower() in p.name.lower()
           and p.name.lower() != output_name.lower()
           and not p.name.lower().endswith("_full.csv")
    ]
    if not candidates:
        return (0,0,0,0,[])

    all_headers: List[List[str]] = []
    all_rows: List[Dict[str,str]] = []
    files_used: List[Path] = []

    for p in candidates:
        headers, rows = read_csv_as_dicts(p)
        if not headers: 
            continue
        files_used.append(p)
        all_headers.append(headers)
        if ADD_SOURCE_COLUMN:
            for r in rows:
                r.setdefault("source_file", p.name)
        all_rows.extend(rows)

    if not all_rows:
        return (len(files_used), 0, 0, 0, files_used)

    if ADD_SOURCE_COLUMN:
        all_headers.append(["source_file"])
    headers_union = union_headers(all_headers)

    # --- Дедупликация строго по purchase_number ---
    by_pn: Dict[str, Dict[str,str]] = {}
    skipped_empty = 0
    for r in all_rows:
        pn = (r.get("purchase_number") or "").strip()
        if not pn and not KEEP_EMPTY_PURCHASE:
            skipped_empty += 1
            continue
        if pn in by_pn:
            by_pn[pn] = choose_better_row(by_pn[pn], r)
        else:
            by_pn[pn] = r

    unique_rows = list(by_pn.values())

    out_path = input_dir / output_name
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(headers_union)
        for r in unique_rows:
            w.writerow([r.get(h,"") for h in headers_union])

    rows_in = len(all_rows)
    rows_out = len(unique_rows)
    return (len(files_used), rows_in, rows_out, len(headers_union), files_used)

def main():
    input_dir = Path(INPUT_DIR).expanduser()
    if not input_dir.is_dir():
        print(f"Папка не найдена: {input_dir}")
        return

    print(f"Ищем CSV в: {input_dir} (рекурсивно: {RECURSIVE})")
    for keyword, out_name in GROUPS:
        files_count, rows_in, rows_out, cols, files_used = merge_group(input_dir, keyword, out_name)
        print(f"\nГруппа: '{keyword}' -> {out_name}")
        print(f"  Файлов найдено: {files_count}")
        if files_used:
            print("  Использованы файлы:")
            for p in files_used:
                print(f"    - {p.name}")
        print(f"  Строк на входе (до агрегирования): {rows_in}")
        print(f"  Уникальных purchase_number: {rows_out}")
        print(f"  Колонок в результате: {cols}")

    print("\nГотово.")

if __name__ == "__main__":
    main()
