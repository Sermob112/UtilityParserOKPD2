# FilterFoundAnyByTxt.py
import csv
import os
import re
from pathlib import Path
from typing import Optional, Set, List, Dict, Tuple

# ===== ПУТИ =====
BASE_DIR = r"C:\Users\Sergey\MAGA\UtilityParser\NMCK\Files 223"
TXT_PATH = os.path.join(BASE_DIR, "purchase_numbers_from_excel_223.txt")
ANY_FULL_CSV = os.path.join(BASE_DIR, "found_any_full.csv")
MAKE_BACKUP = True  # создать found_any_full.backup.csv перед перезаписью

RE_19 = re.compile(r"(?<!\d)(\d{19})(?!\d)")
RE_11 = re.compile(r"(?<!\d)(\d{11})(?!\d)")
def extract_19(s: str) -> Optional[str]:
    if not s:
        return None
    m = RE_11.search(str(s))
    return m.group(1) if m else None

def sniff_delimiter(sample: str, default=";") -> str:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",",";","\t","|"])
        return dialect.delimiter
    except Exception:
        return default

def load_txt_numbers_ordered(path: str) -> Tuple[List[str], Set[str], Dict[str,int]]:
    """
    Читает TXT, возвращает:
      - ordered: уникальные 19-значные purchase_number в порядке появления
      - allowed: множество тех же номеров
      - order_idx: словарь номер -> индекс порядка (0..)
    """
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] TXT не найден: {path}")
        return [], set(), {}
    ordered: List[str] = []
    allowed: Set[str] = set()
    with p.open("r", encoding="utf-8-sig", errors="ignore") as f:
        for line in f:
            pn = extract_19(line)
            if pn and pn not in allowed:
                allowed.add(pn)
                ordered.append(pn)
    order_idx = {pn: i for i, pn in enumerate(ordered)}
    return ordered, allowed, order_idx

def read_csv_rows(path: str) -> (List[str], List[Dict[str, str]], str):
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] CSV не найден: {path}")
        return [], [], ";"
    with p.open("r", encoding="utf-8-sig", errors="ignore", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        delim = sniff_delimiter(sample, default=";")
        rdr = csv.DictReader(f, delimiter=delim)
        headers = [h.strip("\ufeff") for h in (rdr.fieldnames or [])]
        rows: List[Dict[str,str]] = []
        for row in rdr:
            rows.append({(k or "").strip(): (v if v is not None else "") for k, v in row.items()})
        return headers, rows, delim

def write_csv_rows(path: str, headers: List[str], rows: List[Dict[str,str]], delimiter: str):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=delimiter)
        w.writerow(headers)
        for r in rows:
            w.writerow([r.get(h, "") for h in headers])

def main():
    # 1) Загружаем номера и их порядок из TXT
    ordered, allowed, order_idx = load_txt_numbers_ordered(TXT_PATH)
    if not ordered:
        print("[WARN] Список из TXT пуст — ничего не фильтруем.")
        return

    # 2) Читаем found_any_full.csv
    headers, rows, delim = read_csv_rows(ANY_FULL_CSV)
    if not headers:
        print("[ERROR] Не удалось прочитать заголовки CSV.")
        return

    # 3) Ищем колонку purchase_number
    header_map = {h.strip().lower(): h for h in headers}
    pn_col = header_map.get("purchase_number")
    if not pn_col:
        print("[ERROR] В found_any_full.csv нет колонки 'purchase_number'.")
        return

    # 4) Фильтруем и сортируем по порядку из TXT
    kept: List[Dict[str,str]] = []
    removed_count = 0
    for r in rows:
        pn = extract_19(r.get(pn_col, ""))
        if pn and pn in allowed:
            kept.append(r)
        else:
            removed_count += 1

    # Сортировка: в точном порядке purchase_number, как в TXT
    # (stable sort: если в CSV было несколько строк на один номер — сохранится их относительный порядок)
    def sort_key(r: Dict[str,str]) -> int:
        pn = extract_19(r.get(pn_col, ""))
        # Все kept по определению есть в order_idx, но подстрахуемся
        return order_idx.get(pn, 10**12)

    kept.sort(key=sort_key)

    # 5) Резервная копия и перезапись
    if MAKE_BACKUP and os.path.exists(ANY_FULL_CSV):
        backup_path = os.path.join(BASE_DIR, "found_any_full.backup.csv")
        try:
            os.replace(ANY_FULL_CSV, backup_path)
            print(f"Сделана копия: {backup_path}")
        except Exception as e:
            print(f"[WARN] Не удалось сделать резервную копию: {e}")

    write_csv_rows(ANY_FULL_CSV, headers, kept, delim)

    # 6) Итого
    missing_in_csv = [pn for pn in ordered if pn not in {extract_19(r.get(pn_col,"")) for r in rows}]
    print(f"Всего строк в исходном found_any_full: {len(rows)}")
    print(f"Оставлено (по списку TXT): {len(kept)}")
    print(f"Удалено: {removed_count}")
    if missing_in_csv:
        print(f"Из TXT не найдены в CSV (кол-во {len(missing_in_csv)}): первые 20 -> {missing_in_csv[:20]}")
    print("Готово: found_any_full.csv перезаписан и отсортирован по порядку из purchase_numbers_from_excel_44.txt.")

if __name__ == "__main__":
    main()
