# count_formats_by_files.py
import os
import re
import csv
from collections import Counter
from pathlib import Path

# === НАСТРОЙКИ ===
ROOT = r"C:\Users\Sergey\MAGA\Kimch\НМЦК_сборка_для_подсчета"  # берём ТОЛЬКО файлы из этой папки (без рекурсии)
OUT_DIR = ROOT                                                 # куда класть итоговые CSV

# 19- и 11-значные номера в имени файла
RE_19 = re.compile(r"(?<!\d)\d{19}(?!\d)")
RE_11 = re.compile(r"(?<!\d)\d{11}(?!\d)")

def file_ext(name: str) -> str:
    """
    Возвращает расширение файла без точки, в нижнем регистре.
    Если расширения нет — 'noext'.
    'archive.tar.gz' -> 'gz' (берём последнее расширение).
    """
    base = os.path.basename(name)
    if "." not in base or (base.startswith(".") and base.count(".") == 1):
        return "noext"
    return base.rsplit(".", 1)[-1].lower()

def which_group_by_filename(filename: str):
    """
    Определяем группу по ИМЕНИ ФАЙЛА:
      - если есть 19-значный номер — '19'
      - иначе если есть 11-значный — '11'
      - иначе None (такие файлы пропускаем)
    Приоритет 19-значного, если вдруг встретятся оба.
    """
    name = os.path.basename(filename)
    if RE_19.search(name):
        return "19"
    if RE_11.search(name):
        return "11"
    return None

def main():
    p = Path(ROOT)
    if not p.is_dir():
        print(f"Папка не найдена: {ROOT}")
        return

    counts = {"19": Counter(), "11": Counter()}
    total_files = {"19": 0, "11": 0}

    # Берём только файлы верхнего уровня (без рекурсии)
    for entry in p.iterdir():
        if not entry.is_file():
            continue
        group = which_group_by_filename(entry.name)
        if group not in ("19", "11"):
            continue
        ext = file_ext(entry.name)
        counts[group][ext] += 1
        total_files[group] += 1

    # Печать сводки
    def print_summary(label: str):
        print(f"\n{label}-значная (по ИМЕНАМ ФАЙЛОВ):")
        print(f"  Файлов: {total_files[label]}")
        for ext, cnt in counts[label].most_common():
            print(f"  {ext:8s} {cnt}")

    print_summary("19")
    print_summary("11")

    # Запись CSV
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    out_19 = Path(OUT_DIR) / "ext_counts_files_19.csv"
    out_11 = Path(OUT_DIR) / "ext_counts_files_11.csv"

    def write_csv(path: Path, label: str):
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["group", "ext", "count", "files_in_group"])
            for ext, cnt in counts[label].most_common():
                w.writerow([label, ext, cnt, total_files[label]])

    write_csv(out_19, "19")
    write_csv(out_11, "11")

    print("\nCSV сохранены:")
    print(f"  {out_19}")
    print(f"  {out_11}")

if __name__ == "__main__":
    main()
