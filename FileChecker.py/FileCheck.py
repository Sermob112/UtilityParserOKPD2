# count_formats_by_purchase_len.py
import os
import re
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

# === НАСТРОЙКИ ===
ROOT = r"C:\Users\Sergey\MAGA\Kimch\НМЦК_сборка_для_подсчета"
SAVE_CSV = True  # если True — сохранит ext_counts_19.csv и ext_counts_11.csv рядом со скриптом

# 19- и 11-значные номера
RE_19 = re.compile(r"(?<!\d)\d{19}(?!\d)")
RE_11 = re.compile(r"(?<!\d)\d{11}(?!\d)")

def extract_group_from_dirname(dirname: str) -> Optional[str]:
    """
    Возвращает '19' если в названии папки есть 19-значный номер,
    '11' — если есть 11-значный; иначе None.
    Приоритет 19, если вдруг встретились оба.
    """
    if RE_19.search(dirname):
        return "19"
    if RE_11.search(dirname):
        return "11"
    return None

def file_ext(name: str) -> str:
    """
    Возвращает расширение файла без точки, в нижнем регистре.
    Если расширения нет — 'noext'.
    Примеры:
      'a.docx' -> 'docx'
      'archive.tar.gz' -> 'gz'  (берём последнее)
      'README' -> 'noext'
    """
    base = os.path.basename(name)
    if "." not in base or base.startswith(".") and base.count(".") == 1:
        return "noext"
    return base.rsplit(".", 1)[-1].lower()

def main():
    counts = {
        "19": Counter(),
        "11": Counter(),
    }
    files_total = {"19": 0, "11": 0}
    folders_total = {"19": 0, "11": 0}

    # кеш соответствия папки к группе, чтобы не считать каждый раз
    dir_group_cache = {}

    for dirpath, dirnames, filenames in os.walk(ROOT):
        # Определим группу для текущей папки:
        # 1) по имени этой папки,
        # 2) если не нашли — посмотрим на ближайшего предка в кеше (наследуем группу от родителя)
        p = Path(dirpath)
        key = str(p)
        if key in dir_group_cache:
            group = dir_group_cache[key]
        else:
            group = extract_group_from_dirname(p.name)
            if group is None:
                # унаследуем от родителя, если у него уже известная группа
                parent = str(p.parent)
                group = dir_group_cache.get(parent, None)
            dir_group_cache[key] = group

        # Если эта папка принадлежит к 19/11 — учитываем файлы
        if group in ("19", "11"):
            folders_total[group] += 1
            for fname in filenames:
                ext = file_ext(fname)
                counts[group][ext] += 1
                files_total[group] += 1

        # Проставим группу для подкаталогов в кеше (если у текущей папки группа есть,
        # то и подкаталоги наследуют её, пока у них самих не найдётся явный номер в названии)
        if group in ("19", "11"):
            for d in dirnames:
                child_path = str(Path(dirpath) / d)
                if child_path not in dir_group_cache:
                    # если у ребёнка явно указан другой номер — он его перезапишет на своей итерации
                    dir_group_cache[child_path] = group

    # Печать сводки
    def print_summary(label: str, counter: Counter, total_files: int, total_folders: int):
        print(f"\n{label}-значная:")
        print(f"  Папок: {total_folders}, файлов: {total_files}")
        for ext, cnt in counter.most_common():
            print(f"  {ext:8s} {cnt}")

    print_summary(19, counts["19"], files_total["19"], folders_total["19"])
    print_summary(11, counts["11"], files_total["11"], folders_total["11"])

    # Сохранение в CSV (по желанию)
    if SAVE_CSV:
        base_dir = Path(__file__).parent
        out_19 = base_dir / "ext_counts_19.csv"
        out_11 = base_dir / "ext_counts_11.csv"
        for path, counter, label, total_f, total_d in [
            (out_19, counts["19"], "19", files_total["19"], folders_total["19"]),
            (out_11, counts["11"], "11", files_total["11"], folders_total["11"]),
        ]:
            with path.open("w", encoding="utf-8-sig", newline="") as f:
                w = csv.writer(f, delimiter=";")
                w.writerow(["group", "ext", "count", "folders_in_group", "files_in_group"])
                for ext, cnt in counter.most_common():
                    w.writerow([label, ext, cnt, total_d, total_f])
        print(f"\nCSV сохранены: {out_19} , {out_11}")

if __name__ == "__main__":
    main()
