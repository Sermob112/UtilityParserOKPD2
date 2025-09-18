# analizer.py
import re
import pandas as pd
from typing import Optional

# путь к файлу:
XLSX_PATH = r"C:\Users\Sergey\MAGA\UtilityParser\NMCK\Files 223\НМЦК 223_with_price.xlsx"

# имена колонок с признаками статей/приказов
COL_ART = "Приказ 23"
COL_ORD = "Приказ 639 либо 567"

# --- паттерны ---
re_22  = re.compile(r"(?<!\d)22(?!\d)")
re_34  = re.compile(r"(?<!\d)34(?!\d)")
re_567 = re.compile(r"(?<!\d)567(?!\d)")
re_639 = re.compile(r"(?<!\d)639(?!\d)")
re_not = re.compile(r"не\s*наш[её]л", re.IGNORECASE)

# 19- или 11-значные номера (44-ФЗ и 223-ФЗ)
RE_ID = re.compile(r"(?<!\d)(\d{19}|\d{11})(?!\d)")

def norm(s):
    if pd.isna(s):
        return ""
    return str(s).strip()

def norm_lc(s):
    return norm(s).lower()

def extract_id(s: str) -> Optional[str]:
    m = RE_ID.search(norm(s))
    return m.group(1) if m else None
def add_no_prefix(s: str) -> str:
    s = norm(s)
    if s.startswith("№"):
        return s
    m = RE_ID.search(s)
    return f"№ {m.group(1)}" if m else s
def ensure_prefixed_id(s: str) -> str:
    """Вернёт с префиксом '№ ' если нашли 19/11-значный id; иначе исходное значение."""
    sid = extract_id(s)
    return f"№ {sid}" if sid else norm(s)

def pick_cols_for_rules(df: pd.DataFrame) -> pd.DataFrame:
    # сначала пробуем точные имена
    if COL_ART in df.columns and COL_ORD in df.columns:
        return df[[COL_ART, COL_ORD]].copy()
    # иначе берём 2-й и 3-й столбцы по позиции (если есть)
    if df.shape[1] >= 3:
        sub = df.iloc[:, [1, 2]].copy()
        sub.columns = [COL_ART, COL_ORD]
        return sub
    raise ValueError("В таблице меньше 3 столбцов — не из чего собирать сводную.")

def find_purchase_col(df: pd.DataFrame) -> Optional[str]:
    # приоритет: 'purchase_number' (любой регистр/пробелы)
    for c in df.columns:
        if norm_lc(c).replace(" ", "") == "purchase_number":
            return c
    # вариант по-русски
    for c in df.columns:
        lc = norm_lc(c)
        if "номер" in lc and ("закуп" in lc or "извещ" in lc):
            return c
    # эвристика: любой столбец, где встречаются 19/11-значные номера
    for c in df.columns:
        col = df[c].astype(str)
        if col.map(lambda x: bool(RE_ID.search(x))).any():
            return c
    return None

def find_price_col(df: pd.DataFrame) -> Optional[str]:
    # приоритет — "Цена" (любой регистр)
    for c in df.columns:
        if norm_lc(c) == "цена":
            return c
    # fallback: колонка, где встречаются символы ₽ или ,00
    for c in df.columns:
        sample = " ".join(map(str, df[c].head(20).tolist()))
        if "₽" in sample or "," in sample and any(ch.isdigit() for ch in sample):
            return c
    return None

def parse_price(v) -> float:
    """
    '83 700,00 ₽' -> 83700.00
    Убираем пробелы/неразрывные, '₽', меняем ',' на '.'
    Пустые/непарсящиеся -> 0.0
    """
    s = norm(v).replace("\u00A0", " ").replace("\u202F", " ")
    s = s.replace("₽", "").replace(" ", "").replace("\t", "")
    if not s:
        return 0.0
    s = s.replace(",", ".")
    # оставим только [0-9.] (на случай мусора)
    s = re.sub(r"[^0-9.]", "", s)
    try:
        return float(s) if s else 0.0
    except Exception:
        return 0.0

def classify_pair(a: str, b: str) -> str:
    s = f"{norm_lc(a)}|{norm_lc(b)}"
    has567 = bool(re_567.search(s))
    has639 = bool(re_639.search(s))
    has22  = bool(re_22.search(s))
    has34  = bool(re_34.search(s))
    hasNOT = bool(re_not.search(s))
    # правила
    if has567 and has639:
        return "567|639"
    if has639:
        return "639"
    if has567 or has22 or has34:
        return "567"
    if hasNOT:
        return "не найдено"
    return "не найдено"

def main():
    # читаем всё как строки (purchase_number/тексты) + отдельно сконвертим цену
    df = pd.read_excel(XLSX_PATH, dtype=str, engine="openpyxl")

    # --- purchase_number с префиксом № ---
    pn_col = find_purchase_col(df)
    if pn_col:
        # приводим к строке и добавляем префикс только если его нет
        df[pn_col] = df[pn_col].astype(str).map(add_no_prefix)
    else:
        # создаём новую колонку из конкатенации строки
        df["purchase_number"] = (
            df.astype(str).apply(lambda row: add_no_prefix(" ".join(row.values)), axis=1)
        )
        pn_col = "purchase_number"

    with pd.ExcelWriter(XLSX_PATH, engine="openpyxl") as w:
        df.to_excel(w, index=False)
    print(f"\nИсходный файл перезаписан: {XLSX_PATH}")
    # --- цена ---
    price_col = find_price_col(df)
    if price_col is None:
        # если точно знаем имя — можно указать напрямую:
        # price_col = "Цена"
        pass

    price_num = None
    if price_col:
        price_num = df[price_col].map(parse_price)
    else:
        # не нашли — сделаем пустой столбец нулей
        price_num = pd.Series([0.0] * len(df), index=df.index)

    # --- признаки категорий ---
    sub = pick_cols_for_rules(df)
    categories = [classify_pair(a, b) for a, b in zip(sub.iloc[:, 0], sub.iloc[:, 1])]

    # сводная: количество и сумма
    out = pd.DataFrame({
        "purchase_number": df[pn_col],
        "category": categories,
        "price": price_num
    })

    # группируем
    summary = (out.groupby("category", dropna=False)
                  .agg(count=("purchase_number", "count"),
                       sum_price=("price", "sum"))
                  .reset_index())

    # порядок категорий
    order = ["567|639", "639", "567", "не найдено"]
    summary["sort_key"] = summary["category"].apply(lambda x: order.index(x) if x in order else len(order))
    summary = summary.sort_values("sort_key").drop(columns=["sort_key"])

    # форматирование сумм
    def fmt_money(x: float) -> str:
        # красиво с пробелами для тысяч и запятой как в примере
        s = f"{x:,.2f}".replace(",", " ").replace(".", ",")
        return f"{s} ₽"

    total_count = int(summary["count"].sum())
    total_sum = float(summary["sum_price"].sum())
    
    # печать
    print(f"Файл: {XLSX_PATH}")
    print(f"Всего строк: {len(df)}")
    print("\nСводная по категориям (count / sum):")
    for _, row in summary.iterrows():
        cat = row["category"]
        cnt = int(row["count"])
        sm  = fmt_money(float(row["sum_price"]))
        print(f"  {cat:10s} : {cnt:6d} | {sm}")

    print("\nИТОГО:")
    print(f"  Количество: {total_count}")
    print(f"  Сумма     : {fmt_money(total_sum)}")

if __name__ == "__main__":
    main()
