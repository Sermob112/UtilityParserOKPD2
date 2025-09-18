# join_price_to_nmck.py
import re
import pandas as pd
from typing import Optional, Set

PRICE_XLSX = r"C:\Users\Sergey\MAGA\UtilityParser\NMCK\Files 223\Цена.xlsx"       # без заголовков
NMCK_XLSX  = r"C:\Users\Sergey\MAGA\UtilityParser\NMCK\Files 223\НМЦК 223.xlsx"   # обычно с заголовками
OUT_XLSX   = r"C:\Users\Sergey\MAGA\UtilityParser\NMCK\Files 223\НМЦК 223_with_price.xlsx"

PRICE_SHEET = 0
NMCK_SHEET  = 0

RE_ID = re.compile(r"(?<!\d)(\d{11}|\d{19})(?!\d)")

def extract_id_digits(s: str) -> Optional[str]:
    if s is None: return None
    m = RE_ID.search(str(s))
    return m.group(1) if m else None

def parse_price(v) -> float:
    if v is None: return 0.0
    s = str(v).replace("\u00A0"," ").replace("\u202F"," ").replace("\u2009"," ")
    s = s.replace("₽","").replace(" ","").replace("\t","").replace(",",".")
    s = re.sub(r"[^0-9.]", "", s)
    try: return float(s) if s else 0.0
    except: return 0.0

def detect_price_id_col(df_nohdr: pd.DataFrame) -> int:
    # ищем колонку, где чаще всего встречаются 11/19-значные номера
    best_j, best_score = 0, -1
    for j in range(df_nohdr.shape[1]):
        score = df_nohdr.iloc[:, j].astype(str).map(lambda x: bool(RE_ID.search(x))).sum()
        if score > best_score:
            best_j, best_score = j, score
    return best_j  # для твоего файла это 1 (вторая колонка)

def detect_nmck_pn_col(df: pd.DataFrame) -> str:
    # типичные имена
    for c in df.columns:
        k = str(c).strip().lower().replace(" ", "")
        if k in {"purchase_number","номерзакупки","номеризвещения","номерзакупкиизвещения","номерзакупкиизвещении"}:
            return c
    # иначе – максимальные попадания ID
    best, score = None, -1
    for c in df.columns:
        sc = df[c].astype(str).map(lambda x: bool(RE_ID.search(x))).sum()
        if sc > score:
            best, score = c, sc
    return best if best is not None else df.columns[0]

def main():
    # ---- Цена.xlsx без заголовков ----
    price_raw = pd.read_excel(PRICE_XLSX, sheet_name=PRICE_SHEET, header=None, dtype=str, engine="openpyxl")
    if price_raw.shape[1] < 2:
        raise ValueError("В Цена.xlsx ожидается минимум 2 столбца (ID и Цена).")

    id_col_idx    = detect_price_id_col(price_raw)  # <-- тут ловим вторую колонку с №
    price_col_idx = 3 if price_raw.shape[1] > 3 else detect_price_id_col(price_raw)  # обычно 4-й столбец

    price_df = pd.DataFrame({
        "_key":   price_raw.iloc[:, id_col_idx].map(extract_id_digits),
        "_price": price_raw.iloc[:, price_col_idx].map(parse_price),
    })
    price_df = price_df[price_df["_key"].notna()].copy()
    price_df = price_df.sort_values("_key").drop_duplicates(subset=["_key"], keep="first")

    price_map = dict(zip(price_df["_key"], price_df["_price"]))

    # ---- НМЦК 223.xlsx ----
    nmck_df = pd.read_excel(NMCK_XLSX, sheet_name=NMCK_SHEET, header=0, dtype=str, engine="openpyxl")
    if nmck_df.columns.astype(str).str.contains("Unnamed").all():
        nmck_df = pd.read_excel(NMCK_XLSX, sheet_name=NMCK_SHEET, header=None, dtype=str, engine="openpyxl")
        nmck_df.columns = [f"col_{i+1}" for i in range(nmck_df.shape[1])]

    pn_col = detect_nmck_pn_col(nmck_df)
    nmck_df["_key"] = nmck_df[pn_col].map(extract_id_digits)

    # ---- Маппинг цены по ключу ----
    nmck_df["Цена"] = nmck_df["_key"].map(price_map).fillna(0.0)

    # ---- Диагностика ----
    price_keys: Set[str] = set(k for k in price_df["_key"] if pd.notna(k))
    nmck_keys:  Set[str] = set(k for k in nmck_df["_key"]  if pd.notna(k))
    inter = price_keys & nmck_keys
    print(f"ID-колонка в Цена.xlsx: #{id_col_idx+1}; ценовая колонка: #{price_col_idx+1}")
    print(f"Колонка номера в НМЦК: '{pn_col}'")
    print(f"Ключей в Цена.xlsx: {len(price_keys)}; в НМЦК: {len(nmck_keys)}; пересечение: {len(inter)}")
    print(f"Строк с ненулевой ценой: {(nmck_df['Цена']>0).sum()} из {len(nmck_df)}")
    print(f"Проверка ключа 32514953432: цена =", price_map.get("32514953432"))

    # ---- Сохранение ----
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        nmck_df.to_excel(w, index=False)
    print(f"Файл сохранён: {OUT_XLSX}")

if __name__ == "__main__":
    main()
