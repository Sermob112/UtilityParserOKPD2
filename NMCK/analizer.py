# summarize_nmck_314.py
import re
import pandas as pd

# укажи путь к файлу:
XLSX_PATH = r"C:\Users\Sergey\MAGA\UtilityParser\NMCK\Files 223\НМЦК 223.xlsx"  # например: r"C:\Users\Sergey\MAGA\UtilityParser\NMCK\Files\НМЦК_314.xlsx"

# имена колонок по ТЗ (если не найдутся — возьмём 2-ю и 3-ю по позиции)
COL_ART = "Приказ 23"
COL_ORD = "Приказ 639 либо 567"

re_22  = re.compile(r"(?<!\d)22(?!\d)")
re_34  = re.compile(r"(?<!\d)34(?!\d)")
re_567 = re.compile(r"(?<!\d)567(?!\d)")
re_639 = re.compile(r"(?<!\d)639(?!\d)")
re_not = re.compile(r"не\s*наш[её]л", re.IGNORECASE)

def norm(s):
    if pd.isna(s):
        return ""
    return str(s).strip().lower()

def pick_cols(df: pd.DataFrame):
    # пробуем по именам
    if COL_ART in df.columns and COL_ORD in df.columns:
        return df[[COL_ART, COL_ORD]].copy()
    # иначе — 2-я и 3-я по позиции (0-based -> 1 и 2)
    if df.shape[1] >= 3:
        return df.iloc[:, [1, 2]].copy().rename(columns={df.columns[1]: COL_ART, df.columns[2]: COL_ORD})
    # fallback: если столбцов меньше
    raise ValueError("В таблице меньше 3 столбцов — не из чего собирать сводную.")

def classify_pair(a: str, b: str) -> str:
    s = f"{norm(a)}|{norm(b)}"
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
    df = pd.read_excel(XLSX_PATH, dtype=str, engine="openpyxl")
    sub = pick_cols(df)
    # классификация
    sub["category"] = [classify_pair(a, b) for a, b in zip(sub.iloc[:,0], sub.iloc[:,1])]
    counts = sub["category"].value_counts(dropna=False).rename_axis("category").to_frame("count")
    # аккуратный порядок вывода
    order = ["567|639", "639", "567", "не найдено"]
    counts = counts.reindex(order).fillna(0).astype(int)
    # вывод
    total = len(sub)
    print(f"Всего строк: {total}")
    print("\nСводная по категориям:")
    for cat, row in counts.itertuples():
        print(f"  {cat:10s} : {row}")
    # при желании — показать первые примеры по каждой категории:
    # for cat in order:
    #     ex = sub[sub["category"]==cat].head(3)
    #     if not ex.empty:
    #         print(f"\nПримеры {cat}:")
    #         print(ex.to_string(index=False))

if __name__ == "__main__":
    main()
