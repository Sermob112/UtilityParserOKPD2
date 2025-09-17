# PDFChecker.py
import os
import re
import csv
from typing import Dict, List, Tuple, Set

# ============== НАСТРОЙКИ ==============
TARGET_DIR = r"C:\Users\Sergey\MAGA\Kimch\НМЦК_сборка\223- фз"
RECURSIVE = False          # обходить подпапки
WRITE_SNIPPETS = True      # писать фрагменты в CSV
TARGET_ARTICLES = {"22", "34"}  # какие статьи считаем целевыми

# OCR-фолбэк (для сканов). Требует pytesseract + pdf2image + установленный Tesseract
DO_OCR_FALLBACK = False
OCR_MAX_PAGES = 5          # максимум страниц на OCR при пустом тексте

# ============== ПАТТЕРНЫ ПОИСКА ==============
# «ст. 22», «ст22», «Ст.34», «ст 22», «статья 22», падежи «статьи/статье/статью/статьёй»
# + допускаем латинские look-alike буквы c/t в сокращении «ст.»
RE_ART_ANY = re.compile(
    r"""
    (?<!\w)
    (?:[сc][тt](?:\s*\.)? | стать(?:я|и|е|ю|ёй|ей))
    \s*
    ([0-9]{1,3}(?:\.\d+)*)    # номер статьи: 22, 34, 22.1, 34.2.5
    \b
    """,
    re.IGNORECASE | re.UNICODE | re.VERBOSE
)

# Приказы 567/639: не часть большего числа слева и
# НЕ должны быть дальше цифры (с учётом пробелов/.,,)
RE_ORDERS = re.compile(r"(?<!\d)(567|639)(?![\s.,]*\d)", re.UNICODE)

# Номер закупки из имени: приоритетно "Закупка № <19 цифр>", запасной — любой 19-значный
RE_NUM_FROM_TITLE = re.compile(r"Закупка\s*№\s*(\d{11})", re.UNICODE | re.IGNORECASE)
RE_ANY_19_DIGITS = re.compile(r"(?<!\d)(\d{11})(?!\d)", re.UNICODE)

# ============== УТИЛИТЫ ==============
def _normalize_ws(s: str) -> str:
    # NBSP / thin space / узкий NBSP / мягкий перенос -> обычный пробел/ничего
    return (s or "").replace("\u00A0", " ").replace("\u2009", " ").replace("\u202F", " ").replace("\u00AD", "")

def extract_purchase_number_from_filename(filename: str) -> str:
    m = RE_NUM_FROM_TITLE.search(filename)
    if m:
        return m.group(1)
    m2 = RE_ANY_19_DIGITS.search(filename)
    return m2.group(1) if m2 else "UNKNOWN"

def write_csv(path: str, rows: List[List[str]], header: List[str]):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(header)
        w.writerows(rows)

def list_pdf_files(root: str, recursive: bool = False) -> List[str]:
    res = []
    if recursive:
        for dp, _, fns in os.walk(root):
            for fn in fns:
                if fn.lower().endswith(".pdf"):
                    res.append(os.path.join(dp, fn))
    else:
        if not os.path.isdir(root):
            return []
        for fn in os.listdir(root):
            if fn.lower().endswith(".pdf"):
                res.append(os.path.join(root, fn))
    return res

# ============== ЧТЕНИЕ PDF ==============
def _read_with_pymupdf(path: str) -> List[str]:
    try:
        import fitz  # PyMuPDF
    except Exception:
        return []
    texts: List[str] = []
    try:
        with fitz.open(path) as doc:
            for page in doc:
                t = page.get_text("text")
                t = _normalize_ws(t)
                if t:
                    texts.append(t)
    except Exception as e:
        print(f"[WARN] PyMuPDF не смог прочитать: {path}\n  {e}")
    return texts

def _read_with_pdfminer(path: str) -> List[str]:
    try:
        from pdfminer.high_level import extract_text
    except Exception:
        return []
    try:
        txt = extract_text(path) or ""
        if not txt:
            return []
        # pdfminer кладёт разрывы страниц как \f — разобьём по страницам
        pages = [p for p in txt.split("\f") if p.strip()]
        return [_normalize_ws(p) for p in pages if p]
    except Exception as e:
        print(f"[WARN] pdfminer.six не смог прочитать: {path}\n  {e}")
        return []

def _read_with_pypdf(path: str) -> List[str]:
    try:
        from pypdf import PdfReader
    except Exception:
        return []
    texts: List[str] = []
    try:
        reader = PdfReader(path)
        for pg in reader.pages:
            t = pg.extract_text() or ""
            t = _normalize_ws(t)
            if t:
                texts.append(t)
    except Exception as e:
        print(f"[WARN] pypdf не смог прочитать: {path}\n  {e}")
    return texts

def _read_with_ocr(path: str, max_pages: int = OCR_MAX_PAGES) -> List[str]:
    # Опционально: OCR первых страниц для сканов (если включено и установлено всё необходимое)
    if not DO_OCR_FALLBACK:
        return []
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except Exception:
        return []
    texts: List[str] = []
    try:
        images = convert_from_path(path, first_page=1, last_page=max_pages)
        for img in images:
            t = pytesseract.image_to_string(img, lang="rus+eng")
            t = _normalize_ws(t)
            if t:
                texts.append(t)
    except Exception as e:
        print(f"[WARN] OCR не удалось для: {path}\n  {e}")
    return texts

def read_pdf_texts(path: str) -> List[str]:
    # Пробуем по приоритету: PyMuPDF → pdfminer → pypdf → OCR (опция)
    for reader in (_read_with_pymupdf, _read_with_pdfminer, _read_with_pypdf):
        texts = reader(path)
        if texts:
            return texts
    # Последний шанс — OCR (если включён)
    return _read_with_ocr(path)

# ============== ПОИСК СОВПАДЕНИЙ ==============
def find_matches(texts: List[str]) -> Tuple[Set[str], Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Возвращает:
      - found_articles: множество корней статей {'22', '34'} найденных в тексте
      - orders_snips: {'567': [сниппеты], '639': [сниппеты]}
      - art_snips: {'22': [сниппеты], '34': [сниппеты]}
    """
    found_articles: Set[str] = set()
    orders_snips: Dict[str, List[str]] = {"567": [], "639": []}
    art_snips: Dict[str, List[str]] = {}

    def snippet(s: str, m: re.Match, radius: int = 40) -> str:
        if not WRITE_SNIPPETS:
            return ""
        start = max(0, m.start() - radius)
        end = min(len(s), m.end() + radius)
        return s[start:end].replace("\n", " ")

    for raw in texts:
        s = _normalize_ws(str(raw))

        # Статьи (22/34 + подпункты)
        for m in RE_ART_ANY.finditer(s):
            art_num = m.group(1)
            root = art_num.split('.')[0]  # '22' из '22.1'
            if root in TARGET_ARTICLES:
                found_articles.add(root)
                art_snips.setdefault(root, []).append(snippet(s, m))

        # Приказы 567/639
        for m in RE_ORDERS.finditer(s):
            ord_num = m.group(1)
            orders_snips.setdefault(ord_num, []).append(snippet(s, m))

    # Уберём пустые ключи
    orders_snips = {k: v for k, v in orders_snips.items() if v}
    art_snips = {k: v for k, v in art_snips.items() if v}
    return found_articles, orders_snips, art_snips

# ============== MAIN ==============
def main():
    # Если нужно строго "только 639 и без 567" — поставь True
    STRICT_639_ONLY = False

    files = list_pdf_files(TARGET_DIR, RECURSIVE)
    if not files:
        print(f"В папке нет PDF-файлов: {TARGET_DIR}")
        return

    rows_found_any: List[List[str]] = []   # статья или приказ, или оба
    rows_none: List[List[str]] = []        # ни статьи, ни приказа
    rows_only_639: List[List[str]] = []    # отдельный отчёт: все, где есть 639

    total_files = len(files)
    read_ok = 0

    # счётчики по категориям
    cnt_both = 0
    cnt_orders_only = 0
    cnt_articles_only = 0
    cnt_none = 0
    cnt_with_639 = 0

    for path in files:
        fname = os.path.basename(path)
        purchase = extract_purchase_number_from_filename(fname)

        texts = read_pdf_texts(path)
        if texts:
            read_ok += 1

        found_articles, orders_snips, art_snips = find_matches(texts)

        # агрегаты для CSV
        articles_found = "|".join(sorted(found_articles)) if found_articles else ""
        orders_found   = "|".join(sorted(orders_snips.keys())) if orders_snips else ""

        # сниппеты (по 1–2 на тип)
        if WRITE_SNIPPETS:
            article_snippets = " || ".join(
                f"{k}: " + " | ".join(v[:2]) for k, v in sorted(art_snips.items())
            ) if art_snips else ""
            order_snippets = " || ".join(
                f"{k}: " + " | ".join(v[:2]) for k, v in sorted(orders_snips.items())
            ) if orders_snips else ""
        else:
            article_snippets = ""
            order_snippets = ""

        # статус и распределение (found_any / not_found_any)
        if found_articles and orders_snips:
            status = "orders+article"
            cnt_both += 1
            rows_found_any.append([
                purchase, articles_found, orders_found, status,
                fname, path, article_snippets, order_snippets
            ])
        elif orders_snips:
            status = "orders"
            cnt_orders_only += 1
            rows_found_any.append([
                purchase, articles_found, orders_found, status,
                fname, path, article_snippets, order_snippets
            ])
        elif found_articles:
            status = "article"
            cnt_articles_only += 1
            rows_found_any.append([
                purchase, articles_found, orders_found, status,
                fname, path, article_snippets, order_snippets
            ])
        else:
            status = "missing"
            cnt_none += 1
            rows_none.append([purchase, fname, path])

        # --- ОТЧЁТ ТОЛЬКО ПО 639 ---
        has_639 = "639" in orders_snips
        if STRICT_639_ONLY:
            has_639 = has_639 and ("567" not in orders_snips)
        if has_639:
            cnt_with_639 += 1
            order_639_snippets = " | ".join(orders_snips.get("639", [])[:2]) if WRITE_SNIPPETS else ""
            rows_only_639.append([
                purchase, articles_found, orders_found, status,
                fname, path, order_639_snippets
            ])

    # пишем отчёты
    base = os.path.dirname(os.path.abspath(__file__))
    write_csv(os.path.join(base, "found_any_pdf.csv"),
              rows_found_any,
              header=["purchase_number", "articles_found", "orders_found", "status",
                      "file_name", "file_path", "article_snippets", "order_snippets"])
    write_csv(os.path.join(base, "not_found_any_pdf.csv"),
              rows_none,
              header=["purchase_number", "file_name", "file_path"])
    write_csv(os.path.join(base, "only_639_pdf.csv"),
              rows_only_639,
              header=["purchase_number", "articles_found", "orders_found", "status",
                      "file_name", "file_path", "order_639_snippets"])

    # итоги в консоль
    print(f"Всего PDF: {total_files}")
    print(f"Успешно прочитано (текст извлечён): {read_ok} из {total_files}")
    print(f"Найдено (статья ИЛИ приказ ИЛИ оба): {len(rows_found_any)}")
    print(f"  — только приказ: {cnt_orders_only}")
    print(f"  — только статья: {cnt_articles_only}")
    print(f"  — статья+приказ: {cnt_both}")
    print(f"Ничего не найдено: {cnt_none}")
    print(f"Документов с приказом 639: {cnt_with_639}")
    print("Готово: found_any_pdf.csv, not_found_any_pdf.csv, only_639_pdf.csv")

if __name__ == "__main__":
    main()
