import requests
import json
from time import sleep

BASE_URL = "https://zakupki.gov.ru/epz/api/nsi/okpd2/children.html?parentId={}"
DELAY = 1.0
TIMEOUT = 30
MAX_RETRIES = 3
SAVE_INTERVAL = 100  # Сохранять каждые 100 записей

okpd2_data = {}
item_counter = 0
file_counter = 1  # Счетчик файлов

def fetch_children(parent_id, retry=0):
    try:
        response = requests.get(
            BASE_URL.format(parent_id),
            timeout=TIMEOUT,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36'}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        if retry < MAX_RETRIES:
            sleep(DELAY * 2)
            return fetch_children(parent_id, retry + 1)
        print(f"Timeout error for {parent_id}")
        return None
    except Exception as e:
        print(f"Error fetching {parent_id}: {str(e)}")
        return None

def save_data():
    global file_counter
    filename = f"okpd2_part_{file_counter}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(okpd2_data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(okpd2_data)} items to {filename}")
    file_counter += 1
    okpd2_data.clear()  # Очищаем словарь после сохранения

def parse_node(parent_id, parent_code=""):
    global item_counter
    
    children = fetch_children(parent_id)
    if children is None:
        return
    
    for child in children:            
        try:
            code = child['code']
            key = str(child['key'])
            name = child['name']
            title = child.get('title', f"{code}")
            full_code = f"{parent_code}.{code}" if parent_code else code
            
            okpd2_data[full_code] = {
                "id": key,
                "name": name,
                "title": title
            }
            
            item_counter += 1
            
            # Сохраняем каждые SAVE_INTERVAL записей
            if item_counter % SAVE_INTERVAL == 0:
                print(f"[{item_counter}] Saving data...")
                save_data()
                sleep(DELAY)  # Пауза после сохранения
            
            # Рекурсивный вызов если есть дети
            if child.get('isFolder', False) or child.get('hasChildren', False):
                sleep(DELAY)
                parse_node(key, full_code)
                
        except KeyError as e:
            print(f"Missing key in data: {str(e)}")
            continue

if __name__ == "__main__":
    # Все корневые категории
    root_categories = {
        "A": "8873861",
        "B": "8873862",
        "C": "8873863",
        "D": "8873864",
        "E": "8873865",
        "F": "8873866",
        "G": "8873867",
        "H": "8873868",
        "I": "8873869",
        "J": "8873870",
        "K": "8873871",
        "L": "8873872",
        "M": "8873873",
        "N": "8873874",
        "O": "8873875",
        "P": "8873876",
        "Q": "8873877",
        "R": "8873878",
        "S": "8873879",
        "T": "8873880",
        "U": "8873881"
    }

    print("Starting full parsing (no limit)...")
    
    for letter, root_id in root_categories.items():
        print(f"\nProcessing section {letter} (ID: {root_id})")
        parse_node(root_id, letter)
    
    # Сохраняем оставшиеся данные
    if okpd2_data:
        save_data()
    
    print(f"\nFinished! Total items processed: {item_counter}")
    print(f"Saved to {file_counter-1} files (each {SAVE_INTERVAL} items except last)")