import requests
import json
from time import sleep
from threading import Thread, Lock
from queue import Queue
import os

# Настройки
BASE_URL = "https://zakupki.gov.ru/epz/api/nsi/okpd2/children.html?parentId={}"
DELAY = 0.3  # Уменьшенная задержка
TIMEOUT = 20
MAX_RETRIES = 2
SAVE_INTERVAL = 1000  # Сохранять каждые 1000 записей
THREADS = 8  # Количество потоков

# Глобальные переменные
data_lock = Lock()
file_lock = Lock()
okpd2_data = {}
item_counter = 0
file_counter = 1
task_queue = Queue()
processed_ids = set()  # Для отслеживания обработанных ID

def fetch_children(parent_id):
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = requests.get(
                BASE_URL.format(parent_id),
                timeout=TIMEOUT,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36',
                    'Accept': 'application/json'
                }
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if attempt == MAX_RETRIES:
                print(f"Error fetching {parent_id}: {str(e)}")
                return None
            sleep(DELAY * (attempt + 1))

def save_data():
    global file_counter
    filename = f"okpd2_part_{file_counter}.json"
    
    with file_lock:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(okpd2_data.copy(), f, ensure_ascii=False, indent=2)
        print(f"Saved {len(okpd2_data)} items to {filename}")
        file_counter += 1
        okpd2_data.clear()

def worker():
    while True:
        task = task_queue.get()
        if task is None:  # Сигнал завершения
            break
            
        parent_id, parent_code = task
        children = fetch_children(parent_id)
        
        if not children:
            task_queue.task_done()
            continue
            
        for child in children:
            try:
                code = child['code']
                key = str(child['key'])
                name = child['name']
                title = child.get('title', f"{code}: {name}")
                full_code = f"{parent_code}.{code}" if parent_code else code
                
                with data_lock:
                    if full_code not in okpd2_data:
                        okpd2_data[full_code] = {
                            "id": key,
                            "name": name,
                            "title": title
                        }
                        
                        global item_counter
                        item_counter += 1
                        
                        if item_counter % SAVE_INTERVAL == 0:
                            save_data()
                
                # Добавляем дочерние элементы в очередь
                if (child.get('isFolder', False) or child.get('hasChildren', False)) and key not in processed_ids:
                    with data_lock:
                        processed_ids.add(key)
                    task_queue.put((key, full_code))
                    
            except KeyError as e:
                print(f"Missing key in data: {str(e)}")
                continue
                
        task_queue.task_done()
        sleep(DELAY)

def main():
    # Инициализация корневых категорий
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

    print(f"Starting multi-threaded parser ({THREADS} threads)...")
    
    # Создаем и запускаем потоки
    threads = []
    for _ in range(THREADS):
        t = Thread(target=worker)
        t.start()
        threads.append(t)
    
    # Добавляем начальные задачи в очередь
    for letter, root_id in root_categories.items():
        task_queue.put((root_id, letter))
        processed_ids.add(root_id)
    
    # Ожидаем завершения всех задач
    task_queue.join()
    
    # Останавливаем потоки
    for _ in range(THREADS):
        task_queue.put(None)
    for t in threads:
        t.join()
    
    # Сохраняем оставшиеся данные
    if okpd2_data:
        save_data()
    
    print(f"\nFinished! Total items processed: {item_counter}")
    print(f"Saved to {file_counter-1} files")

if __name__ == "__main__":
    main()