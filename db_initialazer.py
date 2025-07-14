import psycopg2
import json
host = "localhost"
database = "OKPD_2"
user = "postgres"
password = "sa"
port = "5432"  


def insert_data_to_db(conn, data):
    cursor = conn.cursor()
    try:

        insert_query = "INSERT INTO Okpd2 (id, title, code, name) VALUES (%s, %s, %s, %s)"
        cursor.executemany(insert_query, data)
        conn.commit()
        print("Данные успешно добавлены в базу.")
    except Exception as e:
        print(f"Ошибка при добавлении данных: {e}")
    finally:
        cursor.close()

def parse_json_and_insert(filename, conn):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
        

        parsed_data = []
        for key, value in data.items():
            id = value['id']
            title = value['title']
            name = value['name']
  
            code = title.split(":")[0].strip()

            record = (id, title, code, name)
            parsed_data.append(record)
        
  
        insert_data_to_db(conn, parsed_data)

try:
    conn = psycopg2.connect(
        host=host,
        database=database,
        user=user,
        password=password,
        port=port
    )
    
    print("Подключение к базе данных успешно!")
    
    json_filename = "okpd2_full.json"
    
    parse_json_and_insert(json_filename, conn)

except Exception as e:
    print(f"Ошибка подключения: {e}")
finally:
    if conn:
        conn.close()
