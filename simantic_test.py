import pandas as pd
import psycopg2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SemanticAnalyzer:
    def __init__(self):
        self.connection = None
        # Загружаем предобученную модель для русского языка
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    def connect_to_db(self):
        """Подключение к PostgreSQL"""
        try:
            self.connection = psycopg2.connect(
                host="localhost",
                database="471CSV",
                user="postgres",
                password="sa",
                port="5432",
                client_encoding="UTF-8"
            )
            print("Успешное подключение к базе данных!")
            return True
        except (Exception, psycopg2.Error) as error:
            print("Ошибка при подключении к PostgreSQL:", error)
            return False

    def load_data(self):
        """Загрузка данных из базы"""
        if not self.connect_to_db():
            return None

        query = """
        SELECT registry_number, purchase_name, okpd2_name, okpd2_classification
        FROM csvpurchases 
        WHERE purchase_name IS NOT NULL AND okpd2_name IS NOT NULL
        LIMIT 1000  -- Ограничим для теста (правильный SQL-комментарий)
        """
        try:
            df = pd.read_sql(query, self.connection)
            print(f"Загружено {len(df)} записей")
            return df
        except Exception as e:
            print("Ошибка при загрузке данных:", e)
            return None
        finally:
            if self.connection:
                self.connection.close()

    def calculate_semantic_similarity(self, text1, text2):
        """Вычисление семантической близости в % с помощью sBERT"""
        if not text1 or not text2:
            return 0.0

        # Получаем эмбеддинги для текстов
        embedding1 = self.model.encode(text1, convert_to_tensor=True)
        embedding2 = self.model.encode(text2, convert_to_tensor=True)

        # Вычисляем косинусное сходство
        similarity = cosine_similarity(
            embedding1.reshape(1, -1), 
            embedding2.reshape(1, -1)
        )[0][0]

        # Преобразуем в проценты (0-100)
        return round(similarity * 100, 2)

    def analyze_and_save_results(self, df):
        """Анализ и сохранение результатов"""
        if df is None or df.empty:
            return False

        # Добавляем столбец с семантической схожестью
        df["similarity_percent"] = df.apply(
            lambda row: self.calculate_semantic_similarity(
                row["purchase_name"], 
                row["okpd2_name"]
            ),
            axis=1
        )

        # Сохраняем в новую таблицу
        if not self.connect_to_db():
            return False

        try:
            cursor = self.connection.cursor()
            
            # Создаем таблицу (исправлен синтаксис)
            cursor.execute("""
                DROP TABLE IF EXISTS semantic_analysis_results;
                CREATE TABLE semantic_analysis_results (
                    registry_number TEXT,
                    purchase_name TEXT,
                    okpd2_name TEXT,
                    okpd2_classification TEXT,
                    similarity_percent NUMERIC,
                    PRIMARY KEY (registry_number)
                );
            """)

            # Пакетная вставка с использованием executemany для безопасности
            insert_query = """
                INSERT INTO semantic_analysis_results 
                (registry_number, purchase_name, okpd2_name, okpd2_classification, similarity_percent)
                VALUES (%s, %s, %s, %s, %s)
            """
            
            # Преобразуем данные в список кортежей
            data = [
                (row['registry_number'], row['purchase_name'], 
                row['okpd2_name'], row['okpd2_classification'], 
                row['similarity_percent'])
                for _, row in df.iterrows()
            ]
            
            cursor.executemany(insert_query, data)
            self.connection.commit()
            print(f"Успешно сохранено {len(df)} записей")
            return True
            
        except Exception as e:
            print("Ошибка при сохранении:", e)
            self.connection.rollback()
            return False
        finally:
            if self.connection:
                self.connection.close()

# Пример использования
if __name__ == "__main__":
    analyzer = SemanticAnalyzer()
    data = analyzer.load_data()
    if data is not None:
        analyzer.analyze_and_save_results(data)