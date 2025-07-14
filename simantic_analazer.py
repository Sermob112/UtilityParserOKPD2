import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import psycopg2
import re
from string import punctuation
from nltk.corpus import stopwords
import nltk
from natasha import (  # Измененный импорт
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger
)

import matplotlib.pyplot as plt
import seaborn as sns
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)

# Скачиваем необходимые ресурсы для обработки русского языка
nltk.download('stopwords')
nltk.download('punkt')

class SemanticAnalyzer:
    def __init__(self):
        self.connection = None
        self.russian_stopwords = stopwords.words('russian')
        
    def connect_to_db(self):
        """Установка соединения с PostgreSQL"""
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

    def lemmatize_word(self, word):
        """Лемматизация слова с использованием Natasha"""
        # Создаем искусственный документ для обработки
        from natasha import Doc
        doc = Doc(word)
        doc.segment(segmenter)
        doc.tag_morph(morph_tagger)
        
        for token in doc.tokens:
            token.lemmatize(morph_vocab)
        
        return doc.tokens[0].lemma if doc.tokens else word
    def preprocess_text(self, text):
        """Улучшенная предобработка русского текста"""
        if not isinstance(text, str):
            return ""
            
        # Приведение к нижнему регистру
        text = text.lower()
        
        # Удаление пунктуации
        text = re.sub(f'[{punctuation}»«–…№]', ' ', text)
        
        # Удаление цифр
        text = re.sub(r'\d+', ' ', text)
        
        # Удаление стоп-слов
        tokens = text.split()
        tokens = [self.lemmatize_word(token) for token in tokens if token not in self.russian_stopwords]
        
        # Удаление коротких слов (менее 3 символов)
        tokens = [token for token in tokens if len(token) > 2]
        
        return ' '.join(tokens).strip()
    
    def calculate_similarity(self, df):
        """Вычисление семантического сходства с русской обработкой"""
        df['purchase_processed'] = df['purchase_name'].apply(self.preprocess_text)
        df['okpd2_processed'] = df['okpd2_name'].apply(self.preprocess_text)
        
        # Исправленный расчет Jaccard similarity
        df['jaccard_sim'] = df.apply(
            lambda x: len(set(x['purchase_processed'].split()) & set(x['okpd2_processed'].split())) / 
                      len(set(x['purchase_processed'].split()) | set(x['okpd2_processed'].split())) 
                      if len(set(x['purchase_processed'].split()) | set(x['okpd2_processed'].split())) > 0 
                      else 0,
            axis=1
        )
        
        # Создаем TF-IDF вектор с русской обработкой
        vectorizer = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.85,
            sublinear_tf=True
        )
                
        # Векторизуем оба текста вместе для согласованного словаря
        all_texts = pd.concat([df['purchase_processed'], df['okpd2_processed']]).tolist()
        vectorizer.fit(all_texts)
        
        # Векторизуем отдельно
        purchase_vectors = vectorizer.transform(df['purchase_processed'])
        okpd2_vectors = vectorizer.transform(df['okpd2_processed'])
        
        # Вычисляем cosine similarity
        cosine_sim = cosine_similarity(purchase_vectors, okpd2_vectors)
        df['cosine_sim'] = np.diag(cosine_sim)
        
        # Комбинируем метрики
        df['similarity_percent'] = (0.7 * df['cosine_sim'] + 0.3 * df['jaccard_sim']) * 100
        
        return df[['registry_number', 'purchase_name', 'okpd2_name', 'okpd2_classification', 'similarity_percent']]
    
    def create_results_table(self, df):
        """Создание новой таблицы с результатами"""
        if not self.connect_to_db():
            return False
            
        try:
            cursor = self.connection.cursor()
            
            # Создаем новую таблицу с дополнительным полем okpd2_classification
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
            
            # Пакетная вставка данных
            data = [tuple(x) for x in df.to_records(index=False)]
            args = ','.join(cursor.mogrify("(%s,%s,%s,%s,%s)", row).decode('utf-8') for row in data)
            cursor.execute(f"INSERT INTO semantic_analysis_results VALUES {args}")
            
            self.connection.commit()
            print(f"Создана новая таблица с {len(df)} записями")
            return True
        except Exception as e:
            print("Ошибка при создании таблицы:", e)
            self.connection.rollback()
            return False
        finally:
            if self.connection:
                self.connection.close()
    

    def analyze(self):
        """Основной метод анализа с визуализацией"""
        df = self.load_data()
        if df is None or df.empty:
            print("Нет данных для анализа")
            return False
            
        result_df = self.calculate_similarity(df)
        
        # Вывод примеров для проверки
        print("\nПримеры результатов:")
        print(result_df.head())
        
        # Визуализация результатов
        self.plot_results(result_df)
        
        return self.create_results_table(result_df)
    
    def plot_results(self, df):
        """Создание графиков и сохранение текстовой информации"""
        plt.figure(figsize=(15, 10))
        
        # 1. Гистограмма распределения similarity_percent
        plt.subplot(2, 2, 1)
        sns.histplot(df['similarity_percent'], bins=30, kde=True)
        plt.title('Распределение процента схожести')
        plt.xlabel('Процент схожести')
        plt.ylabel('Количество записей')
        
        # 2. Boxplot для similarity_percent
        plt.subplot(2, 2, 2)
        sns.boxplot(y=df['similarity_percent'])
        plt.title('Распределение процента схожести (Boxplot)')
        plt.ylabel('Процент схожести')
        
        # 3. Матрица распределения по категориям схожести
        plt.subplot(2, 2, 3)
        bins = [0, 40, 60, 80, 100]
        labels = ['0-40%: Разные понятия', '40-60%: Умеренная связь', 
                '60-80%: Близкие понятия', '80-100%: Полное соответствие']
        
        # Создаем категории
        df['category'] = pd.cut(df['similarity_percent'], bins=bins, labels=labels, include_lowest=True)
        category_counts = df['category'].value_counts().sort_index()
        
        # Визуализация
        sns.barplot(x=category_counts.values, y=category_counts.index, orient='h')
        plt.title('Распределение по категориям схожести')
        plt.xlabel('Количество записей')
        plt.ylabel('Категория схожести')
        
        # 4. Пустое место для текстовой информации (или можно добавить другой график)
        plt.subplot(2, 2, 4)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('semantic_analysis_results.png')
        plt.show()
        
        # Сохраняем топ-15 результатов в текстовый файл
        self.save_top_results(df)

    def save_top_results(self, df):
        """Сохранение топ-15 результатов в текстовый файл"""
        with open('semantic_analysis_top_results.txt', 'w', encoding='utf-8') as f:
            # Топ-15 самых высоких значений
            f.write("=== ТОП-15 САМЫХ ВЫСОКИХ ЗНАЧЕНИЙ СХОЖЕСТИ ===\n\n")
            top_15 = df.nlargest(15, 'similarity_percent')
            for idx, row in top_15.iterrows():
                f.write(f"{row['similarity_percent']:.2f}%: {row['purchase_name']}\n")
                f.write(f"ОКПД2: {row['okpd2_name']}\n")
                f.write(f"Классификация: {row['okpd2_classification']}\n")
                f.write(f"Номер реестра: {row['registry_number']}\n")
                f.write("-"*80 + "\n")
            
            # Топ-15 самых низких значений
            f.write("\n\n=== ТОП-15 САМЫХ НИЗКИХ ЗНАЧЕНИЙ СХОЖЕСТИ ===\n\n")
            bottom_15 = df.nsmallest(15, 'similarity_percent')
            for idx, row in bottom_15.iterrows():
                f.write(f"{row['similarity_percent']:.2f}%: {row['purchase_name']}\n")
                f.write(f"ОКПД2: {row['okpd2_name']}\n")
                f.write(f"Классификация: {row['okpd2_classification']}\n")
                f.write(f"Номер реестра: {row['registry_number']}\n")
                f.write("-"*80 + "\n")
        
        print("Топ-15 результатов сохранены в файл 'semantic_analysis_top_results.txt'")
    # Использование
if __name__ == "__main__":
    analyzer = SemanticAnalyzer()
    if analyzer.analyze():
        print("\nСемантический анализ успешно завершен! Результаты сохранены в таблице semantic_analysis_results")
        print("Структура таблицы: registry_number, purchase_name, okpd2_name, okpd2_classification, similarity_percent")
    else:
        print("\nПроизошла ошибка при выполнении анализа")