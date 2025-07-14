import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger
import nltk
from nltk.corpus import stopwords
import re
from string import punctuation
import psycopg2

nltk.download('stopwords')
nltk.download('punkt')

class SemanticGrouper:
    def __init__(self):
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)
        self.russian_stopwords = stopwords.words('russian')
        self.vectorizer = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.85,
            sublinear_tf=True
        )
        self.connection = None

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
        from natasha import Doc
        doc = Doc(word)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)
        
        for token in doc.tokens:
            token.lemmatize(self.morph_vocab)
        
        return doc.tokens[0].lemma if doc.tokens else word

    def preprocess_text(self, text):
        """Предобработка текста"""
        if not isinstance(text, str):
            return ""
            
        text = text.lower()
        text = re.sub(f'[{punctuation}»«–…№]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        
        tokens = text.split()
        tokens = [self.lemmatize_word(token) for token in tokens if token not in self.russian_stopwords]
        tokens = [token for token in tokens if len(token) > 2]
        
        return ' '.join(tokens).strip()

    def group_purchases(self, df, threshold=60.0):
        """
        Распределение purchase_name по группам okpd2_name
        :param df: DataFrame с колонками purchase_name и okpd2_name
        :param threshold: минимальный процент схожести для попадания в группу (по умолчанию 60%)
        :return: словарь {okpd2_name: [список соответствующих purchase_name]}
        """
        if df is None or df.empty:
            print("Нет данных для группировки")
            return {}, []

        # Предобработка текстов
        print("Начата предобработка текстов...")
        df['purchase_processed'] = df['purchase_name'].apply(self.preprocess_text)
        df['okpd2_processed'] = df['okpd2_name'].apply(self.preprocess_text)
        
        # Уникальные группы и закупки
        unique_okpd2 = df[['okpd2_name', 'okpd2_processed']].drop_duplicates().reset_index(drop=True)
        unique_purchases = df[['purchase_name', 'purchase_processed']].drop_duplicates().reset_index(drop=True)
        
        # Векторизация
        print("Векторизация текстов...")
        all_texts = pd.concat([unique_okpd2['okpd2_processed'], unique_purchases['purchase_processed']]).tolist()
        self.vectorizer.fit(all_texts)
        
        okpd2_vectors = self.vectorizer.transform(unique_okpd2['okpd2_processed'])
        purchase_vectors = self.vectorizer.transform(unique_purchases['purchase_processed'])
        
        # Расчет схожести
        print("Расчет семантического сходства...")
        similarity_matrix = cosine_similarity(purchase_vectors, okpd2_vectors)
        
        # Создание словаря для группировки
        groups = {okpd2: [] for okpd2 in unique_okpd2['okpd2_name']}
        ungrouped = []
        
        print(f"Группировка {len(unique_purchases)} закупок...")
        for i in range(len(unique_purchases)):
            purchase_row = unique_purchases.iloc[i]
            max_sim_idx = similarity_matrix[i].argmax()
            max_sim = similarity_matrix[i][max_sim_idx] * 100
            
            if max_sim >= threshold:
                okpd2_name = unique_okpd2.iloc[max_sim_idx]['okpd2_name']
                classification = df[df['okpd2_name'] == okpd2_name]['okpd2_classification']
                classification = classification.iloc[0] if not classification.empty else "Не указано"
                
                groups[okpd2_name].append({
                    'purchase_name': purchase_row['purchase_name'],
                    'similarity_percent': max_sim,
                    'okpd2_classification': classification
                })
            else:
                ungrouped.append({
                    'purchase_name': purchase_row['purchase_name'],
                    'max_similarity_percent': max_sim,
                    'best_match_okpd2': unique_okpd2.iloc[max_sim_idx]['okpd2_name']
                })
        
        return groups, ungrouped

    def save_grouping_results(self, groups, ungrouped, filename='grouping_results.txt'):
        """Сохранение результатов группировки в файл"""
        with open(filename, 'w', encoding='utf-8') as f:
            # Группированные закупки
            f.write("=== ГРУППИРОВКА ПО ОКПД2 ===\n\n")
            for okpd2_name, purchases in groups.items():
                if purchases:  # Только непустые группы
                    f.write(f"Группа: {okpd2_name}\n")
                    f.write(f"Классификация: {purchases[0]['okpd2_classification']}\n")
                    f.write(f"Количество закупок: {len(purchases)}\n")
                    f.write("Закупки в группе (отсортированы по убыванию схожести):\n")
                    
                    for purchase in sorted(purchases, key=lambda x: x['similarity_percent'], reverse=True):
                        f.write(f"  - {purchase['purchase_name']} ({purchase['similarity_percent']:.1f}%)\n")
                    
                    f.write("\n" + "="*80 + "\n\n")
            
            # Негруппированные закупки
            f.write("\n\n=== НЕГРУППИРОВАННЫЕ ЗАКУПКИ ===\n\n")
            f.write(f"Количество: {len(ungrouped)}\n\n")
            for purchase in sorted(ungrouped, key=lambda x: x['max_similarity_percent'], reverse=True):
                f.write(f"  - {purchase['purchase_name']}\n")
                f.write(f"    Лучшее соответствие: {purchase['best_match_okpd2']} ({purchase['max_similarity_percent']:.1f}%)\n\n")

        print(f"Результаты группировки сохранены в файл {filename}")

    def process(self, threshold=60.0):
        """Основной метод обработки"""
        df = self.load_data()
        if df is None or df.empty:
            print("Нет данных для обработки")
            return False
            
        groups, ungrouped = self.group_purchases(df, threshold)
        self.save_grouping_results(groups, ungrouped)
        return True


# Пример использования
if __name__ == "__main__":
    grouper = SemanticGrouper()
    if grouper.process(threshold=60.0):
        print("Группировка выполнена успешно! Проверьте файл grouping_results.txt")
    else:
        print("Произошла ошибка при группировке")