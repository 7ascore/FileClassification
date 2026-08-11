import os
import re
import joblib
import pymorphy3
import fitz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

morph = pymorphy3.MorphAnalyzer()

STOPWORDS = ['пусть', 'тогда', 'если', 'теорема', 'доказательство', 'значит', 
    'такой', 'для', 'всё', 'что', 'как', 'быть', 'есть', 'наш', 'этот', 'следует', 'очевидно', 
    'заметим', 'получим', 'равно', 'свойство', 'определение', 'лемма', 'следствие', 'пример', 
    'номер', 'билет', 'являться', 'называться', 'который', 'один', 'нужно', 'иметь', 'найти', 
    'найтись', 'система', 'уравнение', 'решение', 'решить', 'ответ', 'задача', 'вычислить', 
    'докажите', 'равенство', 'вектор', 'пространство', 'множество', 'точка', 'число', 'замечание']

def cleantext(text):
    if not text: 
        return ""
    text = text.replace("-\n", "").replace("- ", "")
    words = re.findall(r'[а-яА-Я]{4,}', text.lower())
    lemms = [morph.parse(w)[0].normal_form for w in words]
    return " ".join(lemms)  

def loaddata(basepath):
    texts = []
    labels = []
    for category in os.listdir(basepath):
        catpath = os.path.join(basepath, category)
        if not os.path.isdir(catpath): 
            continue
        print(f"Загрузка {category}")
        for file in os.listdir(catpath):
            if file.endswith(".pdf"):
                try:
                    doc = fitz.open(os.path.join(catpath, file))
                    try:
                        for page in doc:
                            t = page.get_text()
                            if len(t.strip()) > 150:
                                texts.append(cleantext(t))
                                labels.append(category)
                    finally:
                        doc.close()
                except Exception as e: 
                    print(f"Reading error in {file}: {e}")
    return texts, labels

def train():
    scriptdir = os.path.dirname(os.path.abspath(__file__))
    basepath = os.path.join(scriptdir, '..', 'TrainData')
    X, y = loaddata(basepath)
    if not X: return
    model = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_df=0.85, min_df=2, stop_words=STOPWORDS)),
        ('clf', LogisticRegression(class_weight='balanced', C=1.0, max_iter=2000))])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    model.fit(X_train, y_train)
    print("Отчет:")
    print(classification_report(y_test, model.predict(X_test)))
    model.fit(X, y)
    joblib.dump(model, os.path.join(scriptdir, 'study_model.pkl'))
    print("Готово!")

if __name__ == "__main__":
    train()