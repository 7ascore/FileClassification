import os
import shutil
import joblib
import fitz
import re
import pymorphy3

conf = 0.65
directory = os.path.dirname(os.path.abspath(__file__))
modelpath = os.path.join(directory, 'study_model.pkl')
source = os.path.join(directory, '..', 'Unsorted')
morph = pymorphy3.MorphAnalyzer()

def cleantext(text):
    if not text: 
        return ""
    text = text.replace("-\n", "").replace("- ", "")
    words = re.findall(r'[а-яА-Я]{4,}', text.lower())
    lemms = [morph.parse(w)[0].normal_form for w in words]
    return " ".join(lemms)

def main():
    if not os.path.exists(source):
            os.makedirs(source)
    model = joblib.load(modelpath)
    files = []
    for f in os.listdir(source):
        if f.endswith(".pdf"):
            files.append(f)
    for file in files:
        filepath = os.path.join(source, file)
        try:
            doc = fitz.open(filepath)
            try:
                text = ""
                for page in doc[:15]:
                    text += page.get_text() + " "
            finally:
                doc.close()
            cleaned = cleantext(text)
            if not cleaned.strip():
                targetf = 'Empty'
                confidence = 0
                status = "-> Empty"
            else:
                probs = model.predict_proba([cleaned])[0]
                maxprob = -1
                bestind = 0
                for id, p in enumerate(probs):
                    if p > maxprob:
                        maxprob = p
                        bestind = id
                prediction = model.classes_[bestind]
                confidence = maxprob
                if confidence < conf:
                    targetf = 'DoubleCheck'
                    classprobs = []
                    for i in range(len(model.classes_)):
                        classprobs.append((model.classes_[i], probs[i]))
                    classprobs.sort(key = lambda x: x[1])
                    classprobs.reverse()
                    c1, p1 = classprobs[0]
                    c2, p2 = classprobs[1]
                    status = f"-> {targetf}: {c1}={p1:.1f}, {c2:}={p2:.1f}"
                else:
                    targetf = prediction
                    status = f"-> {targetf}"
            targetpath = os.path.join(directory, '..', targetf)
            os.makedirs(targetpath, exist_ok=True)
            shutil.move(filepath, os.path.join(targetpath, file))
            print(f"{file[:40]} {status}")
        except Exception as e:
            print(f"{file}: {e}")

if __name__ == "__main__":
    main()