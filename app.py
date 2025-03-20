from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import TfidfTransformer

app = Flask(__name__)

with open("stopwords.txt","r") as file:
    stopwords=file.read().splitlines()

vectorizer=TfidfTransformer(stop_words=stopwords, lowercase=True, vocabulary=pickle.load(open("tfidvectorizer.pkl","rb")))
model=pickle.load(open("LinearSVCTuned.pkl","rb"))

@app.route('/',methods=['GET','POST'])

def index():
    prediction=None
    if request.method=='POST':
        user_input=request.form['text']
        transformed_input=vectorizer.fit_transform([user_input])
        prediction=model.predict(transformed_input)[0]

    return render_template("index.html",prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)