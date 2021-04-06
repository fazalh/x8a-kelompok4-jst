from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
import gunicorn
import pickle

# initiate flask
app = Flask(__name__)

# load model
model = pickle.load(open("model-genre.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
genres = ['Drama', 'Comedy', 'Thriller', 'Romance', 'Action',
        'Horror', 'Crime', 'Documentary', 'Adventure',
        'Science Fiction', 'Family', 'Mystery', 'Fantasy',
        'Animation', 'Foreign', 'Music', 'History', 'War',
        'Western', 'TV Movie']

# routing
@app.route("/")
def root():
    return render_template("home.html")

@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/prediction", methods=["GET", "POST"])
def predict():
    if request.method == "POST":

        review = request.form['review']

        pred_vec = vectorizer.transform([review])
        predict = model.predict(pred_vec)
        preds = predict[0].tolist()
        dictgen = dict(zip(genres, preds))
        res = []
        def get_key(val):
            for key, value in dictgen.items():
                if val == value:
                    res.append(key)
                
        get_key(1)
        #pred_vec_res = final_genres.inverse_transform(predict)
        
        listToStr = ', '.join([str(i) for i in res])

        result = 'Predicted Genre is {}'.format(listToStr) 
        return render_template('result.html', results = result)
        
    else : 
        return render_template('prediction.html')

if __name__ == '__main__':
    app.run(debug=True)