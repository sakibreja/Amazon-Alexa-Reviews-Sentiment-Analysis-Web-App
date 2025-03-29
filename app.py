import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from flask import Flask, request, render_template

# Load the dataset
df = pd.read_csv(r"C:\Users\A3MAX SOFTWARE TECH\A VS CODE\11. CAPSTONE PROJECT_DEPLOYMENT\SENTIMENT ANALYSIS MAIN\Data\amazon_alexa.tsv", sep='\t')

# Preprocess the data
df.dropna(subset=['verified_reviews'], inplace=True)

def remove_html_tag(text):
    text_clean = re.compile('<*.?>')
    return re.sub(text_clean, ' ', text)

df['verified_reviews'] = df['verified_reviews'].apply(remove_html_tag)

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean(text):
    word = word_tokenize(text)
    word_to = [w.lower() for w in word if w.isalnum()]
    clean_text = [ps.stem(w) for w in word_to if ps.stem(w) not in stop_words]
    return ' '.join(clean_text)

df['verified_reviews'] = df['verified_reviews'].apply(clean)

cv = CountVectorizer(max_features=1500)
x = cv.fit_transform(df['verified_reviews']).toarray()
y = df['feedback']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=40)

# Train the model
sv = SVC()
sv.fit(X_train, y_train)

# Initialize Flask app
app = Flask(__name__)

# Define routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        clean_review = clean(review)
        vectorized_review = cv.transform([clean_review]).toarray()
        prediction = sv.predict(vectorized_review)
        result = 'Positive' if prediction == 1 else 'Negative'
        return render_template('result.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
