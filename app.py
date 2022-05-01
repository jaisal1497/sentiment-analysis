import sqlite3 as sql
from sklearn.feature_extraction.text import CountVectorizer
import bs4 as bs
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

#change the model name and figure it out
model = pickle.load(open('model.pkl', 'rb'))
vectorizer=pickle.load(open('vectorizer.pkl','rb'))
conn = None
@app.route("/")
def home():
    global conn
    if conn is None:
        conn = sql.connect('database33.db')
        print ("Opened database successfully")

        conn.execute('CREATE TABLE IF NOT EXISTS reviews (movie_review TEXT PRIMARY KEY, result TEXT)')
        print ("Table created successfully")
        conn.close()
    return render_template('index.html')


@app.route("/about")
def about():
    return render_template('about.html')


@app.route('/results', methods=['GET','POST'])
def results():
    if request.method == 'POST':
        try:
            nm = request.form['movierev']
            #sample = np.array([nm]).reshape(1, -1).tolist()
            #prediction = model.predict(sample)
            #pred = ['setosa', 'versicolor', 'virginica'][prediction[0]]
            nm_copy=nm
            
            nm=review_cleaner(nm)
            clean_list=[]
            clean_list.append(nm)
            test_bag = vectorizer.transform(clean_list).toarray()
            test_predictions = model.predict(test_bag)
            pred="POSITIVE"
            if(test_predictions[0]==1):
                pred="POSITIVE"
            else:
                pred="NEGATIVE"
            with sql.connect("database33.db") as con:
                cur = con.cursor()
            
                cur.execute("INSERT INTO reviews (movie_review,result) VALUES (?,?)",(nm_copy,pred) ) # ? and tuple for placeholders
            
                con.commit()
                msg = "Record successfully added"
        except:
            con.rollback()
            msg = "error in insert operation"
      
        finally:
            #return render_template("result.html",msg = msg)
            con.close()

        
        return render_template("results.html", value=pred)
    else:
        return render_template('results.html', value="N/A")
		
        
@app.route('/data')
def list_all():
    con = sql.connect("database33.db")
    con.row_factory = sql.Row
    
    cur = con.cursor()
    cur.execute("select * from reviews")
    
    rows = cur.fetchall() # returns list of dictionaries
    return render_template("data.html",rows = rows)

def review_cleaner(review):
    '''
    Clean and preprocess a review.
    
    1. Remove HTML tags
    2. Use regex to remove all special characters (only keep letters)
    3. Make strings to lower case and tokenize / word split reviews
    4. Remove English stopwords
    5. Rejoin to one string
    '''
    
    #1. Remove HTML tags
    review = bs.BeautifulSoup(review).text
    
    #2. Use regex to find emoticons
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', review)
    
    #3. Remove punctuation
    review = re.sub("[^a-zA-Z]", " ",review)
    
    #4. Tokenize into words (all lower case)
    review = review.lower().split()
    
    #5. Remove stopwords
    eng_stopwords = set(stopwords.words("english"))
    review = [w for w in review if not w in eng_stopwords]
    
    #6. Join the review to one sentence
    review = ' '.join(review+emoticons)
    # add emoticons to the end

    return(review)
