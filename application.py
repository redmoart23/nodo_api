from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, request, jsonify, render_template
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from dotenv import load_dotenv
from selenium import webdriver
from newspaper import Article
from functools import wraps
from time import sleep
import pandas as pd
import spacy
import json
import nltk
import os

load_dotenv()

nltk.download('punkt')
application = Flask(__name__)


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)


application.json_encoder = CustomJSONEncoder

AUTH_TOKEN = os.environ['AUTH_TOKEN']

articles_links_completed = []
summary_articles = []
titles_articles = []
images_articles = []
articles_links = []
keywords_list = []

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.binary_location = os.environ["GOOGLE_CHROME_BIN"]
driver = webdriver.Chrome(executable_path=os.environ["CHROMEDRIVER_PATH"], options=chrome_options)


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')

        if token == AUTH_TOKEN:
            return f(*args, **kwargs)

        return jsonify({'error': 'Invalid auth token'}), 401

    return decorated


@application.route('/')
def entry():
    return render_template('index.html')


@application.route('/api/get-articles', methods=['POST'])
@token_required
def index():
    dropdown_value = request.form.get('dropdown')
    return article_extraction(dropdown_value)


def article_extraction(topic):
    for i in range(4):
        driver.get(
            f'https://www.google.com/search?q={topic}&rlz=1C1YTUH_esCO1015CO1015&tbm=nws&sxsrf=APwXEde09es3cdHcYrHezQNYuVi1RcGZvQ:1687629333214&ei=FS6XZIOwDNqYwbkP0p2b-As&start={i}0&sa=N&ved=2ahUKEwjDjqyXvdz_AhVaTDABHdLOBr84HhDy0wN6BAgEEAQ&biw=1366&bih=663&dpr=1')
        sleep(2)
        driver.execute_script('window.scrollTo(0,' + str(1500) + ')')
        articles_raw = driver.find_elements(By.CSS_SELECTOR, 'a.WlydOe')
        articles_links = [articles_raw[i].get_attribute(
            'href') for i in range(len(articles_raw))]
        articles_links_completed.extend(articles_links)

    return article_analysis(articles_links_completed)


def article_analysis(articles_links_completed):
    for url in articles_links_completed:
        try:
            article = Article(url)
            article.download()
            article.parse()
            article.nlp()

            # Load the English model in spaCy
            nlp = spacy.load('es_core_news_sm')

            # Define the document or article text
            document = article.text

            # Process the document with spaCy
            doc = nlp(document)

            # Extract sentences from the document
            sentences = [sent.text for sent in doc.sents]

            # Initialize the TF-IDF vectorizer
            vectorizer = TfidfVectorizer()

            # Compute the TF-IDF matrix
            tfidf_matrix = vectorizer.fit_transform(sentences)

            # Calculate the sentence scores based on TF-IDF values
            sentence_scores = tfidf_matrix.sum(axis=1)

            # Create a dictionary to store the sentence scores
            sentence_scores_dict = {
                sentence: score for sentence, score in zip(sentences, sentence_scores)}

            # Sort the sentences based on scores in descending order
            sorted_sentences = sorted(
                sentence_scores_dict, key=sentence_scores_dict.get, reverse=True)

            # Extract named entities from the document
            named_entities = list(doc.ents)

            # Extract noun phrases from the document
            noun_phrases = list(doc.noun_chunks)

            # Get the main topics or general idea from named entities
            main_topics = list(set(
                [ent.text for ent in named_entities if ent.label_ == 'PERSON' or ent.label_ == 'ORG']))

            # If no main topics from named entities, use noun phrases as general idea
            if not main_topics:
                main_topics = list(
                    set([phrase.text for phrase in noun_phrases]))

            # Set the desired number of sentences for the summary
            summary_length = 3

            # Generate the summary by selecting the top sentences
            summary_sentences = sorted_sentences[:summary_length]

            # Join the summary sentences to form the final summary
            summary = ' '.join(summary_sentences)
            summary_articles.append(summary)

            title = article.title
            titles_articles.append(title)

            images = article.images
            images_articles.append(images)

            keywords_list.append(main_topics)
        except:
            summary = "Not Found"
            summary_articles.append(summary)

            title = "Not Found"
            titles_articles.append(title)

            images = "Not Found"
            images_articles.append(images)

            keyword = "Not Found"
            keywords_list.append(keyword)

    data_to_export = {
        "title": titles_articles,
        "summary": summary_articles,
        "keywords": keywords_list,
        "images": images_articles,
        "link": articles_links_completed
    }

    df = pd.DataFrame(data_to_export)
    json_data = df.to_dict(orient='records')

    return jsonify(json_data)


if __name__ == '__main__':
    application.run(host='127.0.0.1', port=8080)

# if __name__ == '__main__':
#     from waitress import serve
#     serve(app, host='0.0.0.0', port=8080)


# curl -X POST https://api.example.com/users \
#     -H "Content-Type: application/json" \
#     -H "Authorization: <API_KEY>" \
#     -F "dropdown=<VALUE>"
