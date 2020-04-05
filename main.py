from flask import Flask, render_template, request
import re
import random
import string
import datetime
import nltk
import numpy as np
import requests
import pandas as pd
from stop_words import get_stop_words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.snowball import FrenchStemmer

#------------------------------------------------ Préparation du bot ------------------------------------------------------

file = open('./infos_corona.txt','r',errors = 'ignore', encoding = "utf8")
texte = file.read()

#Découper le texte en phrases
phrases_token = []
phrases_token = nltk.sent_tokenize(texte)

#Supprimer les questions
for p in reversed(range(len(phrases_token))):
    if phrases_token[p][-1] == "?":
        del phrases_token[p]

#Traitement des doublons
phrases_token = list(set(phrases_token)) 

#Il est plus propre de nettoyer le texte pour la création de la matrice TF-IDF
#que le texte d'origine
def nettoyage(texte):
    texte = texte.lower()
    texte = re.sub(r"\ufeff", "", texte)
    texte = re.sub(r"\n", " ", texte)
    texte = re.sub(r"n\.c\.a", "nca", texte)
    texte = re.sub(r"coronavirus covid-19", "covid-19", texte)
    texte = re.sub(r"coronavirus", "covid-19", texte)
    texte = re.sub(f"[{string.punctuation}]", " ", texte)
    texte = re.sub('[éèê]', 'e', texte)
    texte = re.sub('[àâ]', 'a', texte)
    texte = re.sub('[ô]', 'o', texte)
    texte = re.sub('mort(\w){0,3}|deces|deced(\w){1,5}', 'deces', texte)
    texte = re.sub('medec(\w){1,5}|medic{1,5}', 'medical', texte)
    return texte

phrases_nettoyees = []
for i in range(len(phrases_token)):
    phrases_nettoyees.append(nettoyage(phrases_token[i]))

#Prendre en compte les mots par leurs racines
stemmer = FrenchStemmer()

def StemToken(tokens):
    return [stemmer.stem(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def StemNormalize(text):
    return StemToken(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

#Stop-words et entrainement de la matrice TF-IDF
french_stop_words = get_stop_words('french')
TfidfVec = TfidfVectorizer(tokenizer=StemNormalize, stop_words = french_stop_words)
tf_idf_chat = TfidfVec.fit(phrases_nettoyees)

#Renvoyer les réponses en fonction de la similarité cosinus
def reponse(phrase_user):
    chat_reponse = ''
    
    # on a besoin de passer la chaîne de caractère dans une liste :
    phrase_user = [phrase_user]
    
    # on créé la matrice TF-IDF sur le texte
    tf_idf_infos = tf_idf_chat.transform(phrases_nettoyees)
    # On calcule les valuers TF-IDF pour la phrase de l'utilisateur
    tf_idf_user = tf_idf_chat.transform(phrase_user)
    
    # on calcule la similarité entre la question posée par l'utilisateur
    # et l'ensemble des phrases du texte corona
    similarity = cosine_similarity(tf_idf_user, tf_idf_infos).flatten()
    
    # on sort l'index de la phrase étant la plus similaire
    index_max_sim = np.argmax(similarity)
    
    # Si la similarité max est égale à 0 == pas de correspondance trouvée
    if(similarity[index_max_sim] == 0):
        chat_reponse = chat_reponse + "Je ne peux pas t'aider là-dessus, pose-moi une autre question !"
    # Sinon, on sort la phrase correspondant le plus : 
    else:
        chat_reponse = chat_reponse + phrases_token[index_max_sim]
        
    return chat_reponse

#Au revoir
quit_pattern = r"au revoir|quitter|ciao|bye"
quit_response = ["A bientôt !", "Reviens quand tu veux", "Ne m'oublies pas..."]

#Bonjour
hi_pattern = r"salut.*|bonjour.*|yo|coucou.*"

#Nombre de cas
case_pattern = r".*cas en .*?|.*cas au .*?"

#------------------------------------------------ Application Flask ------------------------------------------------------

app = Flask(__name__)
app.config.update(DEBUG=True)

flag = True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def process():
    user_msg_origin = request.form['user_msg']
    user_msg_transf = user_msg_origin.lower()
    user_comment = request.form['comment']
    user_comment = user_comment.lower()
    user_comment = nettoyage(user_comment)

    phrases_token.append(f'Expérience utilisateur : {user_comment}')

    if re.search(quit_pattern, user_msg_transf):
        bot_response = random.choice(quit_response)

    elif re.fullmatch(hi_pattern, user_msg_transf):
        CurrentHour = int(datetime.datetime.now().hour)
        if CurrentHour >= 6 and CurrentHour < 18:
            bot_response = "Bonjour, quelle est ta question sur le Covid-19 ?"
        else:
            bot_response = "Bonsoir, quelle est ta question sur le Covid-19 ?"

    elif re.fullmatch(case_pattern, user_msg_transf):
        user_msg_transf = re.sub(f"[{string.punctuation}]", " ", user_msg_transf)
        # On récupère la ville renseigné par user
        country = user_msg_transf.split()[-1]
        # On fait une requête
        response = requests.get(f'https://coronavirus-19-api.herokuapp.com/countries/{country}')
        rep = response.json()
        cas = rep['cases']
        todaysCase = rep['todayCases']
        dead = rep['deaths']
        todayDead = rep['todayDeaths']
        guerison = rep['recovered']
        posit = rep['active']
        critic =rep['critical']
        
        bot_response = f"Dans les dernières 24h, l'état en {country} est le suivant : {cas} cas recensés depuis le début, {todayDead} nouveaux cas durant les dernières 24h, un total de {dead} morts, {todayDead} morts durant les dernières 24h, {guerison} cas guéris, {posit} personnes positifs et {critic} cas critiques."

    elif user_msg_transf == "" and user_comment != "":
        bot_response = "Merci, nous avons bien pris en compte ton expérience."

    elif user_msg_transf != "" and user_comment != "":
        bot_response = f'Merci, nous avons bien pris en compte ton expérience. Ma réponse à ta question est : {reponse(user_msg_transf)}'

    else:
        bot_response = reponse(user_msg_transf)

    return render_template('index.html', user_msg = user_msg_origin, bot_response = bot_response)


if __name__=='__main__':
    app.run(debug=True)