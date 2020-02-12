# coding: utf-8

import numpy as np
import fasttext
import MeCab
from flask import Flask, jsonify, render_template, request, redirect, url_for, send_from_directory, session
from werkzeug import secure_filename
from gensim.test.utils import datapath
from gensim.models.fasttext import load_facebook_model,load_facebook_vectors
from gensim.models._fasttext_bin import load   

import os, io, time, re

MODEL_FILE_PATH = 'model_corona_002auto.ftz'

def text_wakati(text):
    # リクエストで送られてきたテキストデータを分かち書きする前処理
    wakati = MeCab.Tagger("-Owakati")
    tokenized_text = wakati.parse(text)
    
    return tokenized_text


#　flask appを使えるようにするためのインスんタンス化
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # 日本語文字ばけ防止

# <--プログラム-->
# @app.route('/') # /　というURLにアクセスされた時の処理を記述
# def index():
#     return 'Hello World!'

@app.route('/', methods=["GET", "POST"])
def post_json():
    if  request.method == "GET":
        return render_template('test_ajax.html')
    
    elif request.method == "POST":
        response = request.get_json()
        target_text = response["text_to_get_predicted"] 
        target_text = text_wakati(target_text)  


        # load model
        model = fasttext.load_model(MODEL_FILE_PATH)

        # model で推論 result はtuple型
        result = model.predict(target_text.replace('\n',''))
          # result の例 ('(__label___,*******),',[0.743284732])      

        category = str(result[0])

        # 記号の除去
        category = re.sub('__label__','',category)

        category = re.sub("[()',]" , "",category)

        probability = str(result[1])

        # remove []
        probability  = re.sub("[\[\]]" , "", probability )

        
        json = [
          {"Category": category},
          {"Confident": probability}
          ]

        return jsonify(json)



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))