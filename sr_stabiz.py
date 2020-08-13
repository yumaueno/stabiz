
# -*- coding: utf-8 -*-
#ライブラリーインポート
import requests
from bs4 import BeautifulSoup
import sys
import MeCab
from time import sleep
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
# import sqlite3
from flask import Flask, redirect ,request,render_template,jsonify
# from flask_bootstrap import Bootstrap
import json

#スクレイピングして文書加工
class Scr():
    def __init__(self, urls):
        self.urls=urls
#スクレイピング
    def geturl(self):
        all_text=[]
        for url in self.urls:
            r=requests.get(url)
            c=r.content
            soup=BeautifulSoup(c,"html.parser")
            article1_content=soup.find_all("p")
            temp=[]
            for con in article1_content:
                out=con.text
                temp.append(out)
            text=''.join(temp)
            all_text.append(text)
            sleep(1)
        return all_text

#メカブで形態素解析
def mplg(article):
    word_list = ""
    m=MeCab.Tagger()
    m1=m.parse(article)
    for row in m1.split("\n"):
        word =row.split("\t")[0]#タブ区切りになっている１つ目を取り出す。ここには形態素が格納されている
        if word == "EOS":
            break
        else:
            pos = row.split("\t")[1]#タブ区切りになっている2つ目を取り出す。ここには品詞が格納されている
            slice = pos[:2]
            if slice == "名詞":
                word_list = word_list +" "+ word
    return word_list

#文書類似度計算
class CalCos():
    def __init__(self,word_list):
        self.word=word_list
#tf-idf＆cos類似度で文書類似度算出
    def tfidf(self):
        docs = np.array(self.word)#Numpyの配列に変換する
        #単語を配列ベクトル化して、TF-IDFを計算する
        vecs = TfidfVectorizer(
                    token_pattern=u'(?u)\\b\\w+\\b'#文字列長が 1 の単語を処理対象に含めることを意味します。
                    ).fit_transform(docs)
        vecs = vecs.toarray()
        return vecs

    def cossim(self,v1,v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

#こっから実装
nc = Flask(__name__)

@nc.route("/")
def check():
    return render_template('sr_stabiz.html')

@nc.route('/output', methods=['POST'])
def output():
    #json形式でURLを受け取る
    url1 = request.json['url1']
    url2 = request.json['url2']

    word_list=[]
    url=[url1,url2]
    sc=Scr(url)
    texts=sc.geturl()
    for text in texts:
        word_list.append(mplg(text))

    wl=CalCos(word_list)
    vecs=wl.tfidf()
    match_rate=wl.cossim(vecs[1],vecs[0])
    article = (url[0], url[1], match_rate)
    match_rate=article

    return_data = {"result":round(match_rate[2]*100,1)}
    return jsonify(ResultSet=json.dumps(return_data))

if __name__ == '__main__':
    nc.run(host="127.0.0.1", port=8080)
