import os
import MySQLdb
import csv
from flask import Flask, session, url_for, redirect, render_template, request, abort, flash
from database import*
from database import*
from database import*
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import pickle
from sklearn.preprocessing import LabelEncoder
app = Flask(__name__)
app.secret_key = os.urandom(24)


@app.route("/")
def FUN_root():
    return render_template("index.html")

@app.route("/index.html")
def index():
    return render_template("index.html")

@app.route("/admin.html")
def admin():
    return render_template("admin.html")  

@app.route("/train.html")
def train():
    return render_template("train.html") 



@app.route("/userhome.html")
def userhome():
    return render_template("userhome.html")

   

@app.route("/adminhome.html")
def adminhome():
    return render_template("adminhome.html")    

@app.route("/register.html")
def register():
    return render_template("register.html") 

@app.route("/addcategory.html")
def addcategory():
    return render_template("addcategory.html")
    
@app.route("/addproducts.html")
def addproducts():
    a = admin_cate()
    return render_template("addproducts.html",a=a)
















#--------------------------------------------------Login----------------------------------------------------
@app.route("/adminlogact", methods=['GET', 'POST'])
def adminlogact():
    if request.method == 'POST':
        status = admin_loginact(request.form['username'], request.form['password'])
        print(status)
        if status == 1:
            session['username'] = request.form['username']
            return render_template("addcategory.html", m1="sucess")
        else:
            return render_template("admin.html", m1="Login Failed")




#-----------------------------------------------Register------------------------------------------------



@app.route("/predict", methods = ['GET','POST'])
def predictact():
   if request.method == 'POST':    
    id="0"
    age=int(request.form['age'])
    bp=int(request.form['bp'])
    sg=float(request.form['sg'])
    albumin=int(request.form['albumin'])
    sugar=int(request.form['sugar'])
    bpr=float(request.form['bpr'])
    bu=int(request.form['bu'])
    seerum=float(request.form['seerum'])
    sodium=float(request.form['sodium'])
    potaseum=float(request.form['potaseum'])
    hemaglobin=float(request.form['hemaglobin'])
    pcv=int(request.form['pcv'])
    wbc=int(request.form['wbc'])
    hypertension=str(request.form['hypertension'])
    diabetes=str(request.form['diabetes'])
    cad=str(request.form['cad'])
    apetite=str(request.form['apetite'])
    pe=str(request.form['pe'])
    vector = np.vectorize(np.float)
    check = np.array([age, bp, sg, albumin, sugar, bpr, bu,seerum, sodium,potaseum,hemaglobin,pcv,wbc,hypertension,diabetes,cad,apetite,pe]).reshape(1, -1)
    #check = np.array([a,b,c,d,e,g,h,i,j,k,l,m,n,o,p,q,r,s]).reshape(1, -1)
    
    Labe=LabelEncoder()
    check[:,13]=Labe.fit_transform(check[:,13])
    check[:,14]=Labe.fit_transform(check[:,14])
    check[:,15]=Labe.fit_transform(check[:,15])
    check[:,16]=Labe.fit_transform(check[:,16])
    check[:,17]=Labe.fit_transform(check[:,17])
    model_path = os.path.join(os.path.dirname(__file__), 'dataset/mv.sav')
    check = vector(check)
    clf = joblib.load(model_path)
    B_pred = clf.predict(check[[0]])
    if B_pred == 1:
        result="cronical kidney disease detected"
        print("cronical kidney disease detected")
    if B_pred == 0:
        result="No disease detected"
        print("No disease detected")
    
    return render_template('result.html',data=result)
#-------------------------------------------View-------------------------------------------------------
 



@app.route("/viewd", methods = ['GET','POST'])
def viewd():
    global row
    data=pd.read_csv("C:\\Users\\Mrida\\Documents\\Python\\cancer\\chronic_full_cleaned.csv")
    peek=data.head(100)
    return render_template("viewusers.html",tables=[peek.to_html(classes='data')])

















#-------------------------------------------activate---------------------------------------------------





 















if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
