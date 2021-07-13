import sqlite3
import hashlib
import datetime
import MySQLdb
from flask import session
from datetime import datetime
import math
import re
from collections import Counter
WORD = re.compile(r"\w+")
def db_connect():
    _conn = MySQLdb.connect(host="localhost", user="root",
                            passwd="root", db="kidney")
    c = _conn.cursor()

    return c, _conn

# -------------------------------Loginact-----------------------------------------------------------------
def admin_loginact(username, password):
    try:
        c, conn = db_connect()
        j = c.execute("select * from admin where username='" +
                      username+"' and password='"+password+"'")
        c.fetchall()
        conn.close()
        return j
    except Exception as e:
        return(str(e))

def user_loginact(username, password):
    try:
        c, conn = db_connect()
        j = c.execute("select * from user where username='" +
                      username+"' and password='"+password+"' and status = 'Activated'")
        data = c.fetchall()
        # for a in data:
        #   session['uname'] = a[0]
       
        c.fetchall()
        conn.close()
        return j
    except Exception as e:
        return(str(e))

def analyst_loginact(username, password):
    try:
        c, conn = db_connect()
        j = c.execute("select * from analyst where username='" +
                      username+"' and password='"+password+"'")
        c.fetchall()
        conn.close()
        return j
    except Exception as e:
        return(str(e))
#-------------------------------------register---------------------------------------------------
def user_upload(row):
    try:
        c, conn = db_connect()
        print("................")
        print(row)
        j = c.execute("insert into upload (bookid,author,book_desc,book_from,book_title,image,price) values ('"+row[0]+ "','"+row[1]+"','"+row[2]+"','"+row[4]+"','"+row[10]+"','"+row[12]+"','"+row[13]+"')")
        conn.commit()
        conn.close()
        print(j)
        return j
    except Exception as e:
        print(e)
        return(str(e))





def user_reg(id,username, password, email, dob, gender, address, mobile):
    try:
        c, conn = db_connect()
        print(id,username, password, email, dob,
              gender, address, mobile)
        j = c.execute("insert into user (id,username,password,email,dob,gender,address,mobile) values ('"+id+"','"+username +
                      "','"+password+"','"+email+"','"+dob+"','"+gender+"','"+address+"','"+mobile+"')")
        conn.commit()
        conn.close()
        print(j)
        return j
    except Exception as e:
        print(e)
        return(str(e))
#--------------------------------------------view----------------------------------------------
def recomend_books():
    c, conn = db_connect()
    c.execute("select * from upload")
    result = c.fetchall()
    conn.close()
    print(result)
    return result







def admin_viewusers():
    c, conn = db_connect()
    c.execute("select username,email,gender,address,status from user")
    result = c.fetchall()
    conn.close()
    print("result")
    return result


















def user_viewrecommend():
    c, conn = db_connect()
    username=  session['username']
    c.execute("select * from recommends where username='"+username+"'")
    result = c.fetchall()
    conn.close()
    print("result")
    return result









def view_book():
    c, conn = db_connect()
    username=  session['username']
    c.execute("select * from upload")
    result = c.fetchall()
    conn.close()
    print(result)
    return result
#-------------------------------------accept and reject-------------------------------------------------
def uviewact(username, email,gender):
    c, conn = db_connect()
    j = c.execute("update user set status='Accepted' where username='"+username+"'")
    conn.commit()
    conn.close()
    return j

def uviewdeact(username, email,gender):
    c, conn = db_connect()
    j = c.execute("update user set status='Rejected' where username='"+username+"'")
    conn.commit()
    conn.close()
    return j




#----------------------------------------Add-----------------------------------------------------------


def add_recomend(author,title,image,price):
    val=rec_del()
    c, conn = db_connect()    
    j = c.execute("insert into recomend (author,title,image,price) values ('" + author + "','" +
                  title + "','" + image + "','" + price +"')")
    conn.commit()
    conn.close()
    return j













# ----------------------------------------------Calculate Cosine Similarity------------------------------------------
def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)

def cotent_similarity(input_book):
    records=recomend_books()
    recommendbooks=[]
    recommendauthor=[]
    recommendimage=[]
    recommendprice=[]
    for bookid,author,book_desc,book_form,book_title,image,price in records:
        text1 = input_book
        text2 = book_desc
        vector1 = text_to_vector(text1)
        vector2 = text_to_vector(text2)
        cosine = get_cosine(vector1, vector2)
        print("Cosine:", cosine)
        if cosine>0.7:           
            recommendauthor.append(author)
            recommendbooks.append(book_title)
            recommendimage.append(image)
            recommendprice.append(price)
    return recommendauthor,recommendbooks,recommendimage,recommendprice
        
if __name__ == "__main__":
    print(db_connect())
