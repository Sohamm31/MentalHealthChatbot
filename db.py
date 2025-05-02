import mysql.connector

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="mindly"
)

cursor = db.cursor(dictionary=True)
