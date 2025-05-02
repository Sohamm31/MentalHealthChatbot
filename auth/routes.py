from flask import Blueprint, render_template, request, redirect, url_for, session
from flask_bcrypt import Bcrypt
from db import db, cursor

auth_bp = Blueprint("auth", __name__, template_folder="templates")
bcrypt = Bcrypt()

@auth_bp.record_once
def init_bcrypt(setup_state):
    bcrypt.init_app(setup_state.app)

@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
        if user and bcrypt.check_password_hash(user["password"], password):
            session["user"] = user["email"]
            return redirect(url_for("chatbot.chat"))
        else:
            return "Invalid email or password"
    return render_template("login.html")

@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        hashed_pw = bcrypt.generate_password_hash(password).decode("utf-8")
        try:
            cursor.execute("INSERT INTO users (email, password) VALUES (%s, %s)", (email, hashed_pw))
            db.commit()
            return redirect(url_for("auth.login"))
        except:
            return "User already exists"
    return render_template("register.html")

@auth_bp.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("auth.login"))
