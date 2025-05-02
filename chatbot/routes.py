from flask import Blueprint, render_template, session, redirect, url_for

chatbot_bp = Blueprint("chatbot", __name__, template_folder="templates")

@chatbot_bp.route("/chat")
def chat():
    if "user" not in session:
        return redirect(url_for("auth.login"))
    return render_template("chat.html", user=session["user"])
