#app.py file
from flask import Flask, render_template, request, jsonify, g
import json

import pickle
import sqlite3
import datetime
import re
import os
import urllib.request
import urllib.parse
import nltk
import random

try:
    nltk.data.find('tokenizers/punkt_tab')
except (LookupError, OSError):
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except (LookupError, OSError):
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)

app = Flask(__name__)
DATABASE = 'database.db'

# DATABASE SETUP

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context():
        db = get_db()
        cursor = db.cursor()
        # Spam Logs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS spam_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT, prediction TEXT, confidence INTEGER,
                suspicious_flags TEXT, user_feedback TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        try:
            cursor.execute("ALTER TABLE spam_logs ADD COLUMN suspicious_flags TEXT")
        except:
            pass
        # Task Manager
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS task_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT, description TEXT, priority TEXT,
                status TEXT DEFAULT 'pending',
                subtasks TEXT, estimated_time TEXT,
                deadline TEXT, task_link TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        # Email Logs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS email_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                purpose TEXT, generated_email TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        db.commit()

init_db()

# ML MODEL LOADING

spam_model = None
vectorizer = None

try:
    if os.path.exists("spam_model.pkl") and os.path.exists("vectorizer.pkl"):
        with open("spam_model.pkl", "rb") as f:
            spam_model = pickle.load(f)
        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
except Exception as e:
    print(f"Error loading models: {e}")

# APP ROUTES (PAGES)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/email-generator")
def email_generator():
    return render_template("email_generator.html")

@app.route("/spam-classifier")
def spam_classifier():
    return render_template("spam_classifier.html")

@app.route("/task-manager")
def task_manager():
    return render_template("task_manager.html")


# API: SMART EMAIL GENERATOR

def generate_dynamic_email(purpose, tone, name):
    name = name if name else "[Your Name]"
    
    # Construct a highly specific prompt for the LLM
    prompt = (
        f"Write a {tone.lower()} email. "
        f"The main topic/purpose is strictly: '{purpose}'. "
        f"The sender's name is '{name}'. "
        "Do not include any conversational filler (e.g. 'Sure, here is your email:'). "
        "Start directly with 'Subject: ' on the very first line. Write the full body and sign-off."
    )
    
    encoded_prompt = urllib.parse.quote(prompt)
    url = f"https://text.pollinations.ai/{encoded_prompt}"
    
    try:
        # Request a genuine generative AI completion from Pollinations.ai (Free unauthenticated endpoint)
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (AI Smart Office)'})
        with urllib.request.urlopen(req, timeout=15) as response:
            email_text = response.read().decode('utf-8').strip()
            
            # Formatting safety fallback in case the AI missed the subject line
            if not email_text.startswith("Subject:"):
                email_text = f"Subject: Update on {purpose[:30]}\n\n" + email_text
                
            return email_text
    except Exception as e:
        print(f"Pollinations AI Error: {e}")
        # Absolute safety fallback
        return f"Subject: Regarding: {purpose[:40]}...\n\nHello,\n\nI am writing to you regarding the following matter: {purpose}.\n\nPlease let me know your thoughts.\n\nBest,\n{name}"

@app.route("/api/generate-email", methods=["POST"])
def api_generate_email():
    data = request.get_json()
    purpose = data.get("input", "").strip()
    tone = data.get("tone", "Professional").strip()
    sender_name = data.get("name", "").strip()

    if len(purpose) < 5:
        return jsonify({"success": False, "error": "Please provide a more detailed context for the email."})

    try:
        email_content = generate_dynamic_email(purpose, tone, sender_name)
        db = get_db()
        cursor = db.cursor()
        cursor.execute("INSERT INTO email_logs (purpose, generated_email) VALUES (?, ?)", (purpose, email_content))
        db.commit()
        return jsonify({"success": True, "email": email_content})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# API: SPAM & PRIORITY CLASSIFIER

@app.route("/api/classify-spam", methods=["POST"])
def api_classify_spam():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"success": False, "error": "No text provided"})

    try:
        if vectorizer is None or spam_model is None:
            return jsonify({"success": False, "error": "Machine Learning models not found. Please run train_model.py first."})

        # 1. Transform input text
        X_vec = vectorizer.transform([text])
        
        # 2. Predict
        prediction = spam_model.predict(X_vec)[0] # Will output 'Spam', 'Normal', or 'Important'
        probabilities = spam_model.predict_proba(X_vec)[0]
        classes = spam_model.classes_
        
        # Create dict of proba by class
        prob_dict = {classes[i]: int(probabilities[i] * 100) for i in range(len(classes))}
        conf = prob_dict[prediction]

        # 3. Explainability: Extract highest TF-IDF words driving this decision
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = X_vec.toarray()[0]
        # Get indices of top 3 TF-IDF scores
        top_indices = tfidf_scores.argsort()[-3:][::-1]
        
        suspicious_flags = []
        for i in top_indices:
            if tfidf_scores[i] > 0:
                suspicious_flags.append(feature_names[i])
        
        # 4. Save to DB
        db = get_db()
        cursor = db.cursor()
        cursor.execute("""
            INSERT INTO spam_logs (text, prediction, confidence, suspicious_flags)
            VALUES (?, ?, ?, ?)
        """, (text, prediction, conf, ", ".join(suspicious_flags)))
        db.commit()

        return jsonify({
            "success": True, 
            "prediction": prediction, 
            "confidence": conf,
            "suspicious_flags": suspicious_flags,
            "probabilities": prob_dict
        })
    except Exception as e:
        print(f"Spam classification error: {e}")
        return jsonify({"success": False, "error": str(e)})

# API: TASK MANAGER

@app.route("/api/task-manager", methods=["GET"])
def get_tasks():
    try:
        db = get_db()
        cursor = db.cursor()
        # Order: Pending first, then by timestamp
        cursor.execute("SELECT * FROM task_logs ORDER BY status ASC, timestamp DESC")
        rows = cursor.fetchall()
        
        tasks = []
        for row in rows:
            task = dict(row)
            # Parse subtasks from JSON string
            try:
                task['subtasks'] = json.loads(row['subtasks']) if row['subtasks'] else []
            except:
                task['subtasks'] = []
            tasks.append(task)
            
        return jsonify({"success": True, "tasks": tasks})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/api/task-manager", methods=["POST"])
def add_task():
    data = request.get_json()
    title = data.get("title", "").strip()
    description = data.get("description", "").strip()
    deadline = data.get("deadline", "")
    task_link = data.get("task_link", "")

    if not title:
        return jsonify({"success": False, "error": "Title is required"})

    # AI Logic to generate priority, subtasks, and estimate
    prompt = (
        f"Analyze this task: Title: {title}. Description: {description}. "
        "Return ONLY a JSON object with exactly these keys: "
        "'priority' (choose from: 'Do this NOW', 'Do Next', 'Normal'), "
        "'subtasks' (a list of 3-4 specific action steps), "
        "'estimated_time' (a string like '45 mins' or '2 hours')."
    )
    
    encoded_prompt = urllib.parse.quote(prompt)
    url = f"https://text.pollinations.ai/{encoded_prompt}"
    
    # Defaults
    priority = "Normal"
    subtasks = ["Break down the task", "Research requirements", "Execute initial steps"]
    est_time = "1 hour"

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            ai_response = response.read().decode('utf-8').strip()
            # Try to extract JSON if the AI included filler
            if "{" in ai_response and "}" in ai_response:
                ai_response = ai_response[ai_response.find("{"):ai_response.rfind("}")+1]
                ai_data = json.loads(ai_response)
                priority = ai_data.get("priority", priority)
                subtasks = ai_data.get("subtasks", subtasks)
                est_time = ai_data.get("estimated_time", est_time)
    except Exception as e:
        print(f"Task AI Error: {e}")

    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute("""
            INSERT INTO task_logs (title, description, priority, subtasks, estimated_time, deadline, task_link)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (title, description, priority, json.dumps(subtasks), est_time, deadline, task_link))
        db.commit()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/api/task-manager/<int:task_id>", methods=["PATCH"])
def complete_task(task_id):
    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute("UPDATE task_logs SET status = 'completed' WHERE id = ?", (task_id,))
        db.commit()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/api/task-manager/<int:task_id>", methods=["DELETE"])
def delete_task(task_id):
    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute("DELETE FROM task_logs WHERE id = ?", (task_id,))
        db.commit()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})
if __name__ == "__main__":
    app.run(debug=True, port=5000)