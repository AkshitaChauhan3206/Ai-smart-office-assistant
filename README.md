# AI Smart Office Assistant 🚀

A modern, SaaS-style AI-powered web application designed to supercharge workplace productivity. Features a persistent dark mode UI, a genuine Scikit-Learn Spam Classification pipeline, an intelligent Priority Task Manager with Focus Mode, and a real LLM-powered Professional Email Generator.

## ✨ Core Features

### 1. Smart Email Generator
Stop writing tedious corporate emails from scratch. Simply input your context (e.g., *"ask HR for a salary raise"*), select a tone (Professional, Friendly, Urgent, Apologetic), and the integrated genuine **LLM Language AI** will instantly draft a perfect, structured, multi-paragraph email ready to send.

### 2. Spam & Priority Classifier
Powered by a genuine **Scikit-Learn Machine Learning Pipeline** (*Multinomial Naive Bayes & TF-IDF Vectorization*), this tool doesn't just guess if an email is spam—it calculates the mathematical probability. 
- Analyzes custom input against trained datasets.
- Highlights the exact **suspicious keywords** driving the algorithm's decision.

### 3. Intelligent Task Manager
A mathematically driven Kanban-style board that goes beyond a standard To-Do list.
- **Priority Engine:** Automatically calculates missing deadlines and urgency keywords to sort tasks strictly from "Do this NOW" to "Do Later."
- **Anti-Gravity / Focus Mode:** Hides the application UI and isolates your singular, absolute highest-priority task alongside a built-in interactive **Pomodoro Timer (25m/5m)**.

---

## 🛠️ Tech Stack
- **Backend:** Python + Flask API
- **Machine Learning:** Scikit-Learn, Pandas, NumPy, NLTK
- **Generative AI:** Pollinations.ai Text API
- **Database:** SQLite (Persistent Storage)
- **Frontend:** HTML5, CSS Variables (Dynamic System-Wide Dark Mode), Vanilla JavaScript

---

## 🚀 Installation & Setup

### 1. Clone & Setup Environment
Ensure you have Python 3.x installed. Navigate to the project directory and install the required dependencies:
```bash
pip install flask scikit-learn pandas numpy nltk


2. Train the Machine Learning Pipeline
Before running the server, you must build the Spam classification models vectorizer.pkl and spam_model.pkl. A synthetic trainer script is included that generates a dataset and compiles the AI mathematically.
bash
python train_model.py
(Wait for the console to indicate 100% Training Accuracy).

3. Boot the Server
Run the Flask application:
bash
python app.py


4. Open Application
Navigate to the local server in your browser:

text
http://127.0.0.1:5000


📂 Project Architecture


ai-smart-office-assistant
├── app.py                  # Main Flask Backend / API router
├── train_model.py          # TF-IDF Scikit-Learn Model Compiler
├── database.db             # SQLite localized logs (Emails, Tasks, Spam queries)
├── spam_model.pkl          # Compiled Naive Bayes Output (Generated)
├── vectorizer.pkl          # Compiled TF-IDF Array Output (Generated)
├── static/
│   └── css/
│       └── style.css       # Dynamic SaaS UI & Variable Dark Mode
└── templates/
    ├── base.html           # Master Layout + Sidebar Nav + Theme Logic
    ├── index.html          # Dashboard
    ├── email_generator.html
    ├── spam_classifier.html
    └── task_manager.html   # Feature views