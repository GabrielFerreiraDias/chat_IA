from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from document_processador import DocumentProcessor
from agent import AIAgent
import os

app = Flask(__name__, static_folder="web-chat-app/src", static_url_path="/")
CORS(app)

# 🧠 Configuração: LLM LOCAL ATIVADO (respostas mais naturais, ~30-60s)
# Defina USE_LLM=false para modo tradicional rápido
USE_LLM = os.getenv("USE_LLM", "true").lower() == "true"

# ⏱️ Aumenta timeout para permitir respostas LLM completas
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
from werkzeug.serving import WSGIRequestHandler
WSGIRequestHandler.timeout = 120

print("=" * 60)
print(f"🚀 Iniciando Chat {'COM LLM Local' if USE_LLM else 'MODO TRADICIONAL'}")
print("=" * 60)

dp = DocumentProcessor("aprendizado")
dp.process_all_documents()
agent = AIAgent(dp, use_llm=USE_LLM)

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/api/init", methods=["POST"])
def init_agent():
    agent.train_and_save_model()
    return jsonify({"status": "ok"})

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    msg = data.get("message", "").strip()
    print(f"\n>>> Usuário: {msg}")
    if USE_LLM:
        print("⏳ Gerando resposta com LLM (pode levar 30-60 segundos)...")
    reply = agent.chat(msg)
    print(f"<<< IA: {reply}\n")
    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(debug=True)