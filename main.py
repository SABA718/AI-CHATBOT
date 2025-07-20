import gradio as gr
import json
import random
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

nltk.download('punkt')

# Load intents
with open("intents.json") as f:
    data = json.load(f)

# Prepare data
texts = []
labels = []
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        texts.append(pattern)
        labels.append(intent["tag"])

# Train model
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(texts, labels)

# Chatbot function
def chatbot(user_input, history):
    try:
        tag = model.predict([user_input])[0]
        for intent in data["intents"]:
            if intent["tag"] == tag:
                response = random.choice(intent["responses"])
                history.append([user_input, response])
                return history, ""
        return history, ""
    except Exception as e:
        return history, f"Error: {str(e)}"

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🤖 AI Chatbot (ChatGPT-style)")
    chatbot_ui = gr.Chatbot(label="Chatbot")
    msg = gr.Textbox(placeholder="Type your message here...", label="Textbox")
    clear = gr.Button("Clear")

    msg.submit(chatbot, [msg, chatbot_ui], [chatbot_ui, msg])
    clear.click(lambda: [], None, chatbot_ui, queue=False)

demo.launch(inbrowser=True)
