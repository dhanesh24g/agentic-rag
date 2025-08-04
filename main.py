from dotenv import load_dotenv

load_dotenv(override=True)

from graph.graph import app
import gradio as gr


def chat(user_input, history):
    result = app.invoke(input={"question": user_input})
    return {"text": result.get("generation")}


if __name__ == "__main__":
    gr.ChatInterface(chat, type="messages").launch()

