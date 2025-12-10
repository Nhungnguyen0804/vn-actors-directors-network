import gradio as gr

def generate_answer(message, history=None):
    return f"[Fake model] Tôi nhận được: '{message}'"


def chat_fn(message, history):
    reply = generate_answer(message, history)

    # Append history đúng format Gradio
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply}
    ]

    return reply, history


chatbot_ui = gr.ChatInterface(
    fn=chat_fn,
    title="My Chatbot",
    description="Chatbot demo (cắm model thật vào là chạy)",
)

if __name__ == "__main__":
    chatbot_ui.launch()
