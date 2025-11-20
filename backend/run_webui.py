"""Script to launch a simple Gradio Web UI which talks to the Django API or local ChatModel."""
import os

from ai_tripod_backend.chat import ChatModel


def run_gradio():
    try:
        import gradio as gr
    except Exception:
        raise RuntimeError("gradio not installed; see requirements.txt")

    chat = ChatModel()

    def infer(prompt: str):
        out = chat.generate(prompt)
        return out.get("choices", [])[0]["message"]["content"]

    with gr.Blocks() as demo:
        txt = gr.Textbox(label="Prompt")
        out = gr.Textbox(label="Response")
        btn = gr.Button("Run")
        btn.click(fn=infer, inputs=txt, outputs=out)

    demo.launch(share=False)


if __name__ == '__main__':
    run_gradio()
