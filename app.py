"""Step 6 - The RAG-based dialogue system (Gradio web app).

Run:  python app.py           then open http://127.0.0.1:7860 in your browser.
      python app.py --share   additionally creates a temporary public link (*.gradio.live)

Each turn:
  1. take the new message + the chat history
  2. rewrite a follow-up into a standalone question (dialogue context)
  3. retrieve + answer with the engine chosen in the dropdown (RAG / GraphRAG / LightRAG)
  4. show the answer, the rewritten question, the sources and the response time
"""
import sys
import time

import gradio as gr

from engines import ENGINES, rewrite_question


def as_text(content):
    """Gradio may store a message as a string or as a list of parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


async def respond(message, history, engine):
    turns = [{"role": m["role"], "content": as_text(m["content"]).split("\n\n---\n")[0]}
             for m in history if isinstance(m, dict)]
    t0 = time.time()
    question = rewrite_question(message, turns)
    try:
        result = await ENGINES[engine](question)
    except Exception as e:  # show errors in the chat instead of crashing the app
        return f"Error from {engine}: {e}"
    footer = (f"\n\n---\n*Engine:* {engine} · *Time:* {time.time() - t0:.1f} s  \n"
              f"*Searched for:* {question}  \n*Sources:* {', '.join(result['sources'])}")
    return result["answer"] + footer


demo = gr.ChatInterface(
    fn=respond,
    additional_inputs=[gr.Dropdown(list(ENGINES), value="Plain RAG", label="Knowledge augmentation engine")],
    title="Airbus & Boeing Documentation Assistant",
    description=("RAG-based dialogue system over official Airbus and Boeing documents. "
                 "Choose RAG, GraphRAG or LightRAG under 'Additional Inputs'. Follow-up questions are understood."),
    examples=[["What is the maximum design takeoff weight of the Boeing 787-8?", "Plain RAG"],
              ["Which engines are offered on the A320neo?", "LightRAG (hybrid)"],
              ["What are the main safety themes across the Airbus Safety First magazines?", "GraphRAG (global)"]],
)

if __name__ == "__main__":
    demo.launch(inbrowser=True, share="--share" in sys.argv)
