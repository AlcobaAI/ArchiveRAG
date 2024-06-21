import os
import argparse
import gradio as gr
from rag import ChatPDF

COLOR = os.environ.get("COLOR", "blue")
EMOJI = os.environ.get("EMOJI", "😊")

def generate(prompt: str, assistant: ChatPDF) -> str:
    response = assistant.ask(prompt)
    return response.strip()

def chatbot_interface(prompt: str, history: list, assistant: ChatPDF) -> str:
    return generate(prompt, assistant)

def upload_file(files: list, assistant: ChatPDF) -> str:
    for file in files:
        assistant.ingest(file.name)
    return f"Uploaded {len(files)} files"

def main():
    parser = argparse.ArgumentParser(description="Run ITrust AI Chatbot with specified models and paths")
    parser.add_argument('--model', type=str, default="llama3", help="Model name to use")
    parser.add_argument('--embedding_model', type=str, required=True, help="Path to the embedding model")
    parser.add_argument('--data_path', type=str, default="pdfs/", help="Path to the PDF data directory")

    args = parser.parse_args()

    assistant = ChatPDF(model=args.model, embedding_model=args.embedding_model)

    for pdf_file in os.listdir(args.data_path):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(args.data_path, pdf_file)
            try:
                assistant.ingest(pdf_path)
                print(f"Successfully ingested {pdf_path}")
            except Exception as e:
                print(f"Failed to ingest {pdf_path}: {e}")

    upload_component = gr.File(label="Upload PDFs", file_count="multiple")
    upload_interface = gr.Interface(
        fn=lambda files: upload_file(files, assistant),
        inputs=upload_component,
        outputs="text"
    )

    chat_interface = gr.ChatInterface(
        fn=lambda prompt, history: chatbot_interface(prompt, history, assistant),
        title=f"{EMOJI} InterPARES ITrust AI: ArchiveRAG",
        description="Ask questions and get answers related to Interpares Itrust AI.",
        examples=[
            ["Write a very short summary of Data Sanitation Techniques by Edgar Dale, and write a citation in APA style."],
            ["What is the difference between a SGML document and a SGML-compliant document?"],
            ["What is a trustworthy digital repository, where can you find this information?"],
            ["What are things a repository must have?"],
            ["What are some different types of records?"],
            ["What principles should record creators follow?"],
        ],
        theme=gr.themes.Soft(primary_hue=COLOR),
    ).queue()

    demo = gr.TabbedInterface([chat_interface, upload_interface], ["Chat", "File Upload"], css="styles.css")
    demo.launch(share=True, debug=True)

if __name__ == "__main__":
    main()
