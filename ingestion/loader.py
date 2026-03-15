import os
from pathlib import Path
def load_markdown_files(base_path: str):
    """
    Load all markdown files from the knowledge base.
    Expected strcuture:
    data/
        runbooks/
        incidents/
    Returns:
        list[dict] with fields:
            text: document content
            metadata: metadata about the document
    """
    documents = []
    base = Path(base_path)
    for category in ["runbooks", "incidents"]:
        folder = base / category
        if not folder.exists():
            continue
        for file in folder.glob("*.md"):
            with open(file, "r", encoding="utf-8") as f:
                text = f.read()
            documents.append(
                {
                    "text": text,
                    "metadata": {
                        "source": str(file),
                        "filename": file.name,
                        "doc_type": category
                    }
                }
            )
    return documents

if __name__ == "__main__":
    docs = load_markdown_files("incident-copilot/data")
    print(f"Loaded {len(docs)} documents")
    for d in docs[7:]:
        print("\n--")
        print(d["metadata"])













