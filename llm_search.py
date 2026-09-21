import argparse
import os

try:
    from groq import Groq
    HAS_LLM = True
except ImportError:
    HAS_LLM = False

from main import load_or_build, search, model

# Default model to use if GROQ_MODEL is not set in the environment.
# Groq's model lineup changes over time (models get deprecated/replaced),
# so this can be overridden without touching the code:
#   export GROQ_MODEL="openai/gpt-oss-20b"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"


def get_available_model_ids(client):
    """Return the list of model IDs the current API key can access."""
    try:
        return [m.id for m in client.models.list().data]
    except Exception:
        return []


def resolve_groq_model(client, requested_model):
    """
    Make sure the requested model is actually available to this API key.
    If not, fall back to the first available chat-capable model and warn
    the user, instead of crashing on every query.
    """
    available = get_available_model_ids(client)

    if not available:
        return requested_model

    if requested_model in available:
        return requested_model

    # Skip audio/TTS/guard models, keep general chat-capable ones
    excluded_markers = ("whisper", "tts", "guard", "prompt-guard")
    candidates = [
        m for m in available
        if not any(marker in m.lower() for marker in excluded_markers)
    ]
    fallback = candidates[0] if candidates else available[0]

    print(
        f"\n[!] Model '{requested_model}' is not available for your API key.\n"
        f"    Available models: {', '.join(available)}\n"
        f"    Falling back to '{fallback}'.\n"
    )
    return fallback


def run_llm_cli(vectors, graph, state, chunks, k=5):
    print("Semantic search + Groq LLM Summarizer ready. Type 'quit' to exit.")

    client = None
    groq_model = None
    if not HAS_LLM:
        print("\n[!] 'groq' library is not installed. Running in RAG-only mode.")
    else:
        api_key = os.environ.get("GROQ_API_KEY")
        if api_key:
            client = Groq(api_key=api_key)
            requested_model = os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL)
            groq_model = resolve_groq_model(client, requested_model)
        else:
            print("\n[!] GROQ_API_KEY environment variable not found. Running in RAG-only mode.")

    while True:
        query = input("\n> ").strip()

        if query.lower() in ("quit", "exit"):
            break
        if not query:
            continue

        results = search(query, model, vectors, graph, state, chunks, k=k)

        print(f"\nTop {len(results)} retrieved sources:\n")
        for i, chunk in enumerate(results, 1):
            print(f"{i}. [{chunk['source']}]")
            print(f"   {chunk['text'][:100]}...\n")
            
        if client:
            print("Generating AI summary...\n")
            context = "\n\n".join([f"Source ({c['source']}): {c['text']}" for c in results])
            
            prompt = (
                "You are a helpful assistant. Answer the user's question based EXCLUSIVELY "
                "on the provided context. "
                "Provide a single, cohesive response. Do NOT summarize each source individually. "
                "Your answer MUST be written in English and MUST be strictly between 5 and 15 lines long. "
                f"If the answer is not in the context, explicitly state so.\n\n"
                f"Question: {query}\n\nContext:\n{context}"
            )
            
            try:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    model=groq_model,
                )
                print(f"--- AI Response ---\n{chat_completion.choices[0].message.content}\n-------------------")
            except Exception as e:
                error_text = str(e)
                if "model_not_found" in error_text or "does not exist" in error_text:
                    print(
                        f"Error: model '{groq_model}' is not available for your API key.\n"
                        f"Set GROQ_MODEL to one of the models listed at "
                        f"https://console.groq.com/docs/models, e.g.:\n"
                        f'  export GROQ_MODEL="llama-3.1-8b-instant"'
                    )
                else:
                    print(f"Error connecting to Groq API: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Search over PDFs")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuilding the index")
    parser.add_argument("-k", type=int, default=5, help="Number of results to retrieve")
    args = parser.parse_args()

    vectors, graph, node_levels, state, chunks = load_or_build(force_rebuild=args.rebuild)
    
    run_llm_cli(vectors, graph, state, chunks, k=args.k)