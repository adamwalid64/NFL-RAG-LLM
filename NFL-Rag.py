import os
import pickle
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import tiktoken

# Load API key from .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

DATA_DIR = "Data/"  # Standardized processed documents directory
MODEL_NAME = "gpt-4o"  # Use gpt-3.5-turbo if budget is tight
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Cost tracking
def count_tokens(text, model="gpt-4"):
    """Count tokens in text for cost estimation"""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except:
        # Fallback: rough estimate (1 token ≈ 4 characters)
        return len(text) // 4

def estimate_cost(input_tokens, output_tokens, model="gpt-4o"):
    """Estimate cost based on token usage"""
    if model == "gpt-4o":
        input_cost_per_1k = 0.005
        output_cost_per_1k = 0.015
    elif model == "gpt-3.5-turbo":
        input_cost_per_1k = 0.0005
        output_cost_per_1k = 0.0015
    else:
        return 0
    
    input_cost = (input_tokens / 1000) * input_cost_per_1k
    output_cost = (output_tokens / 1000) * output_cost_per_1k
    return input_cost + output_cost

# Load documents from a specific pickle file
def load_documents_from_specific_file(folder_path, filename):
    file_path = os.path.join(folder_path, filename)
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found!")
        return []
    try:
        with open(file_path, 'rb') as f:
            documents = pickle.load(f)
        print(f"Loaded {len(documents)} documents from {filename}")
        return documents
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return []

def get_rag_prediction():
    # Hardcoded dataset file
    chunked_doc_file = os.path.join(DATA_DIR, "sentiment_articles_20250808_035102_chunked_docs.pkl")

    print(f"Loading documents from specific file: {chunked_doc_file}")
    docs = load_documents_from_specific_file(DATA_DIR, "sentiment_articles_20250808_035102_chunked_docs.pkl")

    if not docs:
        print("No documents found! Make sure to run load-sentiment.py first.")
        return None

    print(f"Loaded {len(docs)} total documents")

    # Estimate embedding costs
    total_text = " ".join([doc.page_content for doc in docs])
    embedding_tokens = count_tokens(total_text, "text-embedding-ada-002")
    embedding_cost = (embedding_tokens / 1000) * 0.0001
    print(f"Estimated embedding cost: ${embedding_cost:.4f}")

    print("Embedding and building vector store...")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Setup QA chain
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    llm = ChatOpenAI(model_name=MODEL_NAME)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    query = """System role (use as “system”/“instructions”):
You are an NFL fantasy draft assistant. Always use the most recent, trustworthy sources (news, injuries, depth charts, ADP/ECR, beat reports). Prioritize information from the last 7 days; if anything changed within 24 hours, highlight it. Resolve conflicts by citing multiple sources and favoring the newest/most credible.

Context / League Settings:

Teams: 10

Roster: 1 QB, 2 RB, 2 WR, 1 TE, 1 FLEX (WR/RB), 1 K, 1 DEF, 6 Bench

Draft type: {Snake or Auction} (default: Snake)

Draft slot: {#1–#10} (if unknown, give plans for early/mid/late slots)

Scoring: {Standard | Half-PPR | PPR} (default: Half-PPR)

Waivers/FAAB: {detail if relevant}

Retrieval directives (RAG):

Pull current ADP and ECR; call out ADP vs ECR deltas and injury/status notes (DNP, PUP, holdout, suspension).

Check team depth charts, camp/preseason usage, and coordinator tendencies.

Include bye weeks, playoff weeks (Wk 15–17) schedule strength if available.

Mark any player with red flags (injury setbacks, snap-count limits) as RISK.

Task:
Produce a round-by-round draft plan that maximizes upside while managing risk and positional scarcity for this roster format.

Requirements:

Tiered Positional Boards (QB/RB/WR/TE/K/DEF) with Tier breaks and short blurbs (why they fit this format).

Round-by-Round Targets (Rounds 1–12) for a 10-team snake:

For each round, list Primary Target(s), Backup Options (2–4), Emergency Pivot (if the board collapses), and Positional Goal (e.g., “Leave this round with RB2 or elite WR”).

Note expected ADP range and whether the pick is a value, fair, or a reach.

Roster Construction Rules:

In 10-team leagues, wait on QB/TE unless elite value falls.

Aim to leave Rounds 1–5 with 3–4 RB/WR starters; FLEX should be best available RB/WR value.

K/DEF in the last two rounds unless a truly elite DEF value drops (note if that ever makes sense).

Prefer high-upside bench stashes over low-ceiling vets.

Stacking & Correlation:

If a top QB is drafted, suggest stack candidates (WR/RB/TE) and late bring-backs for playoff weeks.

Risk Controls & Tiebreakers:

Avoid overloading a single bye week across RB/WR starters.

Break ties with: (a) secure role > (b) offensive pace > (c) red‑zone usage > (d) playoff schedule outlook.

Late-Round Plan:

6–8 sleepers and contingent value handcuffs; label Immediate Flex Upside, Injury Stash, or Post‑Week‑1 Cuttable.

News Guardrails:

Flag any player whose rank depends on pending news (e.g., MRI, suspension ruling). Provide a pre-draft check list.

Output format (concise & scannable):

Section A: Tiered Boards (by position).

Section B: Round-by-Round (R1→R12):

Primary, Backups, Emergency Pivot, Positional Goal, ADP vs ECR Note, Risk.

Section C: Sleepers/Handcuffs (labels above).

Section D: K/DEF strategy (stream vs elite hold).

Section E: Last‑Minute News Checklist (bulleted).

Example (format only, not rankings):

Round 3 (Pick ~30–32)

Positional Goal: Lock WR2 or RB2

Primary: {Player A, WR — ADP 28 | ECR 24 | Value}

Backups: {Player B RB}, {Player C WR}, {Player D RB}

Emergency Pivot: {Player E TE — falls 12+ spots}

ADP vs ECR: Player A +4 ECR (market is catching up)

Risk: Minor camp hamstring → monitor Fri practice

Constraints:

Keep the entire plan under 800–1,000 words so it’s usable live.

Clearly bold any item updated in the last 24 hours.

If the draft slot or scoring is missing, generate three paths (Early 1–3, Mid 4–7, Late 8–10) and note key differences.

Final line:
“Return all recommendations assuming the roster: 1 QB, 2 RB, 2 WR, 1 TE, 1 FLEX (WR/RB), 1 K, 1 DEF, 6 Bench, in a 10-team league.”"""

    print("Querying LLM...")

    # Estimate query costs
    input_tokens = count_tokens(query + " " + total_text[:2000])  # Rough estimate
    estimated_query_cost = estimate_cost(input_tokens, 200, MODEL_NAME)  # Assume 200 output tokens
    print(f"Estimated query cost: ${estimated_query_cost:.4f}")

    response = qa_chain.invoke({"query": query})

    # Count actual output tokens
    output_tokens = count_tokens(response["result"])
    actual_query_cost = estimate_cost(input_tokens, output_tokens, MODEL_NAME)
    print(f"Actual query cost: ${actual_query_cost:.4f}")
    print(f"Total estimated cost: ${embedding_cost + actual_query_cost:.4f}")

    print("\n🔮 RAG Prediction:\n", response["result"])

    return {
        "prediction": response["result"],
        "embedding_cost": embedding_cost,
        "query_cost": actual_query_cost,
        "total_cost": embedding_cost + actual_query_cost,
        "documents_loaded": len(docs)
    }

if __name__ == "__main__":
    # Example usage with hardcoded dataset
    result = get_rag_prediction()





