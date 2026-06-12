"""
Medical RAG System
Hybrid Retrieval: Vector Search (Chroma + HuggingFace) + BM25 + Multi-Query + Conversation Memory
LLM: Google Gemini | Framework: LangChain
Multilingual Support: Arabic + English
"""

import os
import json
from collections import defaultdict
from typing import List
from dataclasses import dataclass, field
from datetime import datetime
from dotenv import load_dotenv

import langchain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


@dataclass
class RAGResponse:
    answer: str
    sources: List[Document]
    query: str
    total_records_found: int
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


@dataclass
class ConversationTurn:
    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


class MedicalRAGSystem:
    """
    Medical RAG System with:
    - Vector Search (Chroma + Multilingual HuggingFace embeddings)
    - BM25 Keyword Search
    - Manual Ensemble Retriever (Vector 70% + BM25 30%)
    - Reciprocal Rank Fusion (RRF)
    - Multi-Query Generation (Arabic + English)
    - Conversation History Memory
    - Multilingual Support: Arabic + English
    """

    def __init__(
        self,
        docs_path: str = ".",
        persist_directory: str = "db/chroma_db",
        # [CHANGED] Replaced English-only model with a multilingual model
        # that understands both Arabic and English
        # Options (best to fastest):
        #   "intfloat/multilingual-e5-large"                          (best quality)
        #   "sentence-transformers/paraphrase-multilingual-mpnet-base-v2" (good & fast)
        #   "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" (lightest)
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        # [CHANGED] Filenames in docs_path that should NOT be treated as
        # medical record files (e.g. casual_intents.json lives in the same
        # directory now that there's no dedicated DATA/ folder).
        excluded_files: List[str] = None,
    ):
        self.docs_path = docs_path
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model

        # [CHANGED] Default exclusion list — avoids accidentally ingesting
        # non-record JSON files (like casual_intents.json) as medical records
        # when docs_path is the project root.
        self.excluded_files = excluded_files if excluded_files is not None else ["casual_intents.json"]

        load_dotenv()

        self.records = None
        self.embeddings = None
        self.vectorstore = None
        self.vector_retriever = None
        self.bm25_retriever = None
        self._retriever_k = 10
        self.llm = None

        self.history: List[ConversationTurn] = []

        # Load casual intents
        self.casual_intents = {}
        casual_path = "casual_intents.json"
        if os.path.exists(casual_path):
            with open(casual_path, "r", encoding="utf-8") as f:
                self.casual_intents = json.load(f)
            print(f"Casual intents loaded ({len(self.casual_intents)} categories)")
        else:
            print("Warning: casual_intents.json not found")

        print(f"Medical RAG System initialized")
        print(f"LangChain version: {langchain.__version__}")

    def load_documents(self) -> List[Document]:
        print(f"\nLoading documents from {self.docs_path}...")

        if not os.path.exists(self.docs_path):
            raise FileNotFoundError(f"Directory '{self.docs_path}' not found.")

        all_records = []

        for filename in os.listdir(self.docs_path):
            if not filename.endswith(".json"):
                continue

            # [CHANGED] Skip non-record JSON files (e.g. casual_intents.json)
            if filename in self.excluded_files:
                continue

            filepath = os.path.join(self.docs_path, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                records = json.load(f)

            # [CHANGED] Guard against the loaded JSON not being a list of
            # records (e.g. if a non-record JSON file slips through)
            if not isinstance(records, list):
                print(f"Skipping '{filename}': expected a JSON list of records.")
                continue

            for record in records:
                # Format red_flags as a readable string if it's a list
                red_flags = record.get("red_flags", [])
                red_flags_str = (
                    " | ".join(red_flags) if isinstance(red_flags, list) else str(red_flags)
                )

                # Format extracted_symptoms as a readable string if it's a list
                extracted_symptoms = record.get("extracted_symptoms", [])
                extracted_symptoms_str = (
                    ", ".join(extracted_symptoms)
                    if isinstance(extracted_symptoms, list)
                    else str(extracted_symptoms)
                )

                # Format possible_conditions as a readable string if it's a list
                possible_conditions = record.get("possible_conditions", [])
                possible_conditions_str = (
                    ", ".join(possible_conditions)
                    if isinstance(possible_conditions, list)
                    else str(possible_conditions)
                )

                content = f"""Record ID: {record.get('record_id', 'Unknown')}
Patient: {record.get('age_group', 'Unknown')} | {record.get('gender', 'Unknown')}

Symptoms Description: {record.get('user_symptom_description', '')}

Extracted Symptoms: {extracted_symptoms_str}

Possible Conditions: {possible_conditions_str}
Primary Condition: {record.get('primary_condition', 'Unknown')}
Confidence: {record.get('confidence_level', 'Unknown')}

Urgency: {record.get('urgency_level', 'Unknown')}
Urgency Reason: {record.get('urgency_reason', '')}
Recommended Specialist: {record.get('recommended_specialist', 'Unknown')}
Recommended Action: {record.get('recommended_action', '')}
Red Flags: {red_flags_str}"""

                doc = Document(
                    page_content=content,
                    metadata={
                        "record_id": record.get("record_id", "Unknown"),
                        "urgency": record.get("urgency_level", "Unknown"),
                        "age_group": record.get("age_group", "Unknown"),
                        "gender": record.get("gender", "Unknown"),
                        "primary_condition": record.get("primary_condition", "Unknown"),
                        "recommended_specialist": record.get("recommended_specialist", "Unknown"),
                        "confidence": record.get("confidence_level", "Unknown"),
                        "source": filename,
                    },
                )
                all_records.append(doc)

        if not all_records:
            raise FileNotFoundError(f"No JSON records found in '{self.docs_path}'.")

        self.records = all_records
        print(f"Loaded {len(self.records)} records")
        return self.records

    def setup_embeddings(self):
        print(f"\nLoading embedding model: {self.embedding_model_name}...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={"device": "cpu"},
            # [PERF] Bigger batch = fewer forward passes when embedding all 1300
            # records on CPU. NOTE: don't put show_progress_bar in encode_kwargs
            # — langchain_huggingface injects it from `show_progress`, and passing
            # both raises "got multiple values for keyword argument".
            encode_kwargs={"batch_size": 64},
            # Prints a progress bar to the logs so the one-time index build is
            # visibly moving, not hung at "Creating new vector store...".
            show_progress=True,
        )
        print("Embedding model loaded!")
        return self.embeddings

    def setup_vectorstore(self):
        print("\nSetting up vector store...")

        if self.embeddings is None:
            self.setup_embeddings()

        if self.records is None:
            raise ValueError("Records not loaded. Call load_documents() first.")

        if os.path.exists(self.persist_directory):
            print(f"Loading existing vector store from {self.persist_directory}...")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
            )
        else:
            print("Creating new vector store...")
            self.vectorstore = Chroma.from_documents(
                documents=self.records,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_metadata={"hnsw:space": "cosine"},
            )

        print(f"Vector store ready! ({self.vectorstore._collection.count()} records)")
        return self.vectorstore

    def setup_ensemble_retriever(self, k: int = 10):
        print(f"\nSetting up Manual Ensemble Retriever (Vector + BM25, top-{k})...")

        if self.vectorstore is None:
            self.setup_vectorstore()

        if self.records is None:
            raise ValueError("Records not loaded. Call load_documents() first.")

        self.vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        self.bm25_retriever = BM25Retriever.from_documents(self.records, k=k)
        self._retriever_k = k

        print("Manual Ensemble Retriever ready (Vector 70% + BM25 30%)")
        return self.vector_retriever

    def setup_llm(self, model: str = "gpt-4.1", temperature: float = 0):
        # Model can be overridden with the OPENAI_MODEL env var without code changes.
        model = os.getenv("OPENAI_MODEL", model)
        print(f"\nSetting up LLM: {model}...")

        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found. Set it in your environment / Railway variables.")

        self.llm = ChatOpenAI(model=model, temperature=temperature)
        print("LLM ready")
        return self.llm

    @staticmethod
    def _content_to_text(content) -> str:
        """Normalize an LLM response's `.content` to a plain string.

        Google Gemini (via langchain) sometimes returns `content` as a *list*
        of parts (strings or {"type": "text", "text": ...} dicts) rather than a
        single string. Calling `.split()` / passing that straight to a pydantic
        `str` field then blows up with "'list' object has no attribute 'split'".
        This flattens any shape back into one string.
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    parts.append(part.get("text", part.get("content", "")))
                else:
                    parts.append(str(part))
            return "".join(parts)
        return str(content)

    def reciprocal_rank_fusion(
        self, results_list: List[List[Document]], k: int = 60
    ) -> List[Document]:
        fused_scores = defaultdict(float)
        doc_lookup = {}

        for results in results_list:
            for rank, doc in enumerate(results):
                doc_id = doc.metadata.get("record_id", str(hash(doc.page_content)))
                doc_lookup[doc_id] = doc
                fused_scores[doc_id] += 1 / (k + rank + 1)

        sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_lookup[doc_id] for doc_id, _ in sorted_docs]

    # [CHANGED] generate_multi_queries now generates queries in both Arabic and English
    # when the input is Arabic, maximizing retrieval coverage across both languages
    def generate_multi_queries(self, user_query: str) -> List[str]:
        if self.llm is None:
            self.setup_llm()

        prompt = f"""You are a multilingual medical search assistant.
Generate 3 different medical search queries based on the following question.

Rules:
- If the question is in Arabic: generate 2 queries in Arabic and 1 query in English.
- If the question is in English: generate 3 queries in English.
- Keep all queries medically accurate and focused on symptoms or conditions.
- Return each query on a new line. No numbering, no extra text, no explanations.

User Query: {user_query}"""

        response = self.llm.invoke(prompt)
        content = self._content_to_text(response.content)
        queries = [q.strip() for q in content.split("\n") if q.strip()]
        return queries

    def _manual_ensemble_search(self, query: str) -> List[Document]:
        """Manual hybrid retrieval: 70% vector + 30% BM25 using weighted scores."""
        vector_results = self.vector_retriever.invoke(query)
        bm25_results = self.bm25_retriever.invoke(query)

        scores = defaultdict(float)
        doc_map = {}

        for rank, doc in enumerate(vector_results):
            doc_id = doc.metadata.get("record_id", str(hash(doc.page_content)))
            scores[doc_id] += 0.7 * (1 / (rank + 1))
            doc_map[doc_id] = doc

        for rank, doc in enumerate(bm25_results):
            doc_id = doc.metadata.get("record_id", str(hash(doc.page_content)))
            scores[doc_id] += 0.3 * (1 / (rank + 1))
            doc_map[doc_id] = doc

        sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
        return [doc_map[k] for k in sorted_keys]

    def search(self, query: str, top_k: int = 5) -> List[Document]:
        if self.vector_retriever is None or self.bm25_retriever is None:
            raise ValueError("Retrievers not initialized. Call initialize_all() first.")

        print(f"\nOriginal query: '{query}'")
        print("Generating alternative queries...")

        generated_queries = self.generate_multi_queries(query)
        all_queries = [query] + generated_queries

        print(f"Searching with {len(all_queries)} queries total:")
        for i, q in enumerate(all_queries, 1):
            label = "(original)" if i == 1 else f"(generated {i - 1})"
            print(f"  {i}. {label} {q}")

        all_results = [self._manual_ensemble_search(q) for q in all_queries]
        fused_results = self.reciprocal_rank_fusion(all_results)

        print(f"\nDone! Returning top {top_k} records out of {len(fused_results)} found.")
        return fused_results[:top_k]

    def initialize_all(self):
        print("\n" + "=" * 60)
        print("Initializing Medical RAG System")
        print("=" * 60)

        try:
            self.load_documents()
            self.setup_embeddings()
            self.setup_vectorstore()
            self.setup_ensemble_retriever()
            self.setup_llm()

            print("\n" + "=" * 60)
            print("Medical RAG System is fully initialized and ready!")
            print("=" * 60)

        except Exception as e:
            print(f"\nInitialization error: {str(e)}")
            raise

    def check_casual_intent(self, query: str):
        """Check if query is casual chitchat — return instant response, no LLM needed."""
        query_lower = query.lower().strip()

        # Detect Arabic by checking Unicode range for Arabic characters
        is_arabic = any('\u0600' <= c <= '\u06FF' for c in query)

        for intent, data in self.casual_intents.items():
            if any(trigger in query_lower for trigger in data["triggers"]):
                print(f"[Casual intent detected: {intent}] — skipping RAG pipeline")
                if is_arabic and "response_ar" in data:
                    return data["response_ar"]
                return data.get("response_en", data.get("response", ""))
        return None

    def ask_with_history(
        self,
        query: str,
        top_k: int = 5,
        history: List[ConversationTurn] = None,
    ) -> RAGResponse:
        # `history` is the per-conversation memory to read from and append to.
        # The web server passes a separate list per session so users don't share
        # each other's context. If None, fall back to the instance-wide history
        # (used by the CLI in main()).
        if history is None:
            history = self.history

        print(f"\n{'=' * 60}")
        print(f"QUESTION (turn {len(history) // 2 + 1}): {query}")
        print(f"{'=' * 60}")

        # ── Casual intent check (zero tokens, zero LLM call) ──
        casual_response = self.check_casual_intent(query)
        if casual_response:
            history.append(ConversationTurn(role="user", content=query))
            history.append(ConversationTurn(role="assistant", content=casual_response))
            return RAGResponse(
                answer=casual_response,
                sources=[],
                query=query,
                total_records_found=0,
            )
        # ──────────────────────────────────────────────────────

        print("\n[Step 1/3] Retrieving relevant records...")
        retrieved_docs = self.search(query, top_k=top_k)

        print("\n[Step 2/3] Building conversation context...")
        history_text = self._format_history_for_prompt(history)

        print(f"\n[Step 3/3] Generating answer (with {len(history)} history messages)...")
        answer = self._generate_answer_with_history(
            query=query,
            retrieved_docs=retrieved_docs,
            history_text=history_text,
            max_records_in_context=top_k,
        )

        history.append(ConversationTurn(role="user", content=query))
        history.append(ConversationTurn(role="assistant", content=answer))

        print(f"\nHistory now has {len(history)} messages ({len(history) // 2} turns)")

        return RAGResponse(
            answer=answer,
            sources=retrieved_docs,
            query=query,
            total_records_found=len(retrieved_docs),
        )

    def _format_history_for_prompt(self, history: List[ConversationTurn] = None) -> str:
        if history is None:
            history = self.history
        if not history:
            return "No previous conversation."

        recent_history = history[-10:]
        lines = []
        for turn in recent_history:
            role_label = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{role_label}: {turn.content}")

        return "\n\n".join(lines)

    def _generate_answer_with_history(
        self,
        query: str,
        retrieved_docs: List[Document],
        history_text: str,
        max_records_in_context: int = 5,
    ) -> str:
        if self.llm is None:
            self.setup_llm()

        docs_to_use = retrieved_docs[:max_records_in_context]
        context_parts = [
            f"--- Record {i} | Urgency: {doc.metadata.get('urgency', 'Unknown')} | "
            f"Condition: {doc.metadata.get('primary_condition', 'Unknown')} | "
            f"Specialist: {doc.metadata.get('recommended_specialist', 'Unknown')} | "
            f"Patient: {doc.metadata.get('age_group', 'Unknown')} {doc.metadata.get('gender', 'Unknown')} ---\n"
            f"{doc.page_content}\n"
            for i, doc in enumerate(docs_to_use, 1)
        ]
        context = "\n\n".join(context_parts)

        # [CHANGED] Prompt now leverages new schema fields:
        # primary_condition, urgency_reason, recommended_specialist, and red_flags.
        # The model is instructed to use each field with purpose rather than just listing them.
        prompt = f"""You are a medical assistant helping doctors analyze patient symptoms.

LANGUAGE RULE (VERY IMPORTANT):
- Detect the language of the doctor's current question.
- If the question is in Arabic, you MUST respond entirely in Arabic.
- If the question is in English, respond in English.
- Never mix languages in your answer.

You are in an ongoing conversation. Use the conversation history to understand follow-up questions.

=== CONVERSATION HISTORY ===
{history_text}
============================

=== NEWLY RETRIEVED RECORDS FOR THIS QUESTION ===
{context}
=================================================

Doctor's Current Question: {query}

---

Use the retrieved records as your PRIMARY source of truth.
Only supplement with general medical knowledge if it does NOT contradict the records.

HOW TO USE EACH FIELD IN THE RECORDS:
- Primary Condition: Use this as the anchor diagnosis — it is the most clinically likely condition. Lead with it.
- Urgency Reason: Use this to explain WHY the urgency level is what it is. Do not just state the level — briefly justify it using this field.
- Recommended Specialist: Always mention this specialty. It is grounded in the records, not a generic guess.
- Recommended Action: Always include this as a concrete next step for the patient — it may contain specific clinical instructions (e.g. medication, tests, referral steps). Do not paraphrase it into vague advice; keep it actionable and specific.
- Red Flags: If any red flags from the records apply to the patient's described symptoms, surface them explicitly as warning signs the patient should watch for. If none apply, do not mention red flags.

IMPORTANT CLINICAL RULES:
- Carefully match symptom SEVERITY between the user's query and the records.
- Do NOT escalate urgency unless the user explicitly describes severe, sudden, worsening, or alarming symptoms.
- If high-urgency cases appear in the records but the user's symptoms seem mild or unspecified, present them as rare possibilities — not the most likely cause.
- Base urgency strictly on symptom intensity described in the user query.
- Consider age_group and gender from the records when assessing likelihood — if the doctor mentions patient demographics, prioritize records that match.

IMPORTANT RESPONSE RULES:
- NEVER mention record IDs, record numbers, or SYM codes
- Speak naturally as a medical assistant
- Maximum 4 sentences total. No exceptions.
- Each sentence must add NEW information. Never repeat or rephrase what was already said.
- Write in plain flowing sentences only (no bullet points)

STRUCTURE YOUR ANSWER EXACTLY LIKE THIS (4 sentences max):
1) ONE sentence: most likely condition + one-line clinical reason.
2) ONE sentence: urgency level + why (from Urgency Reason field).
3) ONE sentence: recommended specialist + recommended action combined.
4) ONLY IF red flags exist: one sentence listing them. Otherwise skip entirely.
"""

        try:
            response = self.llm.invoke(prompt)
            return self._content_to_text(response.content)
        except Exception as e:
            error_message = f"Answer generation failed: {str(e)}"
            print(f"\nError: {error_message}")
            return error_message

    def show_history(self):
        if not self.history:
            print("\nNo conversation history yet.")
            return

        print(f"\n{'=' * 60}")
        print(f"CONVERSATION HISTORY ({len(self.history) // 2} turns)")
        print(f"{'=' * 60}")

        for turn in self.history:
            if turn.role == "user":
                print(f"\n[{turn.timestamp}] DOCTOR:")
                print(f"  {turn.content}")
            else:
                print(f"\n[{turn.timestamp}] ASSISTANT:")
                indented = "\n  ".join(turn.content.split("\n"))
                print(f"  {indented}")

        print(f"\n{'=' * 60}")

    def clear_history(self):
        turn_count = len(self.history) // 2
        self.history = []
        print(f"\nConversation history cleared ({turn_count} turns removed).")

    def display_results(self, results: List[Document], max_chars: int = 400):
        print(f"\nFound {len(results)} results:\n")
        for i, doc in enumerate(results, 1):
            print(f"{'=' * 60}")
            print(f"Rank {i}")
            print(f"{'=' * 60}")
            print(f"Record ID: {doc.metadata.get('record_id', 'Unknown')}")
            print(f"Urgency:   {doc.metadata.get('urgency', 'Unknown')}")
            print(f"\nContent:")
            print(doc.page_content[:max_chars])
            if len(doc.page_content) > max_chars:
                print("...")
            print()

    def display_response(self, response: RAGResponse):
        print(f"\n{'=' * 60}")
        print("GENERATED ANSWER")
        print(f"{'=' * 60}")
        print(response.answer)

        print(f"\n{'=' * 60}")
        print(f"SOURCES ({response.total_records_found} records used)")
        print(f"{'=' * 60}")
        for i, doc in enumerate(response.sources, 1):
            print(
                f"  {i}. Record {doc.metadata.get('record_id', 'Unknown')} "
                f"| Urgency: {doc.metadata.get('urgency', 'Unknown')}"
            )


def main():
    rag = MedicalRAGSystem(
        docs_path=".",
        persist_directory="db/chroma_db",
    )

    rag.initialize_all()

    # English questions
    response1 = rag.ask_with_history("What symptoms are associated with diabetes?", top_k=5)
    rag.display_response(response1)

    response2 = rag.ask_with_history("What about the urgency level for those patients?", top_k=5)
    rag.display_response(response2)

    # Arabic questions — the system will now respond in Arabic automatically
    response3 = rag.ask_with_history("هل في حالات أطفال عندهم نفس الأعراض؟", top_k=5)
    rag.display_response(response3)

    response4 = rag.ask_with_history("عندي مريض بيشكو من ألم في الصدر وضيق في التنفس، إيه الاحتمالات؟", top_k=5)
    rag.display_response(response4)

    rag.show_history()
    rag.clear_history()


if __name__ == "__main__":
    main()
