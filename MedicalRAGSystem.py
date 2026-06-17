"""
Medical RAG System
Hybrid Retrieval: Vector Search (Chroma + OpenAI embeddings) + BM25 + Multi-Query + Conversation Memory
LLM: OpenAI GPT-4.1 | Embeddings: text-embedding-3-small | Framework: LangChain
Multilingual Support: Arabic + English
"""

import os
import re
import json
from collections import defaultdict
from typing import List
from dataclasses import dataclass, field
from datetime import datetime
from dotenv import load_dotenv

import langchain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
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
    - Vector Search (Chroma + OpenAI multilingual embeddings)
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
        # OpenAI API embedding model (multilingual, runs server-side so there's
        # no CPU bottleneck — ~100ms/query vs ~15s for a local model on Railway).
        # Overridable via EMBEDDING_MODEL (e.g. "text-embedding-3-large").
        embedding_model: str = "text-embedding-3-small",
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
        # Allow swapping the embedding model from Railway without code changes.
        model_name = os.getenv("EMBEDDING_MODEL", self.embedding_model_name)
        self.embedding_model_name = model_name

        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found. Set it in your environment / Railway variables.")

        # OpenAI API embeddings: no local model download, no CPU inference.
        # Embedding is a fast network call, so query latency drops from ~15s
        # (e5-large on CPU) to ~100ms, and the one-time index build over 1300
        # records finishes in seconds instead of an hour.
        print(f"\nUsing OpenAI embeddings: {model_name}...")
        self.embeddings = OpenAIEmbeddings(model=model_name)
        print("Embedding model ready!")
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
        self, results_list: List[List[Document]], k: int = 60, weights: List[float] = None
    ) -> List[Document]:
        if weights is None:
            weights = [1.0] * len(results_list)

        fused_scores = defaultdict(float)
        doc_lookup = {}

        for results, weight in zip(results_list, weights):
            for rank, doc in enumerate(results):
                doc_id = doc.metadata.get("record_id", str(hash(doc.page_content)))
                doc_lookup[doc_id] = doc
                fused_scores[doc_id] += weight * (1 / (k + rank + 1))

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

    def route_message(self, query: str, history_text: str = "No previous conversation."):
        """ONE LLM call that both classifies the message and (if clinical) writes
        the search queries — so a non-clinical message ("who are you", greetings,
        chit-chat) never reaches the slow vector-embedding retrieval step.

        Returns a dict:
          {"clinical": False, "reply": "<short reply>", "queries": []}   or
          {"clinical": True,  "reply": None,            "queries": [...]}.
        """
        if self.llm is None:
            self.setup_llm()

        prompt = f"""You are a medical assistant. Read the user's LATEST message and the recent conversation.

Recent conversation:
{history_text}

User's latest message: {query}

Decide if the latest message is CLINICAL (describes symptoms, asks about a
condition / diagnosis / urgency / specialist / medication / test / treatment,
or is a follow-up that continues a previous clinical answer such as "are you
sure?", "and for children?", "what's the treatment?") or NON-CLINICAL (greeting,
asking who/what you are, thanks, small talk, or anything not about a medical case).

Respond in EXACTLY one of these two formats, nothing else:

If NON-CLINICAL:
CASUAL
<one short, warm, natural reply in the SAME language as the user; do not diagnose>

If CLINICAL:
CLINICAL
<medical search query 1>
<medical search query 2>
<medical search query 3>
(For Arabic input: 2 Arabic queries + 1 English. For English input: 3 English. No numbering.)"""

        response = self.llm.invoke(prompt)
        lines = [l.strip() for l in self._content_to_text(response.content).split("\n") if l.strip()]

        if not lines:
            # Defensive fallback: treat as clinical with the raw query.
            return {"clinical": True, "reply": None, "queries": [query]}

        tag = lines[0].upper()
        rest = lines[1:]

        if tag.startswith("CASUAL"):
            reply = " ".join(rest).strip() or "أنا مساعدك الطبي — قوللي بتحس بإيه وأنا أساعدك."
            return {"clinical": False, "reply": reply, "queries": []}

        # CLINICAL (or anything unexpected → default to clinical so we never
        # accidentally refuse a real medical question).
        queries = rest if tag.startswith("CLINICAL") else lines
        return {"clinical": True, "reply": None, "queries": queries or [query]}

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

    def search(self, query: str, top_k: int = 5, generated_queries: List[str] = None) -> List[Document]:
        if self.vector_retriever is None or self.bm25_retriever is None:
            raise ValueError("Retrievers not initialized. Call initialize_all() first.")

        print(f"\nOriginal query: '{query}'")

        # `generated_queries` may be supplied by route_message() to avoid a second
        # LLM call. If not provided (e.g. the CLI), generate them here.
        if generated_queries is None:
            print("Generating alternative queries...")
            generated_queries = self.generate_multi_queries(query)

        # OpenAI API embeddings are fast/cheap, so we run the full hybrid
        # (vector + BM25) ensemble on every query variant for best recall.
        all_queries = [query] + generated_queries
        print(f"Hybrid search across {len(all_queries)} query variant(s)")
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

    # Arabic diacritics (tashkeel) + tatweel — stripped so "شكراً" == "شكرا".
    _AR_DIACRITICS = re.compile("[ؐ-ًؚ-ْٰـ]")
    _AR_LETTER_MAP = str.maketrans({
        "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",  # أإآٱ -> ا
        "ى": "ي", "ئ": "ي",  # ى ئ -> ي
        "ؤ": "و",  # ؤ -> و
        "ة": "ه",  # ة -> ه
    })

    @classmethod
    def _normalize(cls, text: str) -> str:
        """Lowercase, strip Arabic diacritics, unify letter shapes, drop
        punctuation and collapse spaces so casual triggers match regardless of
        spelling variants (أهلا/اهلا, شكراً/شكرا, ازيك؟/ازيك)."""
        text = text.strip().lower()
        text = cls._AR_DIACRITICS.sub("", text)
        text = text.translate(cls._AR_LETTER_MAP)
        # keep word chars + Arabic letters; everything else becomes a space
        text = re.sub("[^\\w؀-ۿ]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def check_casual_intent(self, query: str):
        """Instant canned reply for clear chit-chat - no LLM, no retrieval.

        Conservative on purpose: it only fires when the message is essentially
        the casual phrase (exact match, or the trigger plus at most one extra
        word). A message that opens with a greeting but also describes a symptom
        does NOT short-circuit here - it falls through to the smart router, which
        sees the symptom and sends it to the RAG pipeline. This stops a greeting
        from hijacking a real clinical message.
        """
        is_arabic = any('؀' <= c <= 'ۿ' for c in query)
        norm = self._normalize(query)
        if not norm:
            return None
        n_words = len(norm.split())

        for intent, data in self.casual_intents.items():
            for trigger in data["triggers"]:
                t = self._normalize(trigger)
                if not t:
                    continue
                t_words = len(t.split())
                exact = norm == t
                # "contains" only counts when the message is barely longer than
                # the trigger itself (trigger words + 1), i.e. just politeness.
                near = (t in norm) and (n_words <= t_words + 1)
                if exact or near:
                    print(f"[Casual intent detected: {intent}] - skipping RAG pipeline")
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

        history_text = self._format_history_for_prompt(history)

        # ── Router (1 LLM call): classify + generate queries together. A
        #    non-clinical message gets a short reply and SKIPS the slow
        #    embedding retrieval entirely. ──
        print("\n[Step 1/3] Routing message (clinical vs casual)...")
        route = self.route_message(query, history_text)
        if not route["clinical"]:
            print("[Routed as NON-CLINICAL] — skipping retrieval")
            history.append(ConversationTurn(role="user", content=query))
            history.append(ConversationTurn(role="assistant", content=route["reply"]))
            return RAGResponse(
                answer=route["reply"],
                sources=[],
                query=query,
                total_records_found=0,
            )

        print("[Routed as CLINICAL] Retrieving relevant records...")
        retrieved_docs = self.search(query, top_k=top_k, generated_queries=route["queries"])

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
- If the question is in Arabic, you MUST respond entirely in Arabic (use natural Egyptian/Modern Standard Arabic, not formal classical).
- If the question is in English, respond in English.
- Never mix languages in your answer.

This message has already been classified as a clinical question. Answer it
clinically. Use the conversation history to understand follow-up questions
(e.g. "are you sure?", "what about children?", "and the treatment?").

=== CONVERSATION HISTORY ===
{history_text}
============================

=== NEWLY RETRIEVED RECORDS FOR THIS QUESTION ===
{context}
=================================================

Doctor's Current Question: {query}

---

STRICT SAFETY RULES — NEVER VIOLATE THESE:
1. NEVER recommend, prescribe, suggest, or name any medication, drug, antibiotic,
   painkiller, supplement, or treatment plan. If asked, politely explain you cannot
   recommend medications and advise consulting a licensed healthcare professional.
2. NEVER provide dosing instructions or tell anyone to start or stop any medication.
3. NEVER present a diagnosis as certain. Always use language such as "possible
   condition", "likely cause", or "requires medical evaluation to confirm".
4. NEVER infer or assume the patient's age, gender, or demographics from the
   conversation. Only use age/gender information if the doctor has explicitly stated
   it in the current message (e.g. "45-year-old male patient"). Do not carry over
   assumed demographics from previous turns.
5. Focus only on: symptom assessment, possible conditions, urgency level, red flags,
   and the appropriate specialist or next step.
6. If symptoms suggest an emergency, clearly instruct the doctor to refer the patient
   to the nearest emergency department immediately.

---

HOW TO USE EACH FIELD IN THE RECORDS:
- Primary Condition: Use as the most likely condition — always frame it as "possible"
  or "likely", never as a confirmed diagnosis.
- Urgency Reason: Briefly justify WHY the urgency level is what it is using this
  field. Do not just state the level.
- Recommended Specialist: Always mention the appropriate specialty grounded in the
  records.
- Recommended Action: Include only non-medication next steps — referral, tests,
  imaging, or when to seek emergency care. Remove any medication references.
- Red Flags: If any red flags from the records apply to the described symptoms,
  surface them explicitly. If none apply, skip entirely.

IMPORTANT CLINICAL RULES:
- Carefully match symptom SEVERITY between the query and the records.
- Do NOT escalate urgency unless the doctor explicitly describes severe, sudden,
  worsening, or alarming symptoms.
- If high-urgency records appear but the described symptoms seem mild, present those
  conditions as rare possibilities — not the most likely cause.
- Base urgency strictly on the symptom intensity described in the current query.

---

FIRST, CHECK THE CONVERSATION HISTORY — is this a NEW case or a FOLLOW-UP?
- NEW CASE: the patient's symptoms are being described for the first time.
- FOLLOW-UP: you have ALREADY given an assessment in the history, and the current
  message asks something more (e.g. "are you sure?", "what should I do?",
  "and the treatment?", "what about children?", "why?").

IF FOLLOW-UP (VERY IMPORTANT — DO NOT REPEAT YOURSELF):
- Answer ONLY the specific new thing being asked. 1–3 sentences.
- Do NOT restate the diagnosis, urgency, specialist, or action you already gave
  unless the user explicitly asks for that exact part again.
- Add NEW, specific information that moves the conversation forward.
- Never reply with the same answer as before plus one extra word. That is wrong.

IF NEW CASE, structure the answer EXACTLY like this (4 sentences max):
1) ONE sentence: most likely possible condition + one-line clinical reason.
2) ONE sentence: urgency level + why (from Urgency Reason field).
3) ONE sentence: recommended specialist + recommended non-medication action combined.
4) ONLY IF red flags exist: one sentence listing them. Otherwise skip entirely.

IN ALL CASES:
- NEVER mention record IDs, record numbers, or SYM codes.
- Speak naturally in plain flowing sentences — no bullet points.
- Every sentence must add NEW information — never repeat or rephrase earlier content.
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
