"""
RAG (Retrieval-Augmented Generation) Service
Retrieves policy documents with full source metadata for every experiment run.

Key design principles:
- Every retrieved document must include: document_id, version, effective_date, status, is_authoritative
- Policy scenario controls WHICH documents are eligible for retrieval
- Source metadata is stored per-run for complete audit trail
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ============================================================
# POLICY SCENARIOS
# ============================================================
POLICY_SCENARIOS = {
    "current_only": {
        "description": "Only current, authoritative policy documents",
        "include_current": True,
        "include_superseded": False,
        "include_conflict": False,
        "exclude_authoritative": False,
    },
    "current_plus_superseded": {
        "description": "Current and superseded policies in retrieval pool together",
        "include_current": True,
        "include_superseded": True,
        "include_conflict": False,
        "exclude_authoritative": False,
    },
    "superseded_only": {
        "description": "Only superseded (obsolete) policy documents",
        "include_current": False,
        "include_superseded": True,
        "include_conflict": False,
        "exclude_authoritative": False,
    },
    "current_plus_conflict": {
        "description": "Current policy plus a conflicting document in the retrieval pool",
        "include_current": True,
        "include_superseded": False,
        "include_conflict": True,
        "exclude_authoritative": False,
    },
    "authoritative_missing": {
        "description": "Authoritative current policy is NOT in retrieval pool (simulates missing document)",
        "include_current": True,
        "include_superseded": True,
        "include_conflict": False,
        "exclude_authoritative": True,
    },
}


# ============================================================
# RAG SERVICE
# ============================================================
class RAGService:
    """
    Manages the policy knowledge base and retrieval for GenAI experiments.

    The RAG service is stateful per experiment configuration — it adjusts
    which documents are eligible based on the policy_scenario parameter.
    """

    COLLECTION_NAME = "genai_risk_policy_kb"

    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
    ):
        self.persist_directory = persist_directory
        self.top_k = top_k

        # Initialize ChromaDB
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._embedding_model = SentenceTransformer(embedding_model_name)

        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(f"RAG Service initialized. Collection docs: {self._collection.count()}")

    def ingest_document(self, document: Dict[str, Any]) -> str:
        """
        Ingest a policy document into the vector store.

        Args:
            document: Dict with fields:
                - document_id (str)
                - document_code (str)
                - title (str)
                - version (str)
                - effective_date (str, ISO format)
                - status (str: current/superseded/conflict)
                - is_authoritative (bool)
                - policy_category (str)
                - content (str)
                - summary (str, optional)

        Returns:
            The document_id used in the vector store.
        """
        doc_id = document["document_id"]
        content = document["content"]
        if document.get("summary"):
            indexed_text = f"{document['title']}\n\n{document['summary']}\n\n{content}"
        else:
            indexed_text = f"{document['title']}\n\n{content}"

        metadata = {
            "document_id": str(doc_id),
            "document_code": document["document_code"],
            "title": document["title"],
            "version": document["version"],
            "effective_date": document.get("effective_date", ""),
            "status": document["status"],
            "is_authoritative": str(document.get("is_authoritative", True)).lower(),
            "policy_category": document.get("policy_category", "general"),
        }

        self._collection.upsert(
            ids=[str(doc_id)],
            documents=[indexed_text],
            metadatas=[metadata],
        )
        logger.info(f"Ingested policy document: {document['document_code']} v{document['version']} [{document['status']}]")
        return str(doc_id)

    def retrieve(
        self,
        query: str,
        policy_scenario: str = "current_only",
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant policy documents for a given query and policy scenario.

        The policy_scenario determines which documents are eligible.
        The retrieval always returns full source metadata.

        Args:
            query: The query string (e.g., derived from applicant context)
            policy_scenario: Controls which documents are in scope
            top_k: Number of results to return (defaults to self.top_k)

        Returns:
            List of retrieved documents with full metadata and scores
        """
        n_results = top_k or self.top_k
        scenario_config = POLICY_SCENARIOS.get(policy_scenario, POLICY_SCENARIOS["current_only"])

        # Get all documents to apply scenario filtering
        # ChromaDB where clause for filtering
        where_conditions = self._build_where_clause(scenario_config)

        try:
            if where_conditions:
                results = self._collection.query(
                    query_texts=[query],
                    n_results=min(n_results, self._collection.count()),
                    where=where_conditions,
                    include=["documents", "metadatas", "distances"],
                )
            else:
                results = self._collection.query(
                    query_texts=[query],
                    n_results=min(n_results, self._collection.count()),
                    include=["documents", "metadatas", "distances"],
                )
        except Exception as e:
            logger.error(f"RAG retrieval error: {e}")
            return []

        # Format results with full metadata
        retrieved = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else None
                content = results["documents"][0][i] if results["documents"] else ""

                # Post-filter: remove authoritative if scenario requires it
                if (
                    scenario_config.get("exclude_authoritative")
                    and metadata.get("is_authoritative", "true").lower() == "true"
                ):
                    continue

                score = 1 - distance if distance is not None else None

                retrieved.append({
                    "document_id": metadata.get("document_id", doc_id),
                    "document_code": metadata.get("document_code", ""),
                    "title": metadata.get("title", ""),
                    "version": metadata.get("version", ""),
                    "effective_date": metadata.get("effective_date", ""),
                    "status": metadata.get("status", ""),
                    "is_authoritative": metadata.get("is_authoritative", "true").lower() == "true",
                    "policy_category": metadata.get("policy_category", ""),
                    "retrieval_rank": i + 1,
                    "retrieval_score": score,
                    "content": content,
                })

        logger.info(
            f"RAG retrieved {len(retrieved)} docs for scenario={policy_scenario}"
        )
        return retrieved

    def _build_where_clause(self, scenario_config: Dict[str, Any]) -> Optional[Dict]:
        """Build ChromaDB where clause for scenario filtering."""
        include_statuses = []
        if scenario_config.get("include_current"):
            include_statuses.append("current")
        if scenario_config.get("include_superseded"):
            include_statuses.append("superseded")
        if scenario_config.get("include_conflict"):
            include_statuses.append("conflict")

        if not include_statuses:
            return None

        if len(include_statuses) == 1:
            return {"status": {"$eq": include_statuses[0]}}
        else:
            return {"status": {"$in": include_statuses}}

    def build_context_string(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Build a formatted context string from retrieved documents for the GenAI prompt."""
        if not retrieved_docs:
            return "No policy documents were retrieved."

        sections = []
        for doc in retrieved_docs:
            status_label = (
                "[CURRENT AUTHORITATIVE POLICY]"
                if doc["is_authoritative"] and doc["status"] == "current"
                else f"[{doc['status'].upper()}]"
            )
            section = (
                f"--- Policy Document ---\n"
                f"Document ID: {doc['document_code']}\n"
                f"Title: {doc['title']}\n"
                f"Version: {doc['version']}\n"
                f"Effective Date: {doc['effective_date']}\n"
                f"Status: {status_label}\n"
                f"Category: {doc['policy_category']}\n"
                f"\n{doc['content']}\n"
            )
            sections.append(section)

        return "\n\n".join(sections)

    def get_collection_stats(self) -> Dict[str, Any]:
        """Return statistics about the knowledge base."""
        count = self._collection.count()
        return {
            "collection_name": self.COLLECTION_NAME,
            "total_documents": count,
            "persist_directory": self.persist_directory,
        }

    def clear_collection(self) -> None:
        """Clear all documents from the collection (for testing only)."""
        self._client.delete_collection(self.COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning("RAG collection cleared.")
