"""Shared API contracts: explicit task scope, no implicit banking-to-docs routing."""

from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HistoryTurn(StrictModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12000)


class RequestBase(StrictModel):
    question: str = Field(min_length=1, max_length=8000)
    history: list[HistoryTurn] = Field(default_factory=list, max_length=40)

    @field_validator("question")
    @classmethod
    def non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must contain non-whitespace text")
        return value.strip()


class DocsQARequest(RequestBase):
    mode: Literal["docs_qa"]
    corpus_id: Literal["multidoc2dial"] = "multidoc2dial"
    retriever: Literal["R1", "R2", "R3", "R4", "R5"] = "R1"


class BankingRequest(RequestBase):
    mode: Literal["banking_triage"]
    model_scope: Literal["handoff50"] = "handoff50"
    score_method: Literal["H1", "H2", "H3", "H4"] = "H1"


AssistRequest = Annotated[DocsQARequest | BankingRequest, Field(discriminator="mode")]


class Evidence(StrictModel):
    evidence_id: str
    doc_id: str
    domain: str
    title: str
    text: str
    score: float = Field(allow_inf_nan=False)
    unit: Literal["document", "chunk"]


class Latency(StrictModel):
    total_ms: float = Field(ge=0)
    retrieve_ms: float | None = Field(default=None, ge=0)
    classify_ms: float | None = Field(default=None, ge=0)
    generate_ms: float | None = Field(default=None, ge=0)
    ttft_ms: float | None = Field(default=None, ge=0)


class AssistResponse(StrictModel):
    request_id: str
    mode: Literal["docs_qa", "banking_triage"]
    model_scope: Literal["multidoc2dial", "handoff50"]
    artifact_version: str
    decision: Literal["retrieved", "answer", "accepted", "handoff"]
    reason: str
    answer: str | None = None
    predicted_intent: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    score: float | None = Field(default=None, allow_inf_nan=False)
    score_method: str | None = None
    cache_status: Literal["not_used", "miss", "hit", "replay"] = "not_used"
    latency: Latency

    @model_validator(mode="after")
    def consistent_scope(self):
        if self.mode == "banking_triage":
            if self.decision not in {"accepted", "handoff"}:
                raise ValueError("invalid banking decision")
            if self.model_scope != "handoff50" or self.evidence or self.citations:
                raise ValueError("banking responses require handoff50 and no document evidence")
            if self.decision == "accepted" and not self.predicted_intent:
                raise ValueError("accepted banking request requires an intent")
        elif self.model_scope != "multidoc2dial" or self.predicted_intent is not None:
            raise ValueError("Docs-QA responses must use the document scope")
        elif self.decision not in {"retrieved", "answer", "handoff"}:
            raise ValueError("invalid Docs-QA decision")
        known = {x.evidence_id for x in self.evidence}
        if any(x not in known for x in self.citations):
            raise ValueError("citation not present in supplied evidence")
        if self.decision == "answer" and (not self.answer or not self.citations):
            raise ValueError("a generated answer requires text and citations")
        return self
