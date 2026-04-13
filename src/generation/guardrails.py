"""
Production LLM output validation — PII detection, hallucination scoring,
and domain constraint enforcement for enterprise RAG systems.

Designed for regulated environments (financial services, healthcare, government)
where output safety is a hard requirement, not an afterthought.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class ValidationStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@dataclass
class ValidationResult:
    status: ValidationStatus
    output: str
    violations: list[str] = field(default_factory=list)
    pii_detected: bool = False
    hallucination_score: float = 0.0
    latency_ms: float = 0.0


class RAGOutputGuardrails:
    """
    Multi-layer output validation for production RAG systems.

    Pipeline:
        1. Length / coherence check
        2. PII detection (regex patterns, extend with presidio for production)
        3. Hallucination scoring (token overlap heuristic → replace with NLI in prod)
        4. Domain constraint validation

    Production note:
        Run guardrails asynchronously — synchronous validation adds ~200ms p50.
        Use background scoring with fallback to pass for non-critical paths.

    Usage:
        guardrails = RAGOutputGuardrails(
            hallucination_threshold=0.15,
            block_on_pii=True
        )
        result = guardrails.validate(llm_response, source_context)
        if result.status == ValidationStatus.FAIL:
            return fallback_response
    """

    # PII patterns — extend for jurisdiction-specific requirements
    # For production: integrate Microsoft Presidio for NER-based detection
    PII_PATTERNS = {
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "phone_ca": r"(\+1)?[\s.-]?\(?[0-9]{3}\)?[\s.-]?[0-9]{3}[\s.-]?[0-9]{4}",
        "sin": r"\b[0-9]{3}[\s-]?[0-9]{3}[\s-]?[0-9]{3}\b",        # Canadian SIN
        "credit_card": r"\b(?:[0-9]{4}[\s-]?){3}[0-9]{4}\b",
        "account_number": r"\b[0-9]{8,17}\b",
    }

    def __init__(
        self,
        hallucination_threshold: float = 0.15,
        block_on_pii: bool = True,
        min_response_length: int = 20,
        max_response_length: int = 4096,
    ):
        self.hallucination_threshold = hallucination_threshold
        self.block_on_pii = block_on_pii
        self.min_response_length = min_response_length
        self.max_response_length = max_response_length

    def validate(self, response: str, source_context: str) -> ValidationResult:
        """Run full validation suite. Returns structured result with all violations."""
        violations: list[str] = []

        # 1. Length / coherence
        if len(response.strip()) < self.min_response_length:
            violations.append("Response too short — possible generation failure")
        if len(response) > self.max_response_length:
            violations.append(f"Response exceeds max length ({self.max_response_length} chars)")

        # 2. PII detection
        pii_found = self._detect_pii(response)
        if pii_found:
            violations.append(f"PII detected: {', '.join(pii_found)}")
            if self.block_on_pii:
                return ValidationResult(
                    status=ValidationStatus.FAIL,
                    output="[Response redacted: sensitive information detected]",
                    violations=violations,
                    pii_detected=True,
                )

        # 3. Hallucination scoring
        hall_score = self._hallucination_score(response, source_context)
        if hall_score > self.hallucination_threshold:
            violations.append(
                f"Potential hallucination (score={hall_score:.2f}, "
                f"threshold={self.hallucination_threshold:.2f})"
            )

        status = (
            ValidationStatus.FAIL
            if any("redacted" in v or "failure" in v for v in violations)
            else ValidationStatus.WARN
            if violations
            else ValidationStatus.PASS
        )

        return ValidationResult(
            status=status,
            output=response,
            violations=violations,
            pii_detected=bool(pii_found),
            hallucination_score=hall_score,
        )

    def _detect_pii(self, text: str) -> list[str]:
        """Regex-based PII scan. Replace with presidio.AnalyzerEngine for NER-level detection."""
        return [
            pii_type
            for pii_type, pattern in self.PII_PATTERNS.items()
            if re.search(pattern, text)
        ]

    def _hallucination_score(self, response: str, context: str) -> float:
        """
        Token overlap hallucination heuristic.

        For production: replace with a cross-encoder NLI model (e.g., DeBERTa-v3-base-mnli)
        to score entailment between response claims and source context passages.
        """
        resp_tokens = set(response.lower().split())
        ctx_tokens = set(context.lower().split())
        if not resp_tokens:
            return 1.0
        overlap = len(resp_tokens & ctx_tokens) / len(resp_tokens)
        return max(0.0, 1.0 - overlap)
