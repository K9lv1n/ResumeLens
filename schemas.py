import re
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


def validate_single_sentence(
    value: str,
) -> str:
    normalized = " ".join(
        value.split()
    )

    if not normalized:
        raise ValueError(
            "Explanation must not be empty."
        )

    without_abbreviations = re.sub(
        r"\b(?:e\.g|i\.e|etc|vs)\.",
        "abbreviation",
        normalized,
        flags=re.IGNORECASE,
    )
    without_abbreviations = re.sub(
        r"(?:\b[A-Z]\.){2,}",
        "acronym",
        without_abbreviations,
    )

    if re.search(
        r"[.!?][\"')\]]*\s+(?=[A-Z0-9])",
        without_abbreviations,
    ):
        raise ValueError(
            "Explanation must contain exactly one sentence."
        )

    return normalized


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )


class RequirementAssessment(StrictModel):
    requirement: str

    status: Literal[
        "matched",
        "partial",
        "missing",
    ]

    evidence_type: Literal[
        "direct",
        "inferred",
        "missing",
    ]

    evidence: list[str] = Field(
        default_factory=list,
        max_length=2,
    )

    explanation: str

    @field_validator(
        "explanation",
    )
    @classmethod
    def require_single_sentence(
        cls,
        value: str,
    ) -> str:
        return validate_single_sentence(
            value
        )

    @model_validator(
        mode="after",
    )
    def validate_missing_evidence(
        self,
    ) -> "RequirementAssessment":
        if self.status == "missing":
            if (
                self.evidence_type != "missing"
                or self.evidence
            ):
                raise ValueError(
                    "Missing requirements must use missing "
                    "evidence_type and have no evidence."
                )
        elif self.evidence_type == "missing":
            raise ValueError(
                "Matched or partial requirements cannot use "
                "a missing evidence_type."
            )
        elif not self.evidence:
            raise ValueError(
                "Matched or partial requirements must include "
                "at least one evidence item."
            )

        return self


class SoftSkillAssessment(StrictModel):
    skill: str

    status: Literal[
        "supported",
        "weak",
        "missing",
    ]

    evidence_type: Literal[
        "direct",
        "inferred",
        "missing",
    ]

    evidence: list[str] = Field(
        default_factory=list,
        max_length=2,
    )

    explanation: str

    @model_validator(
        mode="after",
    )
    def validate_missing_evidence(
        self,
    ) -> "SoftSkillAssessment":
        if self.status == "missing":
            if (
                self.evidence_type != "missing"
                or self.evidence
            ):
                raise ValueError(
                    "Missing soft skills must use missing "
                    "evidence_type and have no evidence."
                )
        elif self.evidence_type == "missing":
            raise ValueError(
                "Supported or weak soft skills cannot use "
                "a missing evidence_type."
            )
        elif not self.evidence:
            raise ValueError(
                "Supported or weak soft skills must include "
                "at least one evidence item."
            )

        return self


class ResumeIssue(StrictModel):
    section: str
    issue: str
    evidence: str
    recommendation: str


class ResumeReview(StrictModel):
    overall_fit: Literal[
        "strong_match",
        "partial_match",
        "weak_match",
    ]

    overall_summary: str

    requirements: list[
        RequirementAssessment
    ]

    technical_strengths: list[str] = Field(
        max_length=6
    )

    experience_highlights: list[str] = Field(
        max_length=4
    )

    soft_skills: list[
        SoftSkillAssessment
    ] = Field(
        max_length=4
    )

    resume_issues: list[
        ResumeIssue
    ] = Field(
        max_length=5
    )

    missing_information: list[str] = Field(
        max_length=5
    )

    candidate_questions: list[str] = Field(
        max_length=5
    )

    recommendations: list[str] = Field(
        max_length=6
    )
