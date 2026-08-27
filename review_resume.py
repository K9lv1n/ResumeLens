from pathlib import Path

import torch
from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
    BitsAndBytesConfig,
)

from resume_parser import extract_pdf_text


MODEL_NAME = "Qwen/Qwen3.5-4B"

RESUME_PATH = Path("inputs/resume.pdf")
JOB_PATH = Path("inputs/job_description.txt")
OUTPUT_PATH = Path("outputs/review.txt")


def load_job_description(job_path: Path) -> str:
    if not job_path.exists():
        raise FileNotFoundError(
            f"Job description not found: {job_path}"
        )

    job_description = job_path.read_text(
        encoding="utf-8"
    ).strip()

    if not job_description:
        raise ValueError(
            "Job description is empty."
        )

    return job_description


def load_model():
    print("Loading Qwen3.5 processor...")

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME
    )

    print("Preparing 4-bit quantization...")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading Qwen3.5-4B...")

    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    return processor, model


def build_messages(
    resume_text: str,
    job_description: str,
) -> list:
    system_prompt = """
You are ResumeLens, an AI resume review assistant.

Your job is to compare a candidate's resume against a job description
and provide useful, evidence-based career advice.

Rules:

1. Only use information actually present in the resume.
2. Never invent work experience, skills, achievements, qualifications,
   responsibilities, certifications, or numerical metrics.
3. Clearly distinguish between confirmed experience and missing evidence.
4. If the candidate may have relevant experience that is not shown,
   ask them about it instead of assuming it exists.
5. Evaluate both technical skills and soft skills.
6. Explain why each recommendation matters for the target job.
7. Focus on helping the candidate improve and tailor their resume.
"""

    user_prompt = f"""
Compare the following resume against the target job description.

====================
RESUME
====================

{resume_text}

====================
JOB DESCRIPTION
====================

{job_description}

====================
REVIEW TASK
====================

Provide the following sections:

1. Overall Suitability
Give a concise assessment of how well the candidate fits the position.

2. Strong Matches
Identify job requirements that are clearly demonstrated by the resume.
For every match, provide evidence from the resume.

3. Missing or Weak Requirements
Identify requirements that are absent or not strongly demonstrated.

4. Technical Skills Analysis
Evaluate the candidate's technical skills relative to the job.

5. Experience and Projects
Identify the most relevant experience and projects.

6. Soft Skills
Only identify soft skills when there is evidence in the resume.
Explain the evidence.

7. Resume Quality
Identify weak bullet points, vague wording, missing impact,
or areas that could communicate the candidate's value better.

8. Potential Missing Information
Identify information that could strengthen the application but cannot
be safely inferred from the resume.

9. Questions for the Candidate
Ask useful questions that could uncover genuine additional experience,
achievements, teamwork, leadership, metrics, or responsibilities.

10. Recommendations
Give specific actions for tailoring the resume to this particular role.

Do not fabricate information.
"""

    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt,
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_prompt,
                }
            ],
        },
    ]


def generate_review(
    processor,
    model,
    messages: list,
) -> str:
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(model.device)

    print("Analysing resume...\n")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1500,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
        )

    input_length = inputs["input_ids"].shape[1]

    generated_tokens = outputs[0][
        input_length:
    ]

    review = processor.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return review.strip()


def save_review(
    review: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        review,
        encoding="utf-8",
    )


def main():
    print("=" * 60)
    print("ResumeLens Resume Review")
    print("=" * 60)

    print("Extracting resume...")

    resume_text = extract_pdf_text(
        RESUME_PATH
    )

    print("Loading job description...")

    job_description = load_job_description(
        JOB_PATH
    )

    processor, model = load_model()

    messages = build_messages(
        resume_text,
        job_description,
    )

    review = generate_review(
        processor,
        model,
        messages,
    )

    save_review(
        review,
        OUTPUT_PATH,
    )

    print()
    print("=" * 60)
    print("RESUMELENS REVIEW")
    print("=" * 60)
    print(review)

    print()
    print("=" * 60)
    print(f"Review saved to: {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()