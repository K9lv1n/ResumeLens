import torch

from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
    BitsAndBytesConfig,
)


MODEL_NAME = "Qwen/Qwen3.5-4B"


def main():
    print("Loading processor...")

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
        device_map="auto",
        dtype=torch.bfloat16,
    )

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are ResumeLens, an AI resume reviewer. "
                        "Evaluate candidates only from evidence provided."
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """
Resume:
Diploma student studying Applied AI and Analytics.

Skills:
Python, Pandas, Scikit-learn, FastAPI, Docker, Google Cloud.

Projects:
Built and deployed a machine learning API using FastAPI and Docker.

Job Description:
Looking for an AI engineering intern with Python, machine learning,
Docker, cloud deployment and Kubernetes experience.

Review the candidate for this role.
""",
                }
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(model.device)

    print("Generating response...\n")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=False,
        )

    generated_tokens = outputs[0][
        inputs["input_ids"].shape[1]:
    ]

    response = processor.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    print(response)


if __name__ == "__main__":
    main()