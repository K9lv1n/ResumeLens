from pathlib import Path

import pymupdf


INPUT_PATH = Path("inputs/resume.pdf")
OUTPUT_PATH = Path("outputs/resume_text.txt")


def extract_pdf_text(pdf_path: Path) -> str:
    """
    Extract readable text from a PDF resume.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Extracted resume text.

    Raises:
        FileNotFoundError: If the PDF does not exist.
        ValueError: If the file is not a PDF or contains no readable text.
    """

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Resume not found: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(
            "ResumeLens currently only accepts PDF files."
        )

    pages = []

    with pymupdf.open(pdf_path) as document:
        print(f"Pages found: {len(document)}")

        for page_number, page in enumerate(
            document,
            start=1,
        ):
            text = page.get_text(
                "text",
                sort=True,
            ).strip()

            if text:
                pages.append(
                    f"--- Page {page_number} ---\n{text}"
                )

    resume_text = "\n\n".join(pages).strip()

    if not resume_text:
        raise ValueError(
            "No readable text was found in the PDF. "
            "The resume may be scanned or image-based."
        )

    return resume_text


def save_text(
    text: str,
    output_path: Path,
) -> None:
    """
    Save extracted resume text for inspection.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        text,
        encoding="utf-8",
    )


def main():
    print("=" * 60)
    print("ResumeLens PDF Parser")
    print("=" * 60)

    print(f"Reading: {INPUT_PATH}")

    resume_text = extract_pdf_text(
        INPUT_PATH
    )

    save_text(
        resume_text,
        OUTPUT_PATH,
    )

    print()
    print("=" * 60)
    print("EXTRACTED RESUME")
    print("=" * 60)
    print(resume_text)

    print()
    print("=" * 60)
    print("Extraction complete")
    print("=" * 60)

    print(
        f"Characters extracted: "
        f"{len(resume_text):,}"
    )

    print(
        f"Saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()