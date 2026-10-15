
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from resume_analyzer.jd_parser import parse_job_description
from resume_analyzer.resume_parser import read_resume, parse_resume
from resume_analyzer.evaluator import final_score


# ==========================================
# 1. ENVIRONMENT AND GROQ CLIENT SETUP
# ==========================================

# Locate .env at the AIengg project root
env_path = Path(__file__).resolve().parents[4] / ".env"
load_dotenv(env_path)

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY not found. Please check your .env file."
    )

client = Groq(api_key=api_key)

MODEL = "openai/gpt-oss-120b"

print("Groq client initialized successfully!")
print("Model:", MODEL)


# ==========================================
# 2. JOB DESCRIPTION
# ==========================================

# Temporary hardcoded JD.
# We will replace this with user input later.

job_description = """
Software Engineer

Requirements:
- Knowledge of Python and SQL
- Understanding of data structures and algorithms
- Bachelor's degree in Computer Science

Responsibilities:
- Develop and maintain APIs
- Write clean and maintainable code
- Debug software issues
"""


# ==========================================
# 3. PARSE JOB DESCRIPTION
# ==========================================

job = parse_job_description(
    job_description=job_description,
    client=client,
    model=MODEL,
)

print("\nJob description parsed successfully.")
print(job.model_dump_json(indent=2))


# ==========================================
# 4. FIND RESUME FILES
# ==========================================

resume_folder = Path("resumes")

if not resume_folder.is_dir():
    raise FileNotFoundError(
        f"Resume folder not found: {resume_folder.resolve()}"
    )

resume_files = sorted(
    file_path
    for file_path in resume_folder.iterdir()
    if file_path.is_file()
    and file_path.suffix.lower() in {".pdf", ".docx"}
)

if not resume_files:
    raise ValueError(
        "No PDF or DOCX resumes found in the resumes folder."
    )

print(f"\nFound {len(resume_files)} resume(s).")


# ==========================================
# 5. PROCESS AND EVALUATE RESUMES
# ==========================================

all_results = []

for index, file_path in enumerate(resume_files):
    print(f"\n{'=' * 45}")
    print(f"Processing: {file_path.name}")
    print(f"{'=' * 45}")

    try:
        # Step A: Extract text from PDF or DOCX
        resume_text = read_resume(file_path)

        # Step B: Extract structured resume information
        parsed_resume = parse_resume(
            resume_text=resume_text,
            client=client,
            model=MODEL,
        )

        # Step C: Compare resume with the job description
        result = final_score(
            job=job,
            resume=parsed_resume,
            client=client,
            model=MODEL,
        )

        # Save the candidate's evaluation
        all_results.append({
            "name": parsed_resume.name or file_path.stem,
            "filename": file_path.name,
            "score": result.score,
            "details": result.details,
        })

        print("Score:", result.score, "/ 100")
        print("Verdict:", result.details.verdict)

    except Exception as error:
        print(f"Error processing {file_path.name}: {error}")
        print("Skipping this resume and continuing.")
        continue

    # Basic pacing between candidates
    if index < len(resume_files) - 1:
        time.sleep(2)


# ==========================================
# 6. HANDLE RESULTS
# ==========================================

if not all_results:
    print("\nNo resumes were successfully evaluated.")
    raise SystemExit(1)


# ==========================================
# 7. RANK CANDIDATES
# ==========================================

all_results.sort(
    key=lambda candidate: candidate["score"],
    reverse=True,
)


# ==========================================
# 8. DISPLAY COMPLETE RANKING
# ==========================================

print("\n========== CANDIDATE RANKING ==========")

for rank, candidate in enumerate(all_results, start=1):
    print(
        f"{rank}. {candidate['name']} - "
        f"{candidate['score']}/100 "
        f"({candidate['filename']})"
    )


# ==========================================
# 9. DISPLAY TOP TWO CANDIDATES
# ==========================================

print("\n========== TOP 2 CANDIDATES ==========")

for candidate in all_results[:2]:
    print(f"\nName: {candidate['name']}")
    print(f"Score: {candidate['score']}/100")
    print(f"Verdict: {candidate['details'].verdict}")
    print(f"Reasoning: {candidate['details'].reasoning}")


# ==========================================
# 10. DISPLAY LOWEST TWO CANDIDATES
# ==========================================

print("\n========== LOWEST 2 CANDIDATES ==========")

for candidate in all_results[-2:]:
    print(f"\nName: {candidate['name']}")
    print(f"Score: {candidate['score']}/100")
    print(f"Verdict: {candidate['details'].verdict}")
    print(f"Reasoning: {candidate['details'].reasoning}")


print("\nResume evaluation completed successfully!")
