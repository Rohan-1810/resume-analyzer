
import json

from groq import Groq

from resume_analyzer.schemas import JobD
from resume_analyzer.llm_utils import retry_on_rate_limit

def parse_job_description(
    job_description: str,
    client: Groq,
    model: str,
) -> JobD:
    # Generate the JSON schema from our Pydantic model
    jobd_schema = JobD.model_json_schema()

    system_prompt = f"""
    You are an expert HR assistant.

    Extract structured information from the given job description.

    Return a JSON object matching this schema:

    {json.dumps(jobd_schema, indent=2)}

    IMPORTANT:
    - Return actual extracted information, not the schema itself.
    - Do not return fields such as "properties", "title", or "type"
      as the main response.
    - If minimum experience is not mentioned, return null.
    - If information for a list is missing, return an empty list.
    - Do not invent information.
    """

    user_prompt = f"""
    Analyze the following job description:

    {job_description}
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    response = retry_on_rate_limit(
    lambda: client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        )
    )

    raw_json = response.choices[0].message.content

    if not raw_json:
        raise ValueError("The LLM returned an empty response.")

    # Convert JSON text into a Python dictionary
    job_data = json.loads(raw_json)

    # Validate the dictionary using Pydantic
    job = JobD(**job_data)

    return job
