import time
from .config import API_KEY, MODEL_NAME, TEMPERATURE, REQUEST_PAUSE_SECONDS


class GeminiTextClient:
    def __init__(self):
        if not API_KEY or API_KEY == "YOUR_API_KEY_HERE":
            raise RuntimeError(
                "GEMINI_API_KEY is missing. Copy .env.example to .env and add your key."
            )
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "Gemini SDK is missing. Install project dependencies with: pip install -r requirements.txt"
            ) from exc
        self.client = genai.Client(api_key=API_KEY)
        self.types = types
        self.model_name = MODEL_NAME

    def generate(self, prompt, json_mode=False, temperature=None, max_retries=7):
        temp = TEMPERATURE if temperature is None else temperature
        config_kwargs = {"temperature": temp}
        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"
        config = self.types.GenerateContentConfig(**config_kwargs)

        last_error = None
        for attempt in range(max_retries):
            started = time.time()
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )
                elapsed = time.time() - started
                output = (getattr(response, "text", None) or "").strip()
                if not output:
                    raise RuntimeError(
                        "The API returned no text. The response may have been blocked "
                        "or may have had no candidate text."
                    )

                usage = getattr(response, "usage_metadata", None)
                prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
                output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
                total_tokens = int(
                    getattr(usage, "total_token_count", 0)
                    or (prompt_tokens + output_tokens)
                )
                metadata = {
                    "model": self.model_name,
                    "temperature": temp,
                    "latency_seconds": round(elapsed, 4),
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                }
                if REQUEST_PAUSE_SECONDS > 0:
                    time.sleep(REQUEST_PAUSE_SECONDS)
                return output, metadata

            except Exception as exc:
                last_error = exc
                if attempt == max_retries - 1:
                    break
                wait = min(2 ** (attempt + 1), 30)
                print(f"API attempt {attempt + 1} failed: {exc}. Retrying in {wait}s.")
                time.sleep(wait)

        raise RuntimeError(f"Gemini API call failed after retries: {last_error}") from last_error
