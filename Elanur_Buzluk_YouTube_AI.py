#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from apify_client import ApifyClient
from openai import OpenAI
import tiktoken
from sentence_transformers import SentenceTransformer


def stage_error(stage: str, message: str) -> None:
    sys.stderr.write(f"[{stage}] {message}\n")


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        stage_error("config", f"Missing required environment variable: {name}")
        sys.exit(1)
    return value


def parse_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y"}


def with_retries(func, retries: int, backoff_seconds: float, stage: str, fail_message: str):
    last_error = None
    for attempt in range(retries + 1):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(backoff_seconds * (attempt + 1))
    stage_error(stage, f"{fail_message}: {last_error}")
    sys.exit(1)


def build_apify_client(api_key: str) -> ApifyClient:
    return ApifyClient(api_key)


def run_apify_actor(
    client: ApifyClient,
    actor_id: str,
    run_input: Dict[str, Any],
    stage: str,
    retries: int,
    backoff_seconds: float,
) -> List[Dict[str, Any]]:
    def _run():
        run = client.actor(actor_id).call(run_input=run_input)
        dataset_id = run.get("defaultDatasetId")
        if not dataset_id:
            raise RuntimeError("Actor run did not return dataset id")
        items = client.dataset(dataset_id).list_items().items
        if not isinstance(items, list):
            raise RuntimeError("Actor dataset items missing or invalid")
        return items

    return with_retries(_run, retries, backoff_seconds, stage, "Apify actor failed")


def fetch_youtube_metadata(
    client: ApifyClient,
    url: str,
    use_proxy: bool,
    retries: int,
    backoff_seconds: float,
) -> Dict[str, Any]:
    input_payload = {
        "startUrls": [{"url": url}],
        "maxResults": 1,
        "proxyConfiguration": {"useApifyProxy": use_proxy},
    }
    items = run_apify_actor(
        client,
        "streamers/youtube-scraper",
        input_payload,
        "youtube",
        retries,
        backoff_seconds,
    )
    if not items:
        stage_error("youtube", "No metadata returned for URL")
        sys.exit(1)
    return items[0]


def openrouter_client(api_key: str) -> OpenAI:
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def run_llm_json(client: OpenAI, model: str, messages: List[Dict[str, str]], stage: str) -> Dict[str, Any]:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
        )
    except Exception as exc:  # noqa: BLE001
        stage_error(stage, f"LLM call failed: {exc}")
        sys.exit(1)
    try:
        content = response.choices[0].message.content
    except Exception:  # noqa: BLE001
        stage_error(stage, "LLM response missing content")
        sys.exit(1)
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        stage_error(stage, f"LLM returned non-JSON content: {exc}")
        sys.exit(1)


def extract_artist_name(client: OpenAI, metadata: Dict[str, Any]) -> str:
    title = metadata.get("title") or ""
    channel = metadata.get("channelName") or ""
    description = metadata.get("description") or ""
    system_prompt = (
        "You receive metadata of a YouTube music video. "
        "Return only the artist name responsible for the song. "
        "Output JSON with key artist_name."
    )
    user_content = json.dumps(
        {
            "title": title,
            "channel": channel,
            "description": description[:500],
        }
    )
    result = run_llm_json(
        client,
        "openai/gpt-5.2",
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "artist",
    )
    artist = result.get("artist_name")
    if not artist or not isinstance(artist, str):
        stage_error("artist", "LLM did not return artist_name")
        sys.exit(1)
    return artist.strip()


def search_snippets(
    client: ApifyClient,
    queries: List[str],
    use_proxy: bool,
    retries: int,
    backoff_seconds: float,
) -> List[Dict[str, Any]]:
    run_input = {
        "queries": [{"query": q} for q in queries],
        "maxPagesPerQuery": 1,
        "resultsPerPage": 5,
        "proxyConfiguration": {"useApifyProxy": use_proxy},
    }
    return run_apify_actor(
        client,
        "apify/google-search-scraper",
        run_input,
        "search",
        retries,
        backoff_seconds,
    )


def flatten_search_results(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for item in items:
        for key in ("organicResults", "results", "items"):
            maybe_list = item.get(key) or []
            if isinstance(maybe_list, list):
                for sub in maybe_list:
                    if isinstance(sub, dict):
                        results.append(sub)
    return results


def build_tracklist_prompt(artist: str, snippets: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    snippet_texts: List[str] = []
    for res in snippets:
        url = res.get("link") or res.get("url") or ""
        title = res.get("title") or ""
        snippet = res.get("snippet") or res.get("description") or ""
        if url or title or snippet:
            snippet_texts.append(json.dumps({"url": url, "title": title, "snippet": snippet}))
    evidence_block = "\n".join(snippet_texts[:6])
    user_payload = {
        "artist": artist,
        "evidence": evidence_block,
        "instruction": "Return debut studio album and ordered track list from original release.",
    }
    return [
        {
            "role": "system",
            "content": (
                "You must identify the debut studio album of the given artist and its ordered track list. "
                "Use evidence snippets if available. Output strict JSON with album_name and songs (array of strings)."
            ),
        },
        {"role": "user", "content": json.dumps(user_payload)},
    ]


def fetch_album_and_tracks(
    llm_client: OpenAI,
    apify_client: ApifyClient,
    artist: str,
    use_proxy: bool,
    retries: int,
    backoff_seconds: float,
) -> Tuple[str, List[str]]:
    queries = [
        f'"{artist}" debut studio album tracklist',
        f'"{artist}" first album track listing',
    ]
    search_items = search_snippets(apify_client, queries, use_proxy, retries, backoff_seconds)
    flattened = flatten_search_results(search_items)
    messages = build_tracklist_prompt(artist, flattened)
    result = run_llm_json(llm_client, "openai/gpt-5.2", messages, "tracklist")
    album_name = result.get("album_name")
    songs = result.get("songs")
    if not album_name or not isinstance(album_name, str):
        stage_error("tracklist", "album_name missing in LLM result")
        sys.exit(1)
    if not isinstance(songs, list) or not all(isinstance(s, str) for s in songs):
        stage_error("tracklist", "songs list missing or invalid")
        sys.exit(1)
    cleaned = [s.strip() for s in songs if s.strip()]
    if not cleaned:
        stage_error("tracklist", "No tracks returned")
        sys.exit(1)
    return album_name.strip(), cleaned


def find_genius_url(
    apify_client: ApifyClient,
    artist: str,
    song: str,
    use_proxy: bool,
    retries: int,
    backoff_seconds: float,
) -> Optional[str]:
    query = f'site:genius.com "{song}" "{artist}" lyrics'
    items = search_snippets(apify_client, [query], use_proxy, retries, backoff_seconds)
    results = flatten_search_results(items)
    for res in results:
        url = (res.get("link") or res.get("url") or "").strip()
        if not url:
            continue
        lower_url = url.lower()
        if "genius.com" in lower_url and "lyrics" in lower_url:
            return url
    return None


def scrape_lyrics(
    apify_client: ApifyClient,
    url: str,
    use_proxy: bool,
    retries: int,
    backoff_seconds: float,
) -> Optional[str]:
    page_function = """
    async function pageFunction(context) {
        const { page } = context;
        await page.waitForSelector('[data-lyrics-container="true"]', { timeout: 20000 });
        const text = await page.$$eval('[data-lyrics-container="true"]', (els) =>
            els.map((el) => el.innerText).join('\\n')
        );
        return { lyrics: text };
    }
    """
    run_input = {
        "startUrls": [{"url": url}],
        "pageFunction": page_function,
        "proxyConfiguration": {"useApifyProxy": use_proxy},
        "maxConcurrency": 1,
    }
    try:
        items = run_apify_actor(
            apify_client,
            "apify/web-scraper",
            run_input,
            "lyrics",
            retries,
            backoff_seconds,
        )
    except SystemExit:
        raise
    except Exception:
        return None
    for item in items:
        lyrics = item.get("lyrics")
        if isinstance(lyrics, str) and lyrics.strip():
            return lyrics
    return None


def normalize_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    filtered = [line for line in lines if line]
    normalized_lines = [" ".join(line.split()) for line in filtered]
    return "\n".join(normalized_lines)


def compute_metrics(lyrics: str) -> Dict[str, Any]:
    raw = lyrics.replace("\r\n", "\n")
    cleaned = normalize_text(raw)
    chars = len(cleaned)
    words = len(cleaned.split()) if cleaned else 0
    encoder = tiktoken.get_encoding("cl100k_base")
    tokens = len(encoder.encode(cleaned)) if cleaned else 0
    tokens_per_word = (tokens / words) if words else 0
    lyrics_hash = hashlib.md5(raw.encode("utf-8")).hexdigest() if raw else ""
    return {
        "lyrics_length_chars": chars,
        "lyrics_length_words": words,
        "lyrics_length_tokens": tokens,
        "tokens_per_word": tokens_per_word,
        "lyrics_hash": lyrics_hash,
    }


def build_step5_json(
    album_name: str,
    artist: str,
    songs: List[str],
    apify_client: ApifyClient,
    use_proxy: bool,
    retries: int,
    backoff_seconds: float,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "artist_name": artist,
        "album_name": album_name,
        "songs": [],
        "errors": [],
    }
    total_tokens = 0
    for song in songs:
        song_entry: Dict[str, Any] = {"name": song}
        url = find_genius_url(apify_client, artist, song, use_proxy, retries, backoff_seconds)
        if not url:
            stage_error("genius_lookup", f"Genius URL not found for {song}")
            song_entry.update(
                {
                    "lyrics_length_chars": 0,
                    "lyrics_length_words": 0,
                    "lyrics_length_tokens": 0,
                    "tokens_per_word": 0,
                    "lyrics_hash": "",
                    "genius_url": None,
                    "skipped": True,
                }
            )
            result["errors"].append(f"Genius URL not found for {song}")
            result["songs"].append(song_entry)
            continue
        lyrics = scrape_lyrics(apify_client, url, use_proxy, retries, backoff_seconds)
        if not lyrics:
            stage_error("lyrics", f"Lyrics scrape failed for {song}")
            song_entry.update(
                {
                    "lyrics_length_chars": 0,
                    "lyrics_length_words": 0,
                    "lyrics_length_tokens": 0,
                    "tokens_per_word": 0,
                    "lyrics_hash": "",
                    "genius_url": url,
                    "skipped": True,
                }
            )
            result["errors"].append(f"Lyrics scrape failed for {song}")
            result["songs"].append(song_entry)
            continue
        metrics = compute_metrics(lyrics)
        song_entry.update(
            {
                "lyrics_length_chars": metrics["lyrics_length_chars"],
                "lyrics_length_words": metrics["lyrics_length_words"],
                "lyrics_length_tokens": metrics["lyrics_length_tokens"],
                "tokens_per_word": metrics["tokens_per_word"],
                "lyrics_hash": metrics["lyrics_hash"],
                "genius_url": url,
                "skipped": False,
            }
        )
        total_tokens += metrics["lyrics_length_tokens"]
        result["songs"].append(song_entry)
    song_count = len(songs) if songs else 1
    avg_tokens = total_tokens / song_count
    result["total_tokens_all_songs"] = total_tokens
    result["avg_tokens_per_song"] = avg_tokens
    return result


def build_token_count_string(step5_json: Dict[str, Any]) -> str:
    token_counts = [str(song.get("lyrics_length_tokens", 0)) for song in step5_json.get("songs", [])]
    return ",".join(token_counts)


def embed_text(model: SentenceTransformer, text: str) -> List[float]:
    return model.encode(text, normalize_embeddings=False).tolist()


def format_embedding_vector(vec: List[float]) -> str:
    return ",".join(f"{x:.10f}" for x in vec)


def main() -> None:
    parser = argparse.ArgumentParser(description="YouTube album pipeline")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("mode", choices=["json", "hash"], help="Output mode")
    args = parser.parse_args()

    apify_key = require_env("APIFY_API_KEY")
    openrouter_key = require_env("OPENROUTER_API_KEY")

    use_proxy = parse_bool_env("USE_APIFY_PROXY", True)
    retries = int(os.getenv("APIFY_MAX_RETRIES", "2"))
    backoff_seconds = float(os.getenv("APIFY_BACKOFF_SECONDS", "2"))

    apify_client = build_apify_client(apify_key)
    llm_client = openrouter_client(openrouter_key)

    metadata = fetch_youtube_metadata(apify_client, args.url, use_proxy, retries, backoff_seconds)
    artist_name = extract_artist_name(llm_client, metadata)
    album_name, tracks = fetch_album_and_tracks(llm_client, apify_client, artist_name, use_proxy, retries, backoff_seconds)
    step5_json = build_step5_json(album_name, artist_name, tracks, apify_client, use_proxy, retries, backoff_seconds)

    if args.mode == "json":
        print(json.dumps(step5_json, ensure_ascii=False, indent=2))
        return

    token_string = build_token_count_string(step5_json)
    embedding_model = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True,
        device="cpu",
    )
    vector = embed_text(embedding_model, token_string)
    standardized = format_embedding_vector(vector)
    final_hash = hashlib.md5(standardized.encode("utf-8")).hexdigest()
    print(final_hash)


if __name__ == "__main__":
    main()
