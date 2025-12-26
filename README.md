## YouTube Album Pipeline

Tek dosyalık CLI (`Elanur_Buzluk_YouTube_AI.py`) ile YouTube URL’inden sanatçı, ilk stüdyo albümü tracklist’i, Genius lyrics metrikleri ve final embedding hash’ini üretir.

### Kurulum
- Python 3.10+ önerilir.
- Sanal ortam açın ve bağımlılıkları kurun:
  ```bash
  python3 -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  ```
- Ortam değişkenlerini ayarlayın (`.env` örnek için `.env.example`):
  ```
  APIFY_API_KEY=...
  OPENROUTER_API_KEY=...
  USE_APIFY_PROXY=true
  APIFY_MAX_RETRIES=2
  APIFY_BACKOFF_SECONDS=2
  ```

### Çalıştırma
- Step-5 JSON için:
  ```bash
  python3 Elanur_Buzluk_YouTube_AI.py "https://www.youtube.com/watch?v=..." json
  ```
- Step-8 final hash için:
  ```bash
  python3 Elanur_Buzluk_YouTube_AI.py "https://www.youtube.com/watch?v=..." hash
  ```

### Notlar
- Zorunlu API anahtarları: `APIFY_API_KEY`, `OPENROUTER_API_KEY`.
- Kullanılan Apify actor’lar: `streamers/youtube-scraper`, `apify/google-search-scraper`, `apify/web-scraper`.
- Lyrics kaynağı yalnızca `genius.com`.
- Hash modu Step-5 JSON’dan token-count string → `nomic-ai/nomic-embed-text-v1.5` embedding → `.10f` format → MD5 üretir.
# assesment
# assesment
# assesment
