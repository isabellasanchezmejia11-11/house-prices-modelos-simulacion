import argparse
import json
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--input", type=Path, default=Path(__file__).parent / "data/example_request.json")
    args = parser.parse_args()
    try:
        with urlopen(args.url.rstrip("/") + "/health", timeout=30) as response:
            print("/health:", response.read().decode())
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        request = Request(args.url.rstrip("/") + "/predict", data=json.dumps(payload, allow_nan=False).encode(),
                          headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=60) as response:
            print("/predict:", response.status, response.read().decode())
    except HTTPError as error:
        print(f"HTTP {error.code}: {error.read().decode()}", file=sys.stderr)
        sys.exit(1)
    except (OSError, URLError, ValueError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
