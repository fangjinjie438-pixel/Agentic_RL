"""Frozen CPU BM25 retrieval and a Search-R1-compatible HTTP service."""

import argparse
import collections
import hashlib
import json
import math
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


_TOKEN = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*|[\u4e00-\u9fff]")


def _tokens(text):
    return _TOKEN.findall(text.lower())


class BM25Retriever:
    def __init__(self, documents, k1=1.5, b=0.75):
        if (isinstance(k1, bool) or not isinstance(k1, (int, float)) or
                not math.isfinite(k1) or k1 <= 0):
            raise ValueError("BM25 k1 must be finite and positive")
        if (isinstance(b, bool) or not isinstance(b, (int, float)) or
                not math.isfinite(b) or not 0 <= b <= 1):
            raise ValueError("BM25 b must be finite and between 0 and 1")
        snapshots = [json.loads(json.dumps(doc, ensure_ascii=False)) for doc in documents]
        for doc in snapshots:
            # The index accepts only the public corpus schema. In particular,
            # accidentally passing task records must never publish their labels.
            if not isinstance(doc, dict) or set(doc) - {"id", "contents", "metadata"}:
                raise ValueError("Corpus records may contain only id, contents and public metadata")
            if not isinstance(doc.get("id"), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", doc["id"]):
                raise ValueError("Every corpus document needs a valid string ID")
            if not isinstance(doc.get("contents"), str):
                raise ValueError("Every corpus document needs text contents")
            metadata = doc.get("metadata", {})
            if (not isinstance(metadata, dict) or set(metadata) - {"kind", "family_id"} or
                    any(not isinstance(value, str) for value in metadata.values())):
                raise ValueError("Public metadata may contain only string kind and family_id fields")
        self._documents = tuple(snapshots)
        ids = [doc["id"] for doc in self._documents]
        if len(ids) != len(set(ids)):
            raise ValueError("Corpus document IDs must be unique")
        self.k1, self.b = k1, b
        self.term_counts = [collections.Counter(_tokens(doc["contents"])) for doc in self._documents]
        self.lengths = [sum(counts.values()) for counts in self.term_counts]
        self.average_length = sum(self.lengths) / len(self.lengths) if self.lengths else 1.0
        self.postings = collections.defaultdict(list)
        for index, counts in enumerate(self.term_counts):
            for token, frequency in counts.items():
                self.postings[token].append((index, frequency))
        canonical = json.dumps(sorted(self._documents, key=lambda doc: doc["id"]), ensure_ascii=False,
                               sort_keys=True, separators=(",", ":"))
        self.corpus_sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @property
    def documents(self):
        """Return a snapshot so consumers cannot mutate the frozen index."""
        return json.loads(json.dumps(self._documents, ensure_ascii=False))

    @property
    def document_count(self):
        return len(self._documents)

    @classmethod
    def from_jsonl(cls, path):
        paths = [path] if isinstance(path, (str, Path)) else path
        documents = []
        for corpus_path in paths:
            with Path(corpus_path).open(encoding="utf-8") as source:
                documents.extend(json.loads(line) for line in source if line.strip())
        return cls(documents)

    def search(self, query, topk=2):
        if not isinstance(query, str) or len(query) > 2000:
            raise ValueError("Query must be a string of at most 2000 characters")
        if isinstance(topk, bool) or not isinstance(topk, int) or not 1 <= topk <= 20:
            raise ValueError("topk must be an integer from 1 to 20")
        scores = collections.defaultdict(float)
        total = len(self._documents)
        # Stable summation order matters across processes with different hash seeds.
        for token in sorted(set(_tokens(query))):
            postings = self.postings.get(token, ())
            inverse = math.log(1 + (total - len(postings) + 0.5) / (len(postings) + 0.5))
            for index, frequency in postings:
                norm = self.k1 * (1 - self.b + self.b * self.lengths[index] / self.average_length)
                scores[index] += inverse * frequency * (self.k1 + 1) / (frequency + norm)
        ranked = sorted(scores, key=lambda index: (-scores[index], self._documents[index]["id"]))[:topk]
        return [dict(json.loads(json.dumps(self._documents[index], ensure_ascii=False)), score=scores[index])
                for index in ranked if scores[index] > 0]


def create_server(retriever, host="127.0.0.1", port=8000):
    """Construct a service (port=0 is supported for ephemeral test servers)."""
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise ValueError("Port must be an integer from 0 to 65535")

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, data):
            payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path != "/health":
                return self._send(404, {"error": "not_found"})
            self._send(200, {"status": "ok", "retriever": "bm25", "document_count": retriever.document_count,
                             "corpus_sha256": retriever.corpus_sha256})

        def do_POST(self):
            if self.path != "/retrieve":
                return self._send(404, {"error": "not_found"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1_000_000:
                    raise ValueError("Body must contain 1 to 1000000 bytes")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("Request body must be a JSON object")
                queries, topk = payload.get("queries"), payload.get("topk", 2)
                if not isinstance(queries, list) or not 1 <= len(queries) <= 128:
                    raise ValueError("queries must contain 1 to 128 strings")
                results = []
                for query in queries:
                    found = retriever.search(query, topk)
                    results.append([{"document": {key: value for key, value in doc.items() if key != "score"},
                                     "score": doc["score"]} for doc in found])
                self._send(200, {"result": results})
            except (ValueError, TypeError, UnicodeDecodeError) as error:
                self._send(400, {"error": str(error)})

        def log_message(self, format, *args):
            return

    return ThreadingHTTPServer((host, port), Handler)


def _port(value):
    try:
        port = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Port must be an integer") from error
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("Port must be between 1 and 65535")
    return port


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, nargs="+", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=_port, default=8000)
    args = parser.parse_args()
    retriever = BM25Retriever.from_jsonl(args.corpus)
    server = create_server(retriever, args.host, args.port)
    print(json.dumps({"listening": f"http://{args.host}:{args.port}", "document_count": retriever.document_count,
                      "corpus_sha256": retriever.corpus_sha256}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
