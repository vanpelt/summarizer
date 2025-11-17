#!/usr/bin/env python3
"""
Development server for testing the browser inference page with local models.

Usage:
    python docs/dev_server.py
    python docs/dev_server.py --port 8080

Then open: http://localhost:8000/?model=gemma3-270m-synthetic-two-line-v1
"""

import argparse
import http.server
import json
import socketserver
from pathlib import Path
from urllib.parse import urlparse, parse_qs


class LocalModelHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler that serves the web app and local GGUF models."""

    def __init__(self, *args, **kwargs):
        # Set the directory to serve from (docs/)
        super().__init__(*args, directory=str(Path(__file__).parent), **kwargs)

    def do_HEAD(self):
        """Handle HEAD requests (needed for file size detection)."""
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        # Handle HEAD requests for GGUF models
        if path.startswith('/gguf/') and path.endswith('.gguf'):
            model_name = path[6:-5]
            self.serve_gguf_model_head(model_name)
            return

        # Default HEAD handling
        super().do_HEAD()

    def do_GET(self):
        """Handle GET requests."""
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        # Route 1: List available models at /gguf
        if path == '/gguf' or path == '/gguf/':
            self.serve_model_list()
            return

        # Route 2: Serve specific GGUF model at /gguf/{NAME}.gguf
        if path.startswith('/gguf/') and path.endswith('.gguf'):
            # Extract model name (everything between /gguf/ and .gguf)
            model_name = path[6:-5]  # Strip '/gguf/' and '.gguf'
            self.serve_gguf_model(model_name)
            return

        # Route 3: Serve static files (index.html, etc.)
        super().do_GET()

    def serve_model_list(self):
        """Return JSON list of available model names."""
        try:
            # Get the project root (parent of docs/)
            project_root = Path(__file__).parent.parent
            models_dir = project_root / "models" / "gguf"

            if not models_dir.exists():
                self.send_json_response({"models": []})
                return

            # Find all directories with GGUF files
            model_names = []
            for model_dir in models_dir.iterdir():
                if model_dir.is_dir():
                    # Check if it has a .gguf file
                    gguf_files = list(model_dir.glob("*.gguf"))
                    if gguf_files:
                        model_names.append(model_dir.name)

            # Sort alphabetically
            model_names.sort()

            self.send_json_response({
                "models": model_names,
                "count": len(model_names)
            })

        except Exception as e:
            self.send_error(500, f"Error listing models: {e}")

    def serve_gguf_model_head(self, model_name):
        """Handle HEAD request for GGUF model (returns headers only, no body)."""
        try:
            # Get the project root (parent of docs/)
            project_root = Path(__file__).parent.parent
            model_dir = project_root / "models" / "gguf" / model_name

            if not model_dir.exists():
                self.send_error(404, f"Model directory not found: {model_name}")
                return

            # Find the GGUF file
            gguf_files = list(model_dir.glob("*.gguf"))

            if not gguf_files:
                self.send_error(404, f"No GGUF file found in: {model_name}")
                return

            # Use the first GGUF file found
            gguf_path = gguf_files[0]

            # Get file size
            file_size = gguf_path.stat().st_size

            # Send headers only (no body)
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(file_size))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Expose-Headers", "Content-Length, Accept-Ranges")
            self.send_header("Cache-Control", "public, max-age=31536000")
            self.end_headers()

            print(f"📋 HEAD request for {gguf_path.name} ({file_size / 1024 / 1024:.1f}MB)")

        except Exception as e:
            self.send_error(500, f"Error handling HEAD request: {e}")

    def serve_gguf_model(self, model_name):
        """Serve a GGUF model file from models/gguf/{model_name}/."""
        try:
            # Get the project root (parent of docs/)
            project_root = Path(__file__).parent.parent
            model_dir = project_root / "models" / "gguf" / model_name

            if not model_dir.exists():
                self.send_error(404, f"Model directory not found: {model_name}")
                return

            # Find the GGUF file (should be named gemma3-270m-summarizer-Q4_K_M.gguf)
            gguf_files = list(model_dir.glob("*.gguf"))

            if not gguf_files:
                self.send_error(404, f"No GGUF file found in: {model_name}")
                return

            # Use the first GGUF file found
            gguf_path = gguf_files[0]

            # Get file size
            file_size = gguf_path.stat().st_size

            # Send headers
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(file_size))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Expose-Headers", "Content-Length, Accept-Ranges")
            self.send_header("Cache-Control", "public, max-age=31536000")  # Cache for 1 year
            self.end_headers()

            # Stream the file in chunks
            with open(gguf_path, 'rb') as f:
                while True:
                    chunk = f.read(8192)
                    if not chunk:
                        break
                    self.wfile.write(chunk)

            print(f"✅ Served {gguf_path.name} ({gguf_path.stat().st_size / 1024 / 1024:.1f}MB)")

        except Exception as e:
            self.send_error(500, f"Error serving model: {e}")

    def send_json_response(self, data):
        """Send a JSON response."""
        json_data = json.dumps(data, indent=2)
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.send_header("Content-Length", str(len(json_data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json_data.encode('utf-8'))


def main():
    """Start the development server."""
    parser = argparse.ArgumentParser(description="Development server for browser inference")
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on (default: 8000)"
    )
    args = parser.parse_args()

    # Print banner
    print("=" * 60)
    print("Summarizer Browser Inference - Dev Server")
    print("=" * 60)
    print(f"\nServer starting on http://localhost:{args.port}")
    print("\nRoutes:")
    print(f"  http://localhost:{args.port}/              - Web UI")
    print(f"  http://localhost:{args.port}/gguf          - List models (JSON)")
    print(f"  http://localhost:{args.port}/gguf/{{name}}.gguf - Serve model")
    print("\nExamples:")
    print(f"  http://localhost:{args.port}/?model=gemma3-270m-synthetic-two-line-v1")
    print(f"  http://localhost:{args.port}/?model=gemma3-270m-synthetic-v11")
    print("\nPress Ctrl+C to stop")
    print("=" * 60)
    print()

    # Check if models directory exists
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "models" / "gguf"
    if models_dir.exists():
        model_count = sum(1 for d in models_dir.iterdir() if d.is_dir() and list(d.glob("*.gguf")))
        print(f"📦 Found {model_count} models in ./models/gguf/")
    else:
        print("⚠️  No models directory found at ./models/gguf/")
        print("   Models will be loaded from HuggingFace")

    print()

    # Start server
    with socketserver.TCPServer(("", args.port), LocalModelHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n🛑 Server stopped")


if __name__ == "__main__":
    main()
