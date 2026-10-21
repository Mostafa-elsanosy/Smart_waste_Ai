"""
Smart Waste AI & Decision Support System (DSS)
Unified Application Launcher
"""

import sys
import os
import argparse
import webbrowser
import uvicorn


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

def main():
    parser = argparse.ArgumentParser(description="Smart Waste AI & DSS Unified Launcher")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port for FastAPI backend & dashboard (default: 8000)")
    parser.add_argument("--reload", action="store_true", default=True, help="Enable auto-reload for development")
    parser.add_argument("--open-browser", action="store_true", help="Automatically open browser upon startup")

    args = parser.parse_args()

    print("\n" + "=" * 66)
    print("      SMART WASTE AI & DECISION SUPPORT SYSTEM (DSS)")
    print("=" * 66)
    print(f"[*] Web Dashboard & API : http://localhost:{args.port}")
    print(f"[*] Mobile Web Simulator: http://localhost:{args.port}/mobile")
    print(f"[*] Interactive Swagger : http://localhost:{args.port}/docs")
    print("=" * 66 + "\n")

    if args.open_browser:
        webbrowser.open(f"http://localhost:{args.port}")

    uvicorn.run("backend.app.main:app", host=args.host, port=args.port, reload=args.reload)

if __name__ == "__main__":
    main()
