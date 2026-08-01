"""Local MVP entry point: python backend/run_mvp.py"""
from pathlib import Path

import uvicorn

def main() -> None:
    uvicorn.run(
        "cocoa_platform.api.app:app",
        app_dir=str(Path(__file__).resolve().parent / "src"),
        host="127.0.0.1",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
