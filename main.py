"""Entry point of the hosted demo (Hugging Face Space): the same app as `python -m app.app`.

The Space runs this file from the repository root, so `app` and `src` are importable as packages.
"""

from app.app import demo, ui

demo.launch(theme=ui.THEME, css=ui.CSS)
