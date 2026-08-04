"""
utils/markdown_render.py
=========================
Renders the AI advisory report (real Markdown: headers, bold, tables,
blockquotes — see src/ai_advisor/advisor.py) to styled HTML for display
inside the themed `.explanation-box` div.

Wrapping raw Markdown text directly inside a `<div>` (as CommonMark treats
that as an opaque HTML block) suppresses Streamlit's Markdown parsing, so
headers/bold/tables show up as literal `#`/`**`/`|` characters. Converting
to HTML explicitly avoids that and lets `.explanation-box`'s CSS style the
result consistently with the rest of the app.
"""
import markdown as _markdown


def advisory_to_html(markdown_text: str) -> str:
    """Convert an advisory report's Markdown to HTML, wrapped in .explanation-box."""
    body = _markdown.markdown(
        markdown_text or "",
        extensions=["tables", "fenced_code", "sane_lists"],
    )
    return f'<div class="explanation-box">{body}</div>'
