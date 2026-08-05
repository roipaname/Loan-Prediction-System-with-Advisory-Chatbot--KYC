"""
Renders the AI advisory report's Markdown to HTML for the `.explanation-box`
div. Dropping raw Markdown straight into a <div> doesn't work — CommonMark
treats that as an opaque HTML block and stops parsing inside it, so
headers/bold/tables show up as literal #/**/| characters instead.
"""
import markdown as _markdown


def advisory_to_html(markdown_text: str) -> str:
    """Convert an advisory report's Markdown to HTML, wrapped in .explanation-box."""
    body = _markdown.markdown(
        markdown_text or "",
        extensions=["tables", "fenced_code", "sane_lists"],
    )
    return f'<div class="explanation-box">{body}</div>'
