from __future__ import annotations

from markupsafe import Markup
from markdown_it import MarkdownIt


_MARKDOWN = MarkdownIt(
    "commonmark",
    {
        "html": False,
        "linkify": False,
        "typographer": False,
    },
)


def render_markdown(value: object) -> Markup:
    """Render model-produced Markdown as safe HTML for the answer page."""
    return Markup(_MARKDOWN.render(str(value or "")))
