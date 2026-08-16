"""Front/back-matter classification and narration text cleanup."""

from cathy.normalize import (
    drop_front_back_matter,
    normalize_chapters,
    normalize_paragraph,
    normalize_text,
    skip_reason,
)

PROSE = ["It was a dark and stormy night." * 3] * 4


class TestSkipReason:
    def test_skips_by_title(self):
        for title in ("Contents", "TABLE OF CONTENTS", "Copyright", "Index",
                      "Title Page", "3. Bibliography"):
            assert skip_reason(5, title, PROSE) is not None, title

    def test_skips_by_title_prefix(self):
        assert skip_reason(0, "Praise for Great Book", PROSE)
        assert skip_reason(0, "Also by Jane Doe", PROSE)

    def test_keeps_normal_chapters(self):
        assert skip_reason(4, "Chapter One", PROSE) is None
        assert skip_reason(4, "The Index Card", PROSE) is None

    def test_copyright_by_content(self):
        paragraphs = ["Great Book", "All rights reserved.", "ISBN 978-0-00"]
        assert skip_reason(1, "", paragraphs) == "copyright page"
        # a long chapter mentioning an ISBN is not a copyright page
        assert skip_reason(9, "Notes on Sources", ["isbn things"] + PROSE * 4) is None

    def test_toc_by_content(self):
        toc = [f"Chapter {n}" for n in range(8)]
        assert skip_reason(1, "", toc) == "table of contents"
        # only untitled chapters near the start qualify
        assert skip_reason(5, "", toc) is None
        assert skip_reason(1, "Haiku", toc) is None

    def test_drop_front_back_matter(self):
        chapters = [("Contents", ["1", "2", "3", "4", "5"]), ("One", PROSE)]
        kept, skipped = drop_front_back_matter(chapters)
        assert kept == [("One", PROSE)]
        assert skipped == ["Contents"]


class TestNormalizeText:
    def test_footnote_markers(self):
        assert normalize_text("Fact.[12] More.[3]") == "Fact. More."

    def test_superscript_footnotes_after_punctuation(self):
        assert normalize_text("It ended.¹² Then dawn.") == "It ended. Then dawn."
        # a real exponent follows a letter, not punctuation — keep it
        assert normalize_text("about 5 m² of cloth") == "about 5 m² of cloth"

    def test_urls_become_domains(self):
        assert (
            normalize_text("See https://www.example.org/a/b?c=1. Done.")
            == "See example.org. Done."
        )
        assert normalize_text("at www.example.com, later") == "at example.com, later"

    def test_markdown_artifacts(self):
        assert normalize_text("a [link](https://x.com/y) here") == "a link here"
        assert normalize_text("drop ![alt text](img.png) images") == "drop images"
        assert normalize_text("very **bold** and *slanted* and `code`") == (
            "very bold and slanted and code"
        )

    def test_invisible_characters(self):
        assert normalize_text("co­operate​ now then") == (
            "cooperate now then"
        )

    def test_whitespace_collapsed(self):
        assert normalize_text("  a \t b\n c  ") == "a b c"


class TestNormalizeParagraph:
    def test_scene_separators_dropped(self):
        for sep in ("* * *", "***", "— — —", "• • •", "____"):
            assert normalize_paragraph(sep) == "", sep

    def test_stray_page_numbers_dropped(self):
        assert normalize_paragraph("142") == ""
        assert normalize_paragraph("142 sheep") == "142 sheep"

    def test_normalize_chapters_drops_emptied(self):
        chapters = [("Sep", ["* * *", "17"]), ("One", ["Text stays."])]
        assert normalize_chapters(chapters) == [("One", ["Text stays."])]
