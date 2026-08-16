"""Chapter extraction, ffmpeg helpers, and checkpoint/concat plumbing."""

from pathlib import Path

from cathy.cli import (
    BookInfo,
    atempo_chain,
    chapter_hash,
    concat_chapters,
    epub_book,
    ffmetadata,
    html_to_chapters,
    scale_metadata,
    text_to_chapters,
    trim_edges,
)
from cathy.engines import split_sentences


class TestChapterExtraction:
    def test_text_headings_split_chapters(self):
        chapters = text_to_chapters(
            "intro line\n\n# One\n\nfirst body\n\n## Two\n\nsecond body\n"
        )
        assert chapters == [
            ("", ["intro line"]),
            ("One", ["One", "first body"]),
            ("Two", ["Two", "second body"]),
        ]

    def test_text_multiline_paragraphs_joined(self):
        chapters = text_to_chapters("a line\nwrapped here\n\nnext para")
        assert chapters == [("", ["a line wrapped here", "next para"])]

    def test_html_h1_h2_split_and_scripts_dropped(self):
        chapters = html_to_chapters(
            "<h1>One</h1><p>body</p><script>x()</script><h2>Two</h2><p>more</p>"
        )
        assert chapters == [("One", ["One", "body"]), ("Two", ["Two", "more"])]

    def test_html_sup_footnote_markers_removed(self):
        chapters = html_to_chapters("<p>Fact.<sup>12</sup> More.<sup>*</sup></p>")
        assert chapters == [("", ["Fact. More."])]
        # non-footnote superscripts survive
        chapters = html_to_chapters("<p>E = mc<sup>squared</sup></p>")
        assert chapters == [("", ["E = mc squared"])]


class TestBookMetadata:
    def test_epub_title_author_cover(self, tmp_path: Path):
        import base64

        from ebooklib import epub

        png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
            "z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        )
        book = epub.EpubBook()
        book.set_identifier("t")
        book.set_title("A Title")
        book.set_language("en")
        book.add_author("An Author")
        book.set_cover("cover.png", png, create_page=False)
        item = epub.EpubHtml(title="One", file_name="ch1.xhtml", lang="en")
        item.content = "<html><body><h1>One</h1><p>Body text.</p></body></html>"
        book.add_item(item)
        book.spine = [item]
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        path = tmp_path / "b.epub"
        epub.write_epub(str(path), book)

        info, chapters = epub_book(path)
        assert info.title == "A Title"
        assert info.author == "An Author"
        assert info.cover == png
        assert info.cover_ext == ".png"
        assert chapters == [("One", ["One", "Body text."])]

    def test_ffmetadata_global_tags(self):
        info = BookInfo(title="A=Title", author="Someone")
        text = ffmetadata([("One", 0.0, 1.0)], info)
        head = text.split("[CHAPTER]")[0]
        assert "title=A\\=Title" in head
        assert "album=A\\=Title" in head
        assert "artist=Someone" in head

    def test_ffmetadata_no_info(self):
        assert ffmetadata([("One", 0.0, 1.0)]).startswith(";FFMETADATA1\n[CHAPTER]")


class TestSplitSentences:
    def test_short_text_single_group(self):
        assert split_sentences("One. Two. Three.") == ["One. Two. Three."]

    def test_splits_at_sentence_boundary(self):
        groups = split_sentences("Aaaa aaa. Bbbb bbb. Cccc ccc.", limit=12)
        assert groups == ["Aaaa aaa.", "Bbbb bbb.", "Cccc ccc."]


class TestChapterSpec:
    def test_ranges_and_singles(self):
        from cathy.cli import parse_chapter_spec

        assert parse_chapter_spec("3-5", 10) == {3, 4, 5}
        assert parse_chapter_spec("1,4,7-9", 10) == {1, 4, 7, 8, 9}
        assert parse_chapter_spec("-3", 10) == {1, 2, 3}
        assert parse_chapter_spec("8-", 10) == {8, 9, 10}

    def test_bad_specs_exit(self):
        import pytest

        from cathy.cli import parse_chapter_spec

        for spec in ("abc", "1,,2", "1-2-3"):
            with pytest.raises(SystemExit):
                parse_chapter_spec(spec, 10)
        with pytest.raises(SystemExit):
            parse_chapter_spec("5-12", 10)


class TestDurationText:
    def test_buckets(self):
        from cathy.cli import duration_text

        assert duration_text(0.4) == "<1 min"
        assert duration_text(23.4) == "23 min"
        assert duration_text(154) == "2 h 34 min"


class TestFfmpegHelpers:
    def test_atempo_within_range(self):
        assert atempo_chain(1.5) == "atempo=1.5"

    def test_atempo_chains_above_two(self):
        assert atempo_chain(3.0) == "atempo=2,atempo=1.5"

    def test_atempo_chains_below_half(self):
        assert atempo_chain(0.4) == "atempo=0.5,atempo=0.8"

    def test_ffmetadata_escapes_special_characters(self):
        text = ffmetadata([("a=b;c#d\\e", 0.0, 1.5)])
        assert "title=a\\=b\\;c\\#d\\\\e" in text
        assert "START=0" in text and "END=1500" in text

    def test_scale_metadata_rescales_timestamps(self):
        metadata = ffmetadata([("One", 0.0, 10.0), ("Two", 10.0, 30.0)])
        scaled = scale_metadata(metadata, 2.0)
        assert "END=5000" in scaled and "START=5000" in scaled and "END=15000" in scaled

    def test_scale_metadata_passthrough(self):
        assert scale_metadata(None, 2.0) is None
        metadata = ffmetadata([("One", 0.0, 1.0)])
        assert scale_metadata(metadata, 1.0) is metadata


class TestAudioPlumbing:
    def test_trim_edges_keeps_padding(self):
        import numpy as np

        sr = 1000
        audio = np.concatenate(
            [np.zeros(500), np.ones(200), np.zeros(500)]
        ).astype(np.float32)
        trimmed = trim_edges(audio, sr)
        # 200 loud samples plus EDGE_KEEP_SECONDS (0.05 s) padding per side
        assert len(trimmed) == 200 + 2 * 50

    def test_trim_edges_all_silence_untouched(self):
        import numpy as np

        audio = np.zeros(100, dtype=np.float32)
        assert len(trim_edges(audio, 1000)) == 100

    def test_chapter_hash_tracks_settings_and_text(self):
        base = chapter_hash("kokoro|male|1.0", "One", ["text"])
        assert base == chapter_hash("kokoro|male|1.0", "One", ["text"])
        assert base != chapter_hash("kokoro|female|1.0", "One", ["text"])
        assert base != chapter_hash("kokoro|male|1.0", "One", ["other"])

    def test_concat_chapters_marks(self, tmp_path: Path):
        import numpy as np
        import soundfile as sf

        sr = 1000
        paths = []
        for i, seconds in enumerate((1.0, 2.0)):
            path = tmp_path / f"chapter-{i}.wav"
            sf.write(path, np.zeros(int(sr * seconds), dtype=np.float32), sr)
            paths.append(path)
        out = tmp_path / "book.wav"
        marks = concat_chapters(paths, ["One", "Two"], out, sr)
        assert marks == [("One", 0.0, 1.0), ("Two", 1.0, 3.0)]
        assert sf.info(out).frames == 3 * sr
