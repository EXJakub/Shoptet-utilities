from html_translate import TranslationOptions, build_translation_plan, render_translation_plan


def test_build_plan_for_plain_text_and_render() -> None:
    options = TranslationOptions(mode="FORCE_TEXT")
    plan = build_translation_plan("Ahoj světe", options, max_chars=100)

    assert plan.mode == "plain"
    assert plan.segments == ["Ahoj světe"]

    rendered = render_translation_plan(plan, ["Hello world"])
    assert rendered == "Hello world"


def test_build_plan_for_html_and_render() -> None:
    options = TranslationOptions(mode="FORCE_HTML")
    plan = build_translation_plan("<p>Ahoj <b>světe</b></p>", options, max_chars=100)

    assert plan.mode == "html"
    assert plan.segments == ["Ahoj ", "světe"]

    rendered = render_translation_plan(plan, ["Hello ", "world"])
    assert rendered == "<p>Hello <b>world</b></p>"


def test_build_plan_skips_url_text() -> None:
    options = TranslationOptions(mode="FORCE_TEXT", skip_urls=True)
    source = "https://example.com/produkt"
    plan = build_translation_plan(source, options, max_chars=100)

    assert plan.mode == "skip"
    assert plan.segments == []
    assert render_translation_plan(plan, []) == source
