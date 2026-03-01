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


def test_build_plan_skips_pure_url_text() -> None:
    options = TranslationOptions(mode="FORCE_TEXT", skip_urls=True)
    source = "https://example.com/produkt"
    plan = build_translation_plan(source, options, max_chars=100)

    assert plan.mode == "skip"
    assert plan.segments == []
    assert render_translation_plan(plan, []) == source


def test_plain_text_with_units_is_translated_and_unit_preserved() -> None:
    options = TranslationOptions(mode="FORCE_TEXT", skip_units=True)
    source = "Ventil je vhodný pro průměr 63 mm."
    plan = build_translation_plan(source, options, max_chars=200)

    assert plan.mode == "plain"
    assert "63 mm" not in plan.segments[0]
    assert "__KEEP_" in plan.segments[0]

    rendered = render_translation_plan(plan, ["Ventil je vhodný pre priemer __KEEP_0__."])
    assert rendered == "Ventil je vhodný pre priemer 63 mm."


def test_html_text_with_units_is_not_skipped() -> None:
    options = TranslationOptions(mode="FORCE_HTML", skip_units=True)
    source = "<p>Za vířivkou je potřeba nechat volný prostor 46 cm.</p>"
    plan = build_translation_plan(source, options, max_chars=200)

    assert plan.mode == "html"
    assert len(plan.segments) == 1
    assert "46 cm" not in plan.segments[0]

    rendered = render_translation_plan(plan, ["Za vírivkou je potrebné nechať voľný priestor __KEEP_0__."])
    assert rendered == "<p>Za vírivkou je potrebné nechať voľný priestor 46 cm.</p>"
