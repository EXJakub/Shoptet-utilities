from validators import extract_hrefs, validate_hrefs, validate_structure


def test_extract_hrefs():
    html = '<p><a href="https://a.cz">x</a><a href="/lokalni">y</a></p>'
    assert extract_hrefs(html) == ["https://a.cz", "/lokalni"]


def test_validate_hrefs_ok():
    a = '<a href="https://a.cz">Ahoj</a>'
    b = '<a href="https://a.cz">Čau</a>'
    assert validate_hrefs(a, b)[0] is True


def test_validate_hrefs_fail():
    a = '<a href="https://a.cz">Ahoj</a>'
    b = '<a href="https://b.cz">Ahoj</a>'
    assert validate_hrefs(a, b)[0] is False


def test_validate_structure_ok():
    a = '<p class="x">A <a href="/x">B</a></p>'
    b = '<p class="x">AA <a href="/x">BB</a></p>'
    assert validate_structure(a, b)[0] is True


def test_validate_structure_fail():
    a = '<p class="x">A</p>'
    b = '<div class="x">A</div>'
    assert validate_structure(a, b)[0] is False
