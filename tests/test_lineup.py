from mcoc_vision.lineup import character_id, _is_title


def test_character_id_normalizes_english_roster_names():
    assert character_id('SPIDER-PUNK') == 'spider_punk'
    assert character_id('MISTER SINISTER') == 'mister_sinister'
    assert character_id('ANT-MAN') == 'ant_man'


def test_lineup_requires_exact_header_tokens():
    assert _is_title([{'text': 'SERIES MATCH - SET LINEUP'}])
    assert not _is_title([{'text': 'ACCEPT'}, {'text': 'VS'}])
    assert not _is_title([{'text': 'SERIES MATCH - LINEUP'}, {'text': 'SETTINGS'}])
