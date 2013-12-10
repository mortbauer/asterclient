import pytest
from collections import OrderedDict
from asterclient import variant

d = OrderedDict((
    ('a',1),
     ('b',variant.Variants({'eins':1,'zwei':2},'klasse')),
     ('c',9),
     ('d',variant.Variants({'blau':2,'rot':99},'farbe')),
))

def test_basic_variants():
    data = variant.Data(d)
    variants = list(data.multiply())
    assert len(variants) == 4
    assert (['eins','blau'],OrderedDict((('a',1),('b',1),('c',9),('d',2)))) in variants
    assert (['zwei','blau'],OrderedDict((('a',1),('b',2),('c',9),('d',2)))) in variants
    assert (['eins','rot'],OrderedDict((('a',1),('b',1),('c',9),('d',99)))) in variants
    assert (['zwei','rot'],OrderedDict((('a',1),('b',2),('c',9),('d',99)))) in variants


def test_selected_variants():
    data = variant.Data(d)
    variants = list(data.multiply(select={'farbe':['rot']}))
    assert len(variants) == 2
    assert (['eins','rot'],OrderedDict((('a',1),('b',1),('c',9),('d',99)))) in variants
    assert (['zwei','rot'],OrderedDict((('a',1),('b',2),('c',9),('d',99)))) in variants

def test_deselected_variants():
    data = variant.Data(d)
    variants = list(data.multiply(deselect={'farbe':['blau']}))
    assert len(variants) == 2
    assert (['eins','rot'],OrderedDict((('a',1),('b',1),('c',9),('d',99)))) in variants
    assert (['zwei','rot'],OrderedDict((('a',1),('b',2),('c',9),('d',99)))) in variants

def test_deselected_all():
    data = variant.Data(d)
    with pytest.raises(Exception):
        variants = list(data.multiply(deselect={'farbe':['blau','rot']}))

