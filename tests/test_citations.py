from docguard.citations import quote_in_text, validate_answer
from docguard.schemas import Answer, Citation


def test_quote_matching_tolerates_whitespace_and_case(docs):
    assert quote_in_text("seis   FUNCIONES: Gobernar", docs[0].page_content)
    assert not quote_in_text("el CSF tiene doce funciones obligatorias por ley", docs[0].page_content)


def test_valid_answer_passes(docs):
    a = Answer(answer="Son seis.", answerable=True, confidence="alta",
               citations=[Citation(source="csf.pdf", page=3, quote="seis Funciones: Gobernar, Identificar")])
    assert validate_answer(a, docs) == []


def test_wrong_page_and_fabricated_quote_fail(docs):
    a = Answer(answer="x", answerable=True, confidence="alta", citations=[
        Citation(source="csf.pdf", page=99, quote="seis Funciones"),
        Citation(source="csf.pdf", page=3, quote="texto que no existe en el documento original"),
    ])
    assert len(validate_answer(a, docs)) == 2


def test_answer_without_citations_fails(docs):
    a = Answer(answer="x", answerable=True, confidence="alta")
    assert validate_answer(a, docs)


def test_unanswerable_needs_no_citations(docs):
    a = Answer(answer="No está en las fuentes.", answerable=False, confidence="alta")
    assert validate_answer(a, docs) == []


def test_long_quote_is_truncated_and_still_verifiable(docs):
    long_quote = docs[0].page_content * 4
    c = Citation(source="csf.pdf", page=3, quote=long_quote)
    assert len(c.quote) <= 300
    a = Answer(answer="Son seis.", answerable=True, confidence="alta", citations=[c])
    assert validate_answer(a, docs) == []
