from question_processing import prepare_question, split_atomic_questions, strip_parasitic_phrases


def test_strip_parasitic_phrases_removes_polite_prefixes_and_fillers() -> None:
    cleaned, removed = strip_parasitic_phrases(
        "Peux-tu me dire, stp, euh quelle est la duree legale du travail ?"
    )

    assert cleaned == "quelle est la duree legale du travail ?"
    assert removed


def test_split_atomic_questions_breaks_compound_question() -> None:
    atomic = split_atomic_questions(
        "Quelle est la duree legale du travail et quelles sont les regles des conges payes ?"
    )

    assert atomic == [
        "Quelle est la duree legale du travail",
        "les regles des conges payes",
    ]


def test_prepare_question_combines_cleaning_and_atomization() -> None:
    preparation = prepare_question(
        "Peux-tu me dire si le CDD est renouvelable et quelles sont les indemnites ?"
    )

    assert preparation.cleaned_question == "si le CDD est renouvelable et quelles sont les indemnites ?"
    assert preparation.atomic_questions == [
        "si le CDD est renouvelable",
        "les indemnites",
    ]
