from typing import Tuple, List, Set, Iterable

Argument = Tuple[int, int]

QUESTION_FIELDS = ['wh', 'subj', 'obj', 'aux', 'prep', 'obj2', 'is_passive', 'is_negated']


class Question:
    def __init__(self, **kwargs):
        self.text = kwargs['text']
        self.wh = kwargs['wh'].lower()
        self.subj =kwargs['subj']
        self.obj = kwargs['obj']
        self.aux = kwargs['aux']
        self.prep = kwargs['prep']
        self.obj2 = kwargs['obj2']
        self.is_passive = kwargs['is_passive']
        self.is_negated = kwargs['is_negated']

    def __str__(self):
        return self.text

    def __lt__(self, other):
        return self.text < other.text

    def __hash__(self):
        return hash(self.text)

    def __eq__(self, other):
        return self.text == other.text



class Role:
    def __init__(self, question: Question, arguments: Iterable[Tuple[Argument, ...]]):
        self.question = question
        self.arguments = tuple(arguments)

    def text(self):
        return self.question.text

    def __lt__(self, other: 'Role'):
        return self.question < other.question

    def __repr__(self):
        return f"{self.text()} ==> { ' / '.join(str(a) for a in self.arguments)}"





