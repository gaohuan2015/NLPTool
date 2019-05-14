import snorkel
import os
from snorkel import SnorkelSession
from snorkel.parser import TSVDocPreprocessor
from snorkel.parser.spacy_parser import Spacy
from snorkel.parser import CorpusParser
from snorkel.models import Document, Sentence
from snorkel.models import candidate_subclass
from snorkel.candidates import Ngrams, CandidateExtractor
from snorkel.matchers import PersonMatcher
from snorkel.models import Document

def number_of_people(sentence):
    active_sequence = False
    count = 0
    for tag in sentence.ner_tags:
        if tag == 'PERSON' and not active_sequence:
            active_sequence = True
            count += 1
        elif tag != 'PERSON' and active_sequence:
            active_sequence = False
    return count

session = SnorkelSession()
doc_preprocessor = TSVDocPreprocessor('Snorkel/Data/articles.tsv', max_docs=500)
corpus_parser = CorpusParser(parser=Spacy())
corpus_parser.apply(doc_preprocessor, count=500)
print("Documents:", session.query(Document).count())
print("Sentences:", session.query(Sentence).count())
Spouse = candidate_subclass('Spouse', ['person1', 'person2'])
ngrams         = Ngrams(n_max=7)
person_matcher = PersonMatcher(longest_match_only=True)
cand_extractor = CandidateExtractor(Spouse, [ngrams, ngrams], [person_matcher, person_matcher])
docs = session.query(Document).order_by(Document.name).all()
train_sents = set()
dev_sents   = set()
test_sents  = set()
for i, doc in enumerate(docs):
    for s in doc.sentences:
        if number_of_people(s) <= 5:
            if i % 10 == 8:
                dev_sents.add(s)
            elif i % 10 == 9:
                test_sents.add(s)
            else:
                train_sents.add(s)
for i, sents in enumerate([train_sents, dev_sents, test_sents]):
    cand_extractor.apply(sents, split=i)
    print("Number of candidates:", session.query(Spouse).filter(Spouse.split == i).count())