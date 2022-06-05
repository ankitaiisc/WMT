import spacy, string, random

import spacy.cli

spacy.cli.download("en_core_web_lg")
nlp = spacy.load("en_core_web_lg")

with open("pert_sent") as f:
  sent_list = [s.strip('\n') for s in f.readlines()]


stop_subj = ['who','which','what','that','where', '%']


from spacy.symbols import NOUN, PROPN, PRON
from spacy.errors import Errors


def my_noun_chunks(doclike):
    """
    Detect base noun phrases from a dependency parse. Works on both Doc and Span.
    """
    labels = [
        "nsubj",
        "dobj",
        "nsubjpass",
        "pcomp",
        "pobj",
        "dative",
        "appos",
        "attr",
        "ROOT",
    ]
    doc = doclike.doc  # Ensure works on both Doc and Span.

    np_deps = [doc.vocab.strings.add(label) for label in labels]
    conj = doc.vocab.strings.add("conj")
    np_label = doc.vocab.strings.add("NP")
    prev_end = -1
    for i, word in enumerate(doclike):
        if word.pos not in (NOUN, PROPN, PRON):
            continue
        # Prevent nested chunks from being produced
        if word.left_edge.i <= prev_end:
            continue
        if word.dep in np_deps:
            #print(word, word.dep_)
            prev_end = word.i

            left_index = 1e6
            for i in range(word.left_edge.i, word.i + 1):
              if doc[i].dep_=='compound':
                left_index = min(i, left_index)

            #no compound noun found
            if left_index==1e6:
              left_index = word.i

            #print(doc[left_index: word.i + 1])
            #print('------')
            
            yield left_index, word.i + 1, np_label

        elif word.dep == conj:
            head = word.head
            while head.dep == conj and head.head.i < head.i:
                head = head.head
            # If the head is an NP, and we're coordinated to it, we're an NP
            if head.dep in np_deps:
                prev_end = word.i
                yield word.left_edge.i, word.i + 1, np_label


def merge_phrases(doc):
    with doc.retokenize() as retokenizer:
        custom_noun_chunks = my_noun_chunks(doc)
        for np in custom_noun_chunks:
            np = doc[np[0]:np[1]]
            attrs = {
                "tag": np.root.tag_,
                "lemma": np.root.lemma_,
                "ent_type": np.root.ent_type_,
            }
            retokenizer.merge(np, attrs=attrs)
    return doc



with open("pert_sent") as f:
  sent_list = [s.strip('\n') for s in f.readlines()]


def delete_nonsubj_nonne(sent_list, nlp, stop_subj):
  "deletes the non-subj and non-ne NP head; considers compound nouns where applicable; if more than 1 subj - chooses one at random"
  
  import re
  perturbed_output =[]

  for sent in sent_list:
      doc = nlp(sent)
      doc = merge_phrases(doc) #merges tokens in compound phrases
      custom_noun_chunks = my_noun_chunks(doc) #finds nouns phrases

      allowed_nouns = []
      for np in custom_noun_chunks:
        np = doc[np[0]:np[1]]
        if np.root.dep_ not in ["nsubj", "nsubjpass"] and len(nlp(str(np)).ents)==0: #non-subj and non-NE
          allowed_nouns.append(np)

      if len(allowed_nouns) == 0:
        perturbed_output.append((sent, "replace", ''))

      elif len(allowed_nouns) == 1:
        sent = sent.replace(str(allowed_nouns[0]),'')
        sent = re.sub(r'\s([?.!,;"](?:\s|$))', r'\1', sent) #strips spaces before punctuations
        sent = sent[0].upper() + sent[1:]
        perturbed_output.append((sent, "good", str(allowed_nouns[0])))
      else:
        id = random.randint(0,len(allowed_nouns)-1)
        tok = allowed_nouns[id]
        sent = sent.replace(str(tok),'') 
        sent = re.sub(r'\s([?.!,;"](?:\s|$))', r'\1', sent) #strips spaces before punctuations
        sent = sent[0].upper() + sent[1:]
        perturbed_output.append((sent, "good", str(tok)))

  return perturbed_output

#run on test sentences
test_sentences = ['''According to the Vietnam News Agency, Grigory Karasin believes that Vietnam has made great contributions to the development of ASEAN over the past 25 years.''',
'''Chile's top climbing destination, the Cochamó Valley, known as the Yosemite of South America, is home to a wide variety of granite boulders and cliffs.''',
'''Last month, a presidential committee recommended the resignation of the former CEP as part of measures to push the country toward new elections.''',
'''Below them are more medium-sized felines that eat medium-sized prey ranging from rabbits and antelopes to deer.''',
'''The Chaco region is home to other indigenous tribes, such as the Guaycuru and Payagua, who subsist by hunting, gathering and fishing.''']

no_subj = delete_nonsubj_nonne(test_sentences, nlp, stop_subj)

import pandas as pd
df = pd.DataFrame(columns =['pert_text',	'sent_replace',	'subj_replace'])

pert_text = []
sent_replace = []
subj_replace = []

for tup in no_subj:
  pert_text.append(tup[0])
  sent_replace.append(tup[1])
  subj_replace.append(tup[2])
df['pert_text'] = pert_text
df['sent_replace'] = sent_replace
df['subj_replace'] = subj_replace
df.to_csv('./perturbed_output_id4.csv', sep='\t')