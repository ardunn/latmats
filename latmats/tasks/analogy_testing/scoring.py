ok_vocab = [(w, self.vocab[w]) for w in self.index2word[:restrict_vocab]]
ok_vocab = {w.upper(): v for w, v in
            reversed(ok_vocab)} if case_insensitive else dict(ok_vocab)

sections, section = [], None
for line_no, line in enumerate(utils.smart_open(questions)):
    # TODO: use level3 BLAS (=evaluate multiple questions at once), for speed
    line = utils.to_unicode(line)
    if line.startswith(': '):
        # a new section starts => store the old section
        if section:
            sections.append(section)
            self.log_accuracy(section)
        section = {'section': line.lstrip(': ').strip(), 'correct': [],
                   'incorrect': []}
    else:
        if not section:
            raise ValueError("Missing section header before line #%i in %s" % (
            line_no, questions))
        try:
            if case_insensitive:
                a, b, c, expected = [word.upper() for word in line.split()]
            else:
                a, b, c, expected = [word for word in line.split()]
        except ValueError:
            logger.info("Skipping invalid line #%i in %s", line_no, questions)
            continue
        if a not in ok_vocab or b not in ok_vocab or c not in ok_vocab or expected not in ok_vocab:
            logger.debug("Skipping line #%i with OOV words: %s", line_no,
                         line.strip())
            continue
        original_vocab = self.vocab
        self.vocab = ok_vocab
        ignore = {a, b, c}  # input words to be ignored
        predicted = None
        # find the most likely prediction, ignoring OOV words and input words
        sims = most_similar(self, positive=[b, c], negative=[a], topn=False,
                            restrict_vocab=restrict_vocab)
        self.vocab = original_vocab
        for index in matutils.argsort(sims, reverse=True):
            predicted = self.index2word[index].upper() if case_insensitive else \
            self.index2word[index]
            if predicted in ok_vocab and predicted not in ignore:
                if predicted != expected:
                    logger.debug("%s: expected %s, predicted %s", line.strip(),
                                 expected, predicted)
                break
        if predicted == expected:
            section['correct'].append((a, b, c, expected))
        else:
            section['incorrect'].append((a, b, c, expected))
if section:
    # store the last section, too
    sections.append(section)
    self.log_accuracy(section)

total = {
    'section': 'total',
    'correct': list(chain.from_iterable(s['correct'] for s in sections)),
    'incorrect': list(chain.from_iterable(s['incorrect'] for s in sections)),
}
self.log_accuracy(total)
sections.append(total)
return sections