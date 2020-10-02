1. generate analogies for testing (~2.5k analogies with 86 unique compositions, these are essentially the test set)
2. remove compounds in those analogies from corpus
3. train hiddenrep model on the restricted corpus
4. see if the hiddenrep model with restricted corpus peforms well on the analogies wrt. normal W2V model having access on entire corpus

