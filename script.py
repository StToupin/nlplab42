import embeddings

emb = embeddings.EmbeddingsDictionary()

# 3.i nearest neighbour

n = 10
print("The {} nearest neighbours to the word 'geek' are: ".format(n))
print(emb.w2neighbors('geek', top_k = n))

# 3.ii analogy

def word2vec(word):
    return emb.emb[emb.dictionary[word]]

def vec2word(v):
    n_closest = emb.emb2neighbors(v, 20)[1]
    closest = [emb.words[n] for n in n_closest]
    print(closest)
    return closest

def analogy(a, a_ref, b_ref):
    if b_ref == a_ref:
        return a
    v_a = word2vec(a)
    v_a_ref = word2vec(a_ref)
    v_b_ref = word2vec(b_ref)
    v_b = v_b_ref + v_a - v_a_ref
    b_list = vec2word(v_b)
    a = a.lower()
    a_ref = a_ref.lower()
    b_ref = b_ref.lower()
    for b in b_list:
        b = b.lower()
        if b == a or b == a_ref or b == b_ref:
            continue
        if a == b + 's' or b == a + 's' or b + 's' == a_ref or b == a_ref + 's' or b + 's' == b_ref or b == b_ref + 's':
            continue
        return b
    return None

def verbose_analogy(a, a_ref, b_ref):
    b = analogy(a, a_ref, b_ref)
    print('{} is to {} what {} is to {}'.format(b, b_ref, a, a_ref))

verbose_analogy('Tokyo', 'Japan', 'Spain')
verbose_analogy('pizza', 'Italians', 'Frenchs')
verbose_analogy('windows','computers','humans')
