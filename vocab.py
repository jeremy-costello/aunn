def get_vocab(text_data, eot_token):
    letters = []
    [letters.extend(list(text_datum)) for text_datum in text_data]
    unique_letters = sorted(list(set(letters)))
    vocab_dict = {
        "id2letter": dict(),
        "letter2id": dict()
    }
    for idx, letter in enumerate(unique_letters):
        vocab_dict["id2letter"][idx] = letter
        vocab_dict["letter2id"][letter] = idx

    assert len(vocab_dict["id2letter"]) == len(vocab_dict["letter2id"])
    vocab_size = len(vocab_dict["id2letter"])

    vocab_dict["id2letter"][vocab_size] = eot_token
    vocab_dict["letter2id"][eot_token] = vocab_size

    vocab_size += 1

    return vocab_dict, vocab_size
