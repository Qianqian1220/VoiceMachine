from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    features, labels = zip(*batch)
    features_padded = pad_sequence(features, batch_first=True)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)

    input_lengths = torch.tensor([feat.shape[0] for feat in features], dtype=torch.long)
    label_lengths = torch.tensor([len(lbl) for lbl in labels], dtype=torch.long)

    return features_padded, input_lengths, labels_padded, label_lengths