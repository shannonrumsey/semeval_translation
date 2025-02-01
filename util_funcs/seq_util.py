#moving this into a function because there are issues with its implementations in the other files
def find_max_sequence_length(dataset, entity= False): # if entity == True, it will return the max entitry length
    """
    in order to set the proper size for max seq length for our positional embeddings
    """
    if entity:
        if dataset.entity_ids != None:
            longest_entity = max(len(ids) for ids in dataset.entity_ids)
            return longest_entity
    else:
        """
        for ids in dataset.corpus_encoder_ids:
            print(len(ids))
        """
        encoder_max = max(len(ids) for ids in dataset.corpus_encoder_ids)
        decoder_max = max(len(ids) for ids in dataset.corpus_decoder_ids)
        target_max = max(len(ids) for ids in dataset.corpus_target_ids)
        return max(encoder_max, decoder_max, target_max)