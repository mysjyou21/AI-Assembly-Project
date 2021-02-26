import tensorflow as tf


def _pairwise_distances(embeddings, squared=False):
    """ calculate pairwise distances between each embeddings in a batch
        embeddings: (batch_size, embed_dim)
        squared: d(e1, e2) = |e1-e2|(True), d(e1, e2) = |e1-e2|^2(False)"""
    batch_size = tf.shape(embeddings)[0]
    # (batch_size, batch_size)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

    # |e1|^2 part
    # (batch_size, )
    square_norm = tf.diag_part(dot_product)

    # |e1-e2|^2 = |e1|^2 + |e2|^2 - 2 dot(e1,e2)
    # (batch_size, batch_size)
    distances = tf.tile(tf.expand_dims(square_norm, 1), [1, batch_size]) + tf.tile(tf.expand_dims(square_norm, 0), [batch_size, 1]) - 2 * dot_product
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # add a small epsilon to where distances==0.0 to prevent overflowing gradient
        mask = tf.to_float(tf.equal(distances, 0.0))
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        # correct the distances to 0.0
        distances = distances * (1.0 - mask)

    return distances


def _positive_mask(labels, inter=False):
    """ labels: (batch_size, )
        return (batch_size, batch_size) positive mask. each column anchor """

    # indicate same labels
    mask = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    # except exactly same labels
    same = tf.eye(tf.shape(labels)[0], dtype=tf.bool)
#    if inter:
#        inter_same = 

    # final positive mask
    mask = tf.cast(tf.logical_and(mask, tf.logical_not(same)), tf.float32)

    return mask, tf.reduce_sum(mask, axis=1)


def _negative_mask(labels):
    """ labels: (batch_size, )
        return (batch_size, batch_size) negative mask. each column anchor """
    mask = tf.cast(tf.not_equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1)), tf.float32)

    return mask, tf.reduce_sum(mask, axis=1)


def batch_triplet_loss(embeddings, labels, margin, squared=False, hard=True, inter=False):
    """ calculate batchwise triplet loss (optional)with hard negatives, max(d(a,p) - d(a,n) + margin, 0)
        embeddings: (batch_size, embed_dim)
        labels: (batch_size,)
        margin
        """

    #    embeddings = tf.reshape(embeddings, [embeddings.get_shape()[0], -1])
    pairwise_distances = _pairwise_distances(embeddings, squared)

#    if inter:
#        mask = tf.concat([tf.zeros([tf.shape(label)[0]//2, tf.shape(label)[1]], dtype=tf.int32), tf.ones([tf.shape(label)[0]//2, tf.shape(label)[1]], dtype=tf.int32)])
#        labels = labels + mask

    if hard:
        # anchor_positive_distance
        mask_anchor_positive, _ = _positive_mask(labels)

        hardest_positive_dist = tf.multiply(mask_anchor_positive, pairwise_distances)
        hardest_positive_dist = tf.reduce_max(hardest_positive_dist, axis=1, keepdims=False) # Bx1

        # anchor_negative_distance
        mask_anchor_negative, _ = _negative_mask(labels)

        max_anchor_negative_dist_value = tf.reduce_max(pairwise_distances, axis=1, keepdims=True)
        anchor_negative_dist = pairwise_distances + max_anchor_negative_dist_value * (1.0 - mask_anchor_negative) # to prevent 'non-negative' distance to be selected as minimum negative distance

        hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=False)

        triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.)
    else:
        anchor_positive_dist = tf.expand_dims(pairwise_distances, 2)
        anchor_negative_dist = tf.expand_dims(pairwise_distances, 1)

        # triplet_loss[i,j,k] = triplet loss of anchor(i), positive(j), and negative(k)
        triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

        # activate only valid triplets
        mask_anchor_positive, _ = _positive_mask(labels)
        mask_anchor_negative, _ = _negative_mask(labels)
        mask = tf.cast(tf.equal(mask_anchor_positive, mask_anchor_negative), tf.float32)
        triplet_loss = tf.multiply(mask, triplet_loss)

        triplet_loss = tf.maximum(triplet_loss, .0)

        # to compute the positive triplets
        valid_triplets = tf.cast(tf.greater(triplet_loss, 1e-16), tf.float32)
        num_positive_triplets = tf.reduce_sum(valid_triplets)
        num_valid_triplets = tf.reduce_sum(mask)
        fraction_positive_triplets = num_positive_triplets / num_valid_triplets

        triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)

    return tf.reduce_mean(triplet_loss) #, mask_anchor_positive, mask_anchor_negative, hardest_positive_dist, hardest_negative_dist, triplet_loss, pairwise_distances
