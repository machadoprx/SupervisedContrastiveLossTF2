import tensorflow as tf

# Based on https://github.com/HobbitLong/SupContrast
@tf.function
def contrastive_loss(features, labels, temperature=0.07, base_temperature=0.05):
    """
    Supervised Contrastive Loss
    The distinct batch augumentations must be concatenated in the first axis.
    Currently only supports 2 views per batch.

    Args:
        features: hidden vector of shape [bsz * n_views, ...]. 
        labels: ground truth of shape [bsz].
    Returns:
        A loss scalar.
    """

    batch_size = features.shape[0]//2
    f1, f2 = tf.split(features, [batch_size, batch_size], axis=0)
    features = tf.concat([tf.expand_dims(f1, axis=1), tf.expand_dims(f2, axis=1)], axis=1)

    if labels[0].shape[0] != batch_size:
        raise ValueError('Num of labels does not match num of features')
    mask = tf.cast(tf.equal(labels, tf.transpose(labels)),dtype=tf.float32)

    contrast_count = features.shape[1]
    contrast_feature = tf.concat(tf.unstack(features, axis=1), axis=0)  

    anchor_feature = contrast_feature
    anchor_count = contrast_count

    # compute logits
    anchor_dot_contrast = tf.math.divide(
        tf.matmul(anchor_feature, tf.transpose(contrast_feature)),
        temperature)

    # for numerical stability
    logits_max = tf.reduce_max(anchor_dot_contrast, axis=1)
    logits = anchor_dot_contrast - logits_max
    logits = tf.cast(logits, dtype=tf.float32)
    mask = tf.tile(mask, [anchor_count, contrast_count])

    indices = tf.constant([[i, i] for i in range(logits.shape[0])])
    logits_mask = tf.tensor_scatter_nd_update(tf.ones_like(logits), indices, [tf.constant(0)]*logits.shape[0])
    mask = mask * logits_mask
    mask = tf.cast(mask, dtype=tf.float32)

    # compute log_prob
    exp_logits = tf.multiply(tf.cast(tf.math.exp(logits), dtype=tf.float32),logits_mask)
    log_prob = logits - tf.cast(tf.math.log(tf.math.reduce_sum(exp_logits, axis=1)), dtype=tf.float32)
    
    # compute mean of log-likelihood over positive
    mean_log_prob_pos = tf.reduce_sum(mask * log_prob, axis=1) / tf.reduce_sum(mask, axis=1)

    # loss
    loss = -(temperature/base_temperature) * mean_log_prob_pos
    loss = tf.reduce_mean(loss)
    return loss