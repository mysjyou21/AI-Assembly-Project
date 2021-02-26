import tensorflow as tf
import sys
sys.path.append('./miscc')
from triplet import batch_triplet_loss, _positive_mask
from utils import normalize_feature

def triplet_loss(features, labels, margin=0.2):
#    ep=1e-10
#    size = tf.sqrt(tf.reduce_sum(tf.square(features), axis=1)+ep)
#    norm_features = tf.math.divide(features, tf.tile(tf.expand_dims(size, 1), (1,features.shape[1])))
#    features = normalize_feature(features)
    return batch_triplet_loss(features, labels, margin, squared=True, hard=True)

def discriminator_loss(real_logit, fake_logit):
    real_lab = tf.ones_like(real_logit)
    fake_lab = tf.zeros_like(fake_logit)

    real_loss = 1/2*tf.nn.l2_loss(real_logit-real_lab)
    fake_loss = 1/2*tf.nn.l2_loss(fake_logit-fake_lab)

#    real_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=real_lab, logits=real_logit)
#    fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=fake_lab, logits=fake_logit)

    return tf.reduce_mean(real_loss+fake_loss)

def generator_loss(fake_logit):
    fake_lab = tf.ones_like(fake_logit)

    loss = 1/2*tf.nn.l2_loss(fake_logit-fake_lab)

#    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=fake_lab, logits=fake_logit)

    return tf.reduce_mean(loss)

def cross_mean_discrepancy_loss(trans_feature, model_feature, trans_label, model_label, C, K1, K2):
    """ trans_feature: (CxK),channel """
    # CxKxchannel
    trans_feature = tf.reshape(trans_feature, [C, K1, trans_feature.shape[1]])
    model_feature = tf.reshape(model_feature, [C, K2, model_feature.shape[1]])

    # Cxchannel - mean feature of each class
    proto_trans_feature = tf.reduce_mean(trans_feature, axis=[1])
    proto_model_feature = tf.reduce_mean(model_feature, axis=[1])

    mean_discrepancy = tf.reduce_mean(tf.square(proto_trans_feature - proto_model_feature), axis=1) # reduce_sum - in the paper

    return tf.reduce_mean(mean_discrepancy), trans_feature, model_feature
