import time
import numpy as np
import tensorflow as tf
from numpy import linalg as LA

from jhyexps import my_KNN, my_Kmeans
from models import GAT, HeteGAT, HeteGAT_multi
from utils import process
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3"

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True


checkpt_file = 'premodel/'

# training params
batch_size = 1
nb_epochs = 2000
patience = 100
lr = 0.0003  # learning rate
l2_coef = 0.001  # weight decay
# numbers of hidden units per each attention head in each layer
hid_units = [8]
n_heads = [1, 1]  # additional entry for the output layer
residual = False
nonlinearity = tf.nn.elu
model = HeteGAT_multi

data_size = 4057
meta_size = 3

#print('Dataset: ' + dataset)
print('----- Opt. hyperparams -----')
print('lr: ' + str(lr))
print('l2_coef: ' + str(l2_coef))
print('----- Archi. hyperparams -----')
print('nb. layers: ' + str(len(hid_units)))
print('nb. units per layer: ' + str(hid_units))
print('nb. attention heads: ' + str(n_heads))
print('residual: ' + str(residual))
print('nonlinearity: ' + str(nonlinearity))
print('model: ' + str(model))


adj_list = np.load('dataDBLP/small_adj_data.npy')
sim_matirx = np.load('dataDBLP/simpath_matrix.npy')
features = np.load('dataDBLP/features.npy')
fea_list = [features, features, features]

print('features.shape:', features.shape)
print('adj_list.shape:', adj_list.shape)
print('sim_matirx.shape:', sim_matirx.shape)
print('fea_list:', np.array(fea_list).shape)

# jhy data
import scipy.io as sio
import scipy.sparse as sp

lastlabel = np.load('dataDBLP/one_hot_labels.npy')
train_data = np.load('dataDBLP/train_idx.npy')
test_data = np.load('dataDBLP/test_idx.npy')
train_label = lastlabel[train_data]
test_label = lastlabel[test_data]

train_size = train_data.shape[0]
test_size = test_data.shape[0]

print('alllabel_shape:',lastlabel.shape)
print('train_data_shape:',train_data.shape)
print('test_data_shape:',test_data.shape)
print('train_label_shape:',train_label.shape)
print('test_label_shape:',test_label.shape)
print('train_size:',train_size)
print('test_size:',test_size)

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

train_idx = np.zeros([1,train_size],dtype=int)
test_idx = np.zeros([1,test_size],dtype=int)
train_idx[0]=train_data
test_idx[0]=test_data

train_mask = sample_mask(train_idx, lastlabel.shape[0])
test_mask = sample_mask(test_idx, lastlabel.shape[0])

y_train = np.zeros(lastlabel.shape)
y_test = np.zeros(lastlabel.shape)

y_train[train_mask, :] = lastlabel[train_mask, :]#only train part has label delete others
y_test[test_mask, :] = lastlabel[test_mask, :]
print('y_train:{}, y_test:{}, train_mask:{}, test_mask:{}'.format(y_train.shape,
                                                                        y_test.shape,
                                                                        train_mask.shape,
                                                                        test_mask.shape))


nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = y_train.shape[1]

print('nb_nodes:', nb_nodes)
print('ft_size:', ft_size)
print('nb_classes:', nb_classes)

# adj = adj.todense()

# features = features[np.newaxis]
# y_train = y_train[np.newaxis]
# y_test = y_test[np.newaxis]
# train_mask = train_mask[np.newaxis]
# test_mask = test_mask[np.newaxis]

fea_list = [fea[np.newaxis] for fea in fea_list]
adj_list = [adj[np.newaxis] for adj in adj_list]
sim_list = [sim[np.newaxis] for sim in sim_matirx]
y_train = y_train[np.newaxis]
y_test = y_test[np.newaxis]
train_mask = train_mask[np.newaxis]
test_mask = test_mask[np.newaxis]

print('fea_list:{},adj_list:{},sim_list:{} ,y_train:{} ,y_test:{}, train_mask:{},  test_mask:{}'.format(np.array(fea_list).shape,
                                                                                             np.array(adj_list).shape,
																							 np.array(sim_list).shape,
                                                                                             y_train.shape,
                                                                                             y_test.shape,
                                                                                             train_mask.shape,
                                                                                             test_mask.shape))

biases_list = [process.adj_to_bias(adj, [nb_nodes], nhood=1) for adj in adj_list]

print('build graph...')
with tf.Graph().as_default():
    with tf.name_scope('input'):
        ftr_in_list = [tf.placeholder(dtype=tf.float32,
                                      shape=(batch_size, nb_nodes, ft_size),
                                      name='ftr_in_{}'.format(i))
                       for i in range(len(fea_list))]
        bias_in_list = [tf.placeholder(dtype=tf.float32,
                                       shape=(batch_size, nb_nodes, nb_nodes),
                                       name='bias_in_{}'.format(i))
                        for i in range(len(biases_list))]
        sim_in_list = [tf.placeholder(dtype=tf.float32,
                                       shape=(batch_size, nb_nodes, nb_nodes),
                                       name='sim_in_{}'.format(i))
                        for i in range(len(sim_list))]
        lbl_in = tf.placeholder(dtype=tf.int32, shape=(
            batch_size, nb_nodes, nb_classes), name='lbl_in')
        msk_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes),
                                name='msk_in')
        attn_drop = tf.placeholder(dtype=tf.float32, shape=(), name='attn_drop')
        ffd_drop = tf.placeholder(dtype=tf.float32, shape=(), name='ffd_drop')
        is_train = tf.placeholder(dtype=tf.bool, shape=(), name='is_train')
    # forward
    logits, final_embedding, att_val, nodeatt = model.inference(ftr_in_list, nb_classes, nb_nodes, is_train,
                                                       attn_drop, ffd_drop,
                                                       bias_mat_list=bias_in_list, sim_mat_list=sim_in_list,
                                                       hid_units=hid_units, n_heads=n_heads,
                                                       residual=residual, activation=nonlinearity)

    # cal masked_loss
    log_resh = tf.reshape(logits, [-1, nb_classes])
    lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
    msk_resh = tf.reshape(msk_in, [-1])
    loss = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
    accuracy = model.masked_accuracy(log_resh, lab_resh, msk_resh)
    # optimzie
    train_op = model.training(loss, lr, l2_coef)

    saver = tf.train.Saver()

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    vlss_mn = np.inf
    vacc_mx = 0.0
    curr_step = 0

    with tf.Session() as sess:
        sess.run(init_op)

        max_test_acc = -1
        max_train_acc = -1
        max_micro = -1
        max_macro = -1
        max_FMI = -1
        max_NMI = -1
        max_ARI = -1
        for epoch in range(nb_epochs):
		
            train_loss_avg = 0
            train_acc_avg = 0
            tr_step = 0
           
            tr_size = fea_list[0].shape[0]
            # ================   training    ============
            while tr_step * batch_size < tr_size:

                fd1 = {i: d[tr_step * batch_size:(tr_step + 1) * batch_size]
                       for i, d in zip(ftr_in_list, fea_list)}
                fd2 = {i: d[tr_step * batch_size:(tr_step + 1) * batch_size]
                       for i, d in zip(bias_in_list, biases_list)}
                fd3 = {i: d[tr_step * batch_size:(tr_step + 1) * batch_size]
                       for i, d in zip(sim_in_list, sim_list)}
                fd4 = {lbl_in: y_train[tr_step * batch_size:(tr_step + 1) * batch_size],
                       msk_in: train_mask[tr_step * batch_size:(tr_step + 1) * batch_size],
                       is_train: True,
                       attn_drop: 0.0,
                       ffd_drop: 0.0}
                fd = fd1
                fd.update(fd2)
                fd.update(fd3)
                fd.update(fd4)
                #print('train fd:',fd)
                _, loss_value_tr, acc_tr, att_val_train = sess.run([train_op, loss, accuracy, att_val],
                                                                   feed_dict=fd)
                train_loss_avg += loss_value_tr
                train_acc_avg += acc_tr
                tr_step += 1
            print('epoch :', epoch ,'train accuary:',train_acc_avg/tr_step)


        #saver.restore(sess, checkpt_file)
        #print('load model from : {}'.format(checkpt_file))
            ts_size = fea_list[0].shape[0]
            ts_step = 0
            ts_loss = 0.0
            ts_acc = 0.0

            while ts_step * batch_size < ts_size:
                # fd1 = {ftr_in: features[ts_step * batch_size:(ts_step + 1) * batch_size]}
                fd1 = {i: d[ts_step * batch_size:(ts_step + 1) * batch_size]
					   for i, d in zip(ftr_in_list, fea_list)}
                fd2 = {i: d[ts_step * batch_size:(ts_step + 1) * batch_size]
					   for i, d in zip(bias_in_list, biases_list)}
                fd3 = {i: d[ts_step * batch_size:(ts_step + 1) * batch_size]
                       for i, d in zip(sim_in_list, sim_list)}
                fd4 = {lbl_in: y_test[ts_step * batch_size:(ts_step + 1) * batch_size],
					   msk_in: test_mask[ts_step * batch_size:(ts_step + 1) * batch_size],
					   is_train: False,
					   attn_drop: 0.0,
					   ffd_drop: 0.0}
			
                fd = fd1
                fd.update(fd2)
                fd.update(fd3)
                fd.update(fd4)
                loss_value_ts, acc_ts, jhy_final_embedding = sess.run([loss, accuracy, final_embedding],
																	  feed_dict=fd)
                #print('jhy_final_embedding:',jhy_final_embedding.shape)
                ts_loss += loss_value_ts
                ts_acc += acc_ts
                ts_step += 1

			#print('Test loss:', ts_loss / ts_step, '; Test accuracy:', ts_acc / ts_step)
			#with open('premodel//output.txt', 'a+') as fi:
				#fi.write('Test loss:'+str(ts_loss / ts_step)+'; Test accuracy:'+str(ts_acc / ts_step)+ '\n')
			
            if (ts_acc/ts_step) > max_test_acc:
                max_test_acc = ts_acc/ts_step
                print('Test loss:', ts_loss/ts_step, '; Test accuracy:', ts_acc/ts_step)
                #saver.save(sess, checkpt_file)
                print('...................Save Model................')
                feas = np.array(jhy_final_embedding)
                np.save('premodel/feasDBLP.npy',feas)
                #print('meta-path attention:', att_val_train)
				
            xx = np.expand_dims(jhy_final_embedding, axis=0)[test_mask]
            yy = y_test[test_mask]
			
            macro, micro = my_KNN(xx, yy)
            if macro > max_macro:
                max_macro = macro
                print('at present the best macro is :', max_macro)
            if micro > max_micro:
                max_micro = micro
                print('at present the best micro is :', max_micro)
				
            NMI,ARI,FMI = my_Kmeans(xx, yy)
            if NMI > max_NMI:
                max_NMI = NMI
                print('at present the best NMI is :', max_NMI)
            if ARI > max_ARI:
                max_ARI = ARI
                print('at present the best ARI is :', max_ARI)
            if FMI > max_FMI:
                max_FMI = FMI
                print('at present the best FMI is :', max_FMI)                

        print('The best test acc is :', max_test_acc)
        print('The best test micro is :', max_micro)
        print('The best test macro is :', max_macro)
        print('The best test NMI is :', max_NMI)
        print('The best test ARI is :', max_ARI)
        print('The best test FMI is :', max_FMI)

        # print('start knn, kmean.....')
        # xx = np.expand_dims(jhy_final_embedding, axis=0)[test_mask]
        #
        # from numpy import linalg as LA
        #
        # # xx = xx / LA.norm(xx, axis=1)
        # yy = y_test[test_mask]
        #
        # print('xx: {}, yy: {}'.format(xx.shape, yy.shape))
        # from jhyexps import my_KNN, my_Kmeans#, my_TSNE, my_Linear
        #
        # my_KNN(xx, yy)
        # my_Kmeans(xx, yy)

        sess.close()
