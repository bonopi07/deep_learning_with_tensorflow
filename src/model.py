import analysis
import network as net
import numpy as np
import os
import tensorflow as tf
import time


class KISNet:
    def __init__(self, model_num, train_data, eval_data, model_dic):
        '''
            init deep learning environment variable
        '''
        self.gpu_num = model_dic['gpu_num']
        self.model_num = model_num
        self.model_snapshot_name = model_dic['model_storage']

        self.input_layer_size = model_dic['net_input_size']
        self.output_layer_size = model_dic['net_output_size']
        self.network_type = model_dic['net_type']

        self.train_flag = False

        '''
            init deep learning hyper parameter
        '''
        self.keep_prob = model_dic['keep_prob']
        self.train_learning_rate = model_dic['learning_rate']
        self.train_batch_size = train_data.get_batch()
        self.train_epoch = model_dic['epoch']


        '''
            init data
        '''
        self.train_data = train_data
        self.eval_data = eval_data
        pass

    def get_model_snapshot_path(self):
        # create model storage
        model_storage = self.model_snapshot_name + str(self.model_num)
        if not os.path.isdir(model_storage):
            os.makedirs(model_storage)

        return os.path.normpath(os.path.abspath('./{}/model.ckpt'.format(model_storage)))

    def inference(self, x, prob=1.0):
        if self.network_type == 'ANN':
            return net.inference_ANN(x, prob, self.train_flag)
        elif self.network_type == 'CNN':
            return net.inference_CNN(x, prob, self.train_flag)
        else:
            raise NotImplementedError
        pass

    def train(self):
        print('training start')
        self.train_flag = True

        # design network architecture
        with tf.device('/gpu:{}'.format(self.gpu_num)):
            tf.reset_default_graph()

            x = tf.placeholder(tf.float32, shape=[None, self.input_layer_size])
            y = tf.placeholder(tf.float32, shape=[None, self.output_layer_size])
            prob = tf.placeholder(tf.float32)

            y_ = self.inference(x, prob)

            # loss function: softmax, cross-entropy
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_, labels=y))

            # optimizer: Adaptive momentum optimizer
            optimizer = tf.train.AdamOptimizer(self.train_learning_rate).minimize(cost)

            # predict
            prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

        # create model snapshot
        model_path = self.get_model_snapshot_path()

        # training session start
        keep_prob = self.keep_prob
        model_saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True

        with tf.Session(config=tf_config) as sess:
            sess.run(init)

            number_of_data = len(self.train_data)
            print('file # : {}'.format(number_of_data))

            # train_epoch * number_of_data = batch_size * iteration
            train_time = time.time()
            total_iteration = (self.train_epoch*number_of_data)//self.train_batch_size
            for iteration, (train_data, train_label) in enumerate(self.train_data):
                if iteration >= total_iteration:
                    break

                _cost, _, _acc = sess.run([cost, optimizer, accuracy],
                                          feed_dict={x: train_data, y: train_label, prob: keep_prob})
                if iteration % 50 == 0:
                    print('[{i}/{total}] cost: {cost} / acc: {acc} / elapsed time: {time}'.format(
                        i=iteration, total=total_iteration, cost=_cost, acc=_acc, time=time.time()-train_time
                    ))
                if iteration % 100 == 0:
                    model_saver.save(sess, model_path)

                iteration += 1
            train_time = time.time() - train_time
            model_saver.save(sess, model_path)
        print('training time : {}'.format(train_time))
        print('------training finish------')
        pass

    def evaluate(self):
        print('evaluating start')
        self.train_flag = False

        # design network architecture
        with tf.device('/gpu:{}'.format(self.gpu_num)):
            tf.reset_default_graph()

            x = tf.placeholder(tf.float32, shape=[None, self.input_layer_size])
            y_ = self.inference(x)

        # restore model snapshot
        model_path = self.get_model_snapshot_path()

        # evaluating session start
        model_saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        tf_config = tf.ConfigProto(allow_soft_placement=True)

        with tf.Session(config=tf_config) as sess:
            sess.run(init)
            model_saver.restore(sess, model_path)

            answer_cnt = 0
            actual_labels, pred_labels = list(), list()

            number_of_data = len(self.eval_data)
            print('file # : {}'.format(number_of_data))

            eval_time = time.time()
            for iteration, (eval_data, eval_label) in enumerate(self.eval_data):
                try:
                    pred = sess.run(y_, feed_dict={x: eval_data})
                    pred_label = np.array(pred).reshape(-1).argmax(-1)  # 1 if malware else 0
                    actual_label = np.array(eval_label).argmax(-1)

                    if pred_label == actual_label:
                        answer_cnt += 1
                    if iteration % 1000 == 0:
                        print('[{i}/{total}] acc: {acc} / elapsed time: {time}'.format(
                            i=iteration, total=number_of_data, acc=(answer_cnt/(iteration+1)), time=time.time()-eval_time
                        ))
                except Exception as e:
                    print(e)
                    pred_label = -1
                    actual_label = -1
                pred_labels.append(pred_label)
                actual_labels.append(actual_label)
            eval_time = time.time() - eval_time
        total_accuracy = float(100. * (answer_cnt / number_of_data))
        print('test time : {}'.format(eval_time))
        print('accuracy : {}'.format(total_accuracy))
        print('-----evaluating finish-----')

        # plot confusion matrix
        # with open('result{}.pickle'.format(self.model_num), 'wb') as f:
        #     _pickle.dump(file_lists, f)
        #     _pickle.dump(actual_labels, f)
        #     _pickle.dump(pred_labels, f)
        #
        # analysis.plot_confusion_matrix(actual_labels, pred_labels)

        # analyze false negative files
        # analysis.analyze_fn(file_lists, actual_labels, pred_labels, self.model_num)
        pass
