import tensorflow as tf
import input_data
import sys
import numpy as np
import matplotlib.pyplot  as plt
import matplotlib.font_manager as fm

tf = tf.compat.v1
tf.compat.v1.disable_eager_execution()

font = 'TIMESBD.TTF'  #字体文件，可使用自己电脑自带的其他字体文件
myfont = fm.FontProperties(fname=font)  #将所给字体文件转化为此处可以使用的格式

# 训练参数
numberlabels = 2
hiddenunits = [512,64, 1, 64,512]

lamb = 0.001  # regularization parameter
batchsize_test = 10000
learning_rate = 0.001
batch_size = 64
trainstep = (2500 * 10 // batch_size)*30

# defining weighs and initlizatinon    权重初始化
def weight_variable(shape):
    initial = tf.truncated_normal(
        shape, stddev=0.01)  #generate tensor with shape=[x,y],  stddev 表示标准差
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(
        0.01, shape=shape)  #shape=[2]     [[0.01 0.01] [0.01 0.01]]
    return tf.Variable(initial)

# defining the layers
def layers(x, W, b):
    return tf.nn.sigmoid(tf.matmul(x, W)+b)

def layers_R(x, W, b):
    return tf.nn.relu(tf.matmul(x, W)+b)

def layers_N(x, W, b):
    return tf.matmul(x, W)+b


sess = tf.Session()

LENGTH_GET = [40]
STEP = [[1,41]] 
# LENGTH_GET = [8,16,32,48,64]
# STEP = [[1,9],[1,17],[1,33],[1,49],[1,65]] 


# STEP = [[1, TIME + 1], [1, TIME + 1]]  # or Step = [10]

  # or Step = [10]

for size in range(len(LENGTH_GET)):
   
    # Size
    lx = LENGTH_GET[size]

    # Time
    ly = STEP[size][1] - STEP[size][0]

    # defining the model

    #first layer
    #weights and bias
    # encoder weight lx*ly > 100
    W_1 = weight_variable([lx * ly, hiddenunits[0]])
    b_1 = bias_variable([hiddenunits[0]])
    # 100 > 50
    W_2 = weight_variable([hiddenunits[0], hiddenunits[1]])
    b_2 = bias_variable([hiddenunits[1]])
    # 50 > 2
    W_3 = weight_variable([hiddenunits[1], hiddenunits[2]])
    b_3 = bias_variable([hiddenunits[2]])
    # decoder weight 2 > 50
    W_4 = weight_variable([hiddenunits[2], hiddenunits[3]])
    b_4 = bias_variable([hiddenunits[3]])
    # 50 > 100
    W_5 = weight_variable([hiddenunits[3], hiddenunits[4]])
    b_5 = bias_variable([hiddenunits[4]])
    # 100 > lx*ly
    W_6 = weight_variable([hiddenunits[4], lx * ly])
    b_6 = bias_variable([lx * ly])
    

    #Apply a sigmoid
    #x is input_data, y_ is the label
    x = tf.placeholder("float", shape=[None, lx * ly])   #默认是None,就是一维值，[None,3]表示列是3，行不定
    y_ = tf.placeholder("float", shape=[None, lx * ly])

    # encoder
    O1 = layers_R(x, W_1, b_1)
    O2 = layers_R(O1, W_2, b_2)
    O3 = layers(O2, W_3, b_3)

    # encoderzz
    O4 = layers_R(O3, W_4, b_4)
    O5 = layers_R(O4, W_5, b_5)
    O6 = layers(O5, W_6, b_6)
    # O4 = layers_R(O3, W_4, b_4)+O2
    # O5 = layers_R(O4, W_5, b_5)+O1
    # O6 = layers(O5, W_6, b_6)

    y_conv = O6

    #Train and Evaluate the Model

    # cost function to minimize (with L2 regularization)
    cross_entropy = tf.reduce_mean( -y_*tf.log(tf.clip_by_value(y_conv,1e-10,1.0))-(1.0-y_)*tf.log((tf.clip_by_value(1-y_conv,1e-10,1.0))))  \
                     +lamb*(tf.nn.l2_loss(W_1)+tf.nn.l2_loss(W_3) +tf.nn.l2_loss(W_2)+tf.nn.l2_loss(W_4)+tf.nn.l2_loss(W_5)+tf.nn.l2_loss(W_6))     # 定义交叉熵
    # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_ ,logits=y_conv))  \
    #                  +lamb*(tf.nn.l2_loss(W_1)+tf.nn.l2_loss(W_3) +tf.nn.l2_loss(W_2)+tf.nn.l2_loss(W_4)+tf.nn.l2_loss(W_5)+tf.nn.l2_loss(W_6))     # 定义交叉熵

    #defining the optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)  #0.0001 is learn_rate
    train_step = optimizer.minimize(cross_entropy)

    # 判断预测正确与否

    #reading the data in the directory txt
    mnist = input_data.read_data_sets(numberlabels,
                                      lx,
                                      ly,
                                      './data/',
                                      one_hot=True)
 
    print(mnist)

    print('test.images.shape', mnist.test.images.shape)
    print('test.labels.shape', mnist.test.labels.shape)
    print(
        "xxxxxxxxxxxxxxxxxxxxx Training START xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        size)

    sess.run(tf.global_variables_initializer())  # 初始化参数

    # training
    for i in range(1,trainstep+1):

        batch = mnist.train.next_batch(batch_size)

        #每一步迭代，我们都会加载100个训练样本，然后执行一次train_step，并通过feed_dict将x 和 y_张量占位符用训练训练数据替代。

        if i % 2000 == 0:

            # batch_train = mnist.train.next_batch(batchsize_test)

            train_loss = sess.run(cross_entropy,
                                      feed_dict={
                                          x: batch[0],
                                          y_: batch[0]
                                      })
            print("step, train loss:", i, train_loss)

            
        #通过feed_dict对x&y_ 进行赋值，其中x为样例内容， y_为x对应的标签
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[0]})   #0和1分别代表batch中的第一个和第二个元素

    print(
        "xxxxxxxxxxxxxxxxxxxxx Training Done xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    )

    print(
        "test loss",
        sess.run(cross_entropy,
                 feed_dict={
                     x: mnist.test.images,
                     y_: mnist.test.images
                 }))

    # saver = tf.train.Saver()
    # save_path = saver.save(sess, "./model-saved.ckpt")
    # print("Model saved in path: %s" % save_path)

    print("xxxxxxxxxxxxxxxxxxxxx Plot Data xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

    plist = ptrain = [0.0 + x*0.025 for x in range(41)]
    ptest = plist
    Ntemp = len(
        plist)  # number of different temperatures used in the simulation

    samples_per_T = int(mnist.test.num_examples / Ntemp)

    f = open('./plot1maxtree_shuffle/' + str(lx) +'_5'+ '.dat', 'w')
    ii = 0
    av_T =[]
    av_x_ALL= []
    av_y_ALL= []
    av_z_ALL= []
    for i in range(Ntemp):
        # av_z = []
        av=0.0
        for j in range(samples_per_T):
            #X[1, :]取第一行的所有列数据， X[:, 0]取所有行的第0列数据 
            res=sess.run(O3,feed_dict={x: [mnist.test.images[ii]]})  #0和1分别代表batch中的第一个和第二个元素
            av=av+res 
            ii +=1
        av=av/samples_per_T
        print(av)
        plt.scatter(plist[i],av[0][0],label="{}".format(plist[i]))
        f.write(str(plist[i])+' '+str(av[0,0])+' '+"\n")
    plt.xlabel('${p}$',fontsize=20)
    plt.ylabel('${h^*}$',fontsize=20)
    plt.savefig('site_ae_pc_fcn_'+ str(lx)+'_maxtree.pdf')
    plt.show()
    
