import numpy as np
import tensorflow as tf
import os
from PIL import Image

def func(pic):
        modelDir = "/home/tyl/Code/Cprojects/TestDemo/TestCppPython/TestPython3/model"
        sess = tf.Session()
        saver = tf.train.import_meta_graph(os.path.join(modelDir,"model.ckpt.meta"))
        saver.restore(sess, tf.train.latest_checkpoint(modelDir))
        image = Image.open(pic)
        imageArray = np.array(image)
        print("图像大小: ",imageArray.shape)

        inputX = sess.graph.get_tensor_by_name('x:0')
        inputY = sess.graph.get_tensor_by_name('y:0')
        op = sess.graph.get_tensor_by_name('op_to_store:0') 

        add_on_op = tf.multiply(op,2)
        ret = sess.run(add_on_op,{inputX:5,inputY:5}) 
        sess.close()
        print("TF模型计算得到: ", ret)
        return 1

def func2():
	m = {}
	m["person"] = [1, 2, 3, 4]
	m["chair"] =  [5, 6, 7, 8]
	return m