from PIL import Image
import numpy as np
import tensorflow as tf
import os

class TestDemo:
    def __init__(self, model_dir):
        self.sess = tf.Session()
        saver = tf.train.import_meta_graph(os.path.join(model_dir,"model.ckpt.meta"))
        saver.restore(self.sess, tf.train.latest_checkpoint(model_dir))
        print("类初始化成功")

    def evaluate(self, pic):
        print("进入评估...")
        image = Image.open(pic)
        imageArray = np.array(image)
        print("图像大小: ", imageArray.shape)
        
        inputX = self.sess.graph.get_tensor_by_name('x:0')
        inputY = self.sess.graph.get_tensor_by_name('y:0')
        op = self.sess.graph.get_tensor_by_name('op_to_store:0') 

        add_on_op = tf.multiply(op,2)
        ret = self.sess.run(add_on_op,{inputX:5,inputY:5}) 
        self.sess.close()
        print("TF模型计算得到: ", ret)
        return 1



