import os

import cv2
import tensorflow as tf
from django_dlbz.settings import BASE_DIR

from eval_img_class.load_pb_model import LoadPbModel

img_path = os.path.join(BASE_DIR, "test_img/1.jpg")

model_path = saved_model_dir = os.path.join(BASE_DIR, "pb_model/saved_model")
resize_shape = (8000, 6000)
crop_size = (640, 640)
border = 50
show_rate = 0.3
repeat_iou = 0.05

config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
config.gpu_options.allow_growth = True
g1 = tf.Graph()
sess = tf.Session(config=config, graph=g1)
meta_graph_def_sig = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], saved_model_dir)


class DianLuBanZiEval():
    def __init__(self, sess=sess):
        self.load_pb_model = LoadPbModel(sess)


    def get_detect_result(self, img_path, resize_shape=resize_shape, crop_size=crop_size, border=border,
                          show_rate=show_rate, repeat_iou=repeat_iou):
        img_list = self.load_pb_model.read_img(img_path, resize_shape)
        croped_img_list = self.load_pb_model.crop_img(img_list, crop_size, border)
        y = self.load_pb_model.eval_img_data_list(croped_img_list)
        result_list = self.load_pb_model.pingjie_img(y, img_list[0], repeat_iou=repeat_iou, show_rate=show_rate)
        img_result = self.load_pb_model.draw_boxes(result_list,img_list[0])
        # cv2.imwrite("./pp.jpg",img_result)
        # cv2.namedWindow('img_result', cv2.WINDOW_NORMAL)
        # cv2.imshow("img_result", img_result)
        # k = cv2.waitKey(1)
        # if k==ord("n"):
        #     pass
        # cv2.waitKey(0)
        # a = 0
        return result_list,img_result


if __name__ == "__main__":
    load_pb_model = DianLuBanZiEval()
    result_list,img_result = load_pb_model.get_detect_result(img_path)

    # img_result = load_pb_model.draw_boxes(img_list, img_path)
    # cv2.imshow("img_result", img_result)
    # cv2.waitKey(0)
    a = 222
