import time
import base64

from django.shortcuts import render
from django.http import HttpResponse
import tensorflow as tf
import cv2
import numpy as np
from django.http import JsonResponse

from deal_one_model.dianlubanzi.deal_one_img import DianLuBanZiEval


# Create your views here.

def dianlubanzi(request):
    if (request.method == 'POST'):
        t0 = time.time()
        img_data = request.POST.get('image')  # 本质就是解码字符串
        tt = time.time()
        print("接收一张图片需要{}秒".format(tt - t0))
        # print(test_image)
        img_byte = base64.b64decode(img_data)
        img_np_arr = np.fromstring(img_byte, np.uint8)
        image = cv2.imdecode(img_np_arr, cv2.IMREAD_COLOR)
        t1 = time.time()
        print("解码张图片需要{}秒".format(t1 - tt))
        # cv2.imwrite("./dlbz_src.png", image)
        t2 = time.time()
        print("保存一张图片需要{}秒".format(t2 - t1))
        code = 200

        # try:
        load_pb_model_szld = DianLuBanZiEval()
        img_list_szld,img_result = load_pb_model_szld.get_detect_result(image)
            # cv2.imwrite("./dlbz_result.png", image)
            # print("img_list_szld {}".format(img_list_szld))
        # except:
        #     code = 0

        cv2.imwrite("/home/db/图片/test_dlbz/dlbz.jpg", img_result)
        # cv2.namedWindow('img_result', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('img_result', 1920, 1080)
        # cv2.imshow("img_result", img_result)
        # while 1:
        #     key = cv2.waitKey(1)
        #     if key==ord('n'):
        #         print("break!!!!!!!!!!!!!!!!!!!!!!!!")
        #         break
        #     else:
        #         pass

            # result_list = []
    #     num = 0
    #     if not result_list:
    #         for x in result_list:
    #             for y in x:
    #                 for z in y:
    #                     num += 1
    #
    #     else:
    #         num = 0
    #
    #     t3 = time.time()
    #     print("num--------------------{}".format(num))
    #     print("计算一张图片需要{}秒".format(t3 - t2))
    #     data = {
    #         'code': code,
    #         'num': num,
    #         'result': result_list,
    #     }
    #     t4 = time.time()
    #     print("处理一张图片需要{}秒".format(t4 - t0))
    # return JsonResponse(data)
    return HttpResponse("success!!!")

