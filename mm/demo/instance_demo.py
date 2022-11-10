# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import os
import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import cv2
from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)



def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    parser.add_argument(
        '--suffix', type=str, default='', help='suffix of output')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    result = inference_detector(model, args.img)
    show_result_pyplot(model, args.img, result, title='demo/result/{}_{}.jpg'.format(args.img[:-4],args.suffix),score_thr=args.score_thr)
    # test a single image
    # ct=0
    # for filename in os.listdir("./demo/val"):
    # #     print(filename)
    #     sub_file_folder=int(ct/1000+1)
    #     args.img="./demo/val/"+str(filename)
    #     result = inference_detector(model, args.img)
    #     show_result_pyplot(model, args.img, result, title='demo/result/{}/{}_{}.jpg'.format(sub_file_folder,args.img[:-4],args.suffix),score_thr=args.score_thr)
    #     ct+=1
    #     idx=list(np.unique(result['pan_results']))
    #     temp=255/len(idx)
    #     # print(np.unique(result['pan_results']))
    #     for i in range(len(result['pan_results'])):
    #         for j in range(len(result['pan_results'][i])):
    #             result['pan_results'][i][j]=idx.index(result['pan_results'][i][j])*temp
    #     # print(result)
    #     plt.imshow(result['pan_results'])
    #     plt.axis('off')
    #     plt.savefig('./demo/result/{}_{}.jpg'.format(filename[:-4],args.suffix),bbox_inches='tight',dpi=600,pad_inches=0)
    # cv2.imshow(result)
    # cv2.imwrite('{}_{}.jpg'.format(args.img[:-4],args.suffix), result)
    # show the results
    # result = inference_detector(model, args.img)
    # print(result)
    # cv2.imshow(result)
    # cv2.imwrite('{}_vis.jpg'.format(args.img[:-4]), result)
    # show_result_pyplot(model, args.img, result, title='./result/{}_vis.jpg'.format(args.img[:-4]),score_thr=args.score_thr)


async def async_main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    show_result_pyplot(model, args.img, result[0], score_thr=args.score_thr)


if __name__ == '__main__':
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)
