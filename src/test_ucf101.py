import os
import cv2
import sys
import time
import ssim
import imageio

import tensorflow as tf
import scipy.misc as sm
import scipy.io as sio
import numpy as np
import skimage.measure as measure

from mcnet import MCNET
from utils import *
from os import listdir, makedirs, system
from os.path import exists
from argparse import ArgumentParser
from skimage.draw import line_aa
from PIL import Image
from PIL import ImageDraw


def main(lr, prefix, K, T, gpu):
  data_path = "../data/UCF101/UCF-101/"
  f = open(data_path.rsplit("/",2)[0]+"/testlist01.txt","r")
  testfiles = f.readlines()
  image_size = [240,320]
  c_dim = 3
  iters = 0

  if prefix == "paper_models":
    checkpoint_dir = "../models/"+prefix+"/S1M/"
    best_model = "MCNET.model-102502"
  else:
    checkpoint_dir = "../models/"+prefix+"/"
    best_model = None # will pick last model

  with tf.device("/gpu:%d"%gpu[0]):
    model = MCNET(image_size=image_size, batch_size=1, K=K,
                  T=T, c_dim=c_dim, checkpoint_dir=checkpoint_dir,
                  is_train=False)

  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                        log_device_placement=False,
                                        gpu_options=gpu_options)) as sess:

    tf.global_variables_initializer().run()

    loaded, model_name = model.load(sess, checkpoint_dir, best_model)

    if loaded:
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed... exitting")
      return

    quant_dir = "../results/quantitative/UCF101/"+prefix+"/"
    save_path = quant_dir+"results_model="+model_name+".npz"
    if not exists(quant_dir):
      makedirs(quant_dir)


    vid_names = []
    psnr_err = np.zeros((0, T))
    ssim_err = np.zeros((0, T))
    for i in xrange(0,len(testfiles),10):
      print(" Video "+str(i)+"/"+str(len(testfiles)))

      tokens = testfiles[i].split("/")[1].split()

      testfiles[i] = testfiles[i].replace("/HandStandPushups/",
                                          "/HandstandPushups/")

      vid_path = data_path + testfiles[i].split()[0]
      vid = imageio.get_reader(vid_path,"ffmpeg")
      folder_name = vid_path.split("/")[-1].split(".")[0]
      vid_names.append(folder_name)
      vid = imageio.get_reader(vid_path, "ffmpeg")
      savedir = "../results/images/UCF101/"+prefix+"/"+str(i+1)

      seq_batch = np.zeros((1, image_size[0], image_size[1], K+T, c_dim),
                            dtype="float32")
      diff_batch = np.zeros((1, image_size[0], image_size[1], K-1, 1),
                            dtype="float32")
      for t in xrange(K+T):
        img = vid.get_data(t)[:,:,::-1]
        seq_batch[0,:,:,t] = transform(img)

      for t in xrange(1,K):
        prev = inverse_transform(seq_batch[0,:,:,t-1])*255
        prev = cv2.cvtColor(prev.astype("uint8"),cv2.COLOR_BGR2GRAY)
        next = inverse_transform(seq_batch[0,:,:,t])*255
        next = cv2.cvtColor(next.astype("uint8"),cv2.COLOR_BGR2GRAY)
        diff = next.astype("float32")-prev.astype("float32")
        diff_batch[0,:,:,t-1] = diff[:,:,None]/255.

      true_data = seq_batch[:,:,:,K:,:].copy()
      pred_data = np.zeros(true_data.shape, dtype="float32")
      xt = seq_batch[:,:,:,K-1]
      pred_data[0] = sess.run(model.G,
                              feed_dict={model.diff_in: diff_batch,
                                         model.xt: xt})

      if not os.path.exists(savedir):
        os.makedirs(savedir)

      cpsnr = np.zeros((K+T,))
      cssim = np.zeros((K+T,))
      pred_data = np.concatenate((seq_batch[:,:,:,:K],pred_data),axis=3)
      true_data = np.concatenate((seq_batch[:,:,:,:K],true_data),axis=3)
      for t in xrange(K+T):
        pred     = (inverse_transform(pred_data[0,:,:,t])*255).astype("uint8")
        target   = (inverse_transform(true_data[0,:,:,t])*255).astype("uint8")

        cpsnr[t] = measure.compare_psnr(pred,target)
        cssim[t] = ssim.compute_ssim(Image.fromarray(target),
                                     Image.fromarray(pred))

        pred = draw_frame(pred, t < K)
        target = draw_frame(target, t < K)

        cv2.imwrite(savedir+"/pred_"+"{0:04d}".format(t)+".png", pred)
        cv2.imwrite(savedir+"/gt_"+"{0:04d}".format(t)+".png", target)

      cmd1 = "rm "+savedir+"/pred.gif"
      cmd2 = ("ffmpeg -f image2 -framerate 3 -i "+savedir+
              "/pred_%04d.png "+savedir+"/pred.gif")
      cmd3 = "rm "+savedir+"/pred*.png"

      # Comment out "system(cmd3)" if you want to keep the output images
      # Otherwise only the gifs will be kept
      system(cmd1); system(cmd2); system(cmd3);

      cmd1 = "rm "+savedir+"/gt.gif"
      cmd2 = ("ffmpeg -f image2 -framerate 3 -i "+savedir+
              "/gt_%04d.png "+savedir+"/gt.gif")
      cmd3 = "rm "+savedir+"/gt*.png"

      # Comment out "system(cmd3)" if you want to keep the output images
      # Otherwise only the gifs will be kept
      system(cmd1); system(cmd2); system(cmd3);

      psnr_err = np.concatenate((psnr_err, cpsnr[None,K:]), axis=0)
      ssim_err = np.concatenate((ssim_err, cssim[None,K:]), axis=0)

    np.savez(save_path, psnr=psnr_err, ssim=ssim_err)
    print("Results saved to "+save_path)
  print("Done.")

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--lr", type=float, dest="lr",
                      default=0.001, help="Base Learning Rate")
  parser.add_argument("--prefix", type=str, dest="prefix", required=True,
                      help="Prefix for log/snapshot")
  parser.add_argument("--K", type=int, dest="K",
                      default=4, help="Number of input images")
  parser.add_argument("--T", type=int, dest="T",
                      default=7, help="Number of steps into the future")
  parser.add_argument("--gpu", type=int, nargs="+", dest="gpu", required=True,
                      help="GPU device id")

  args = parser.parse_args()
  main(**vars(args))
