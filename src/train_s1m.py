import cv2
import sys
import time
import imageio

import tensorflow as tf
import scipy.misc as sm
import numpy as np
import scipy.io as sio

from mcnet import MCNET
from utils import *
from os import listdir, makedirs, system
from os.path import exists
from argparse import ArgumentParser
from joblib import Parallel, delayed


def main(lr, batch_size, alpha, beta, K, T, num_iter, gpu):
  data_path  = "../data/S1M/"
  trainfiles = sorted([f for f in listdir(data_path) if f.endswith(".mp4")])
  margin = 0.3
  updateD = True
  updateG = True
  iters = 0
  image_size = [240,320]
  c_dim = 3
  prefix = ("S1M_MCNET"
         + "_K="+str(K)
         + "_T="+str(T)
         + "_batch_size="+str(batch_size)
         + "_alpha="+str(alpha)
         + "_beta="+str(beta)
         + "_lr="+str(lr))

  print("\n"+prefix+"\n")
  checkpoint_dir = "../models/"+prefix+"/"
  samples_dir = "../samples/"+prefix+"/"
  summary_dir = "../logs/"+prefix+"/"

  if not exists(checkpoint_dir):
    makedirs(checkpoint_dir)
  if not exists(samples_dir):
    makedirs(samples_dir)
  if not exists(summary_dir):
    makedirs(summary_dir)

  with tf.device("/gpu:%d"%gpu[0]):
    model = MCNET(image_size=image_size, c_dim=c_dim,
                  K=K, batch_size=batch_size, T=T,
                  checkpoint_dir=checkpoint_dir)
    d_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(
        model.d_loss, var_list=model.d_vars
    )
    g_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(
        alpha*model.L_img+beta*model.L_GAN, var_list=model.g_vars
    )

  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                  log_device_placement=False,
                  gpu_options=gpu_options)) as sess:

    tf.global_variables_initializer().run()

    if model.load(sess, checkpoint_dir):
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    g_sum = tf.summary.merge([model.L_p_sum,
                              model.L_gdl_sum, model.loss_sum,
                              model.L_GAN_sum])
    d_sum = tf.summary.merge([model.d_loss_real_sum, model.d_loss_sum,
                              model.d_loss_fake_sum])
    writer = tf.summary.FileWriter(summary_dir, sess.graph)

    counter = iters+1
    start_time = time.time()

    with Parallel(n_jobs=batch_size) as parallel:
      while iters < num_iter:
        mini_batches = get_minibatches_idx(len(trainfiles),
                                           batch_size, shuffle=True)
        for _, batchidx in mini_batches:
          if len(batchidx) == batch_size:
            seq_batch = np.zeros((batch_size, image_size[0], image_size[1],
                                  K+T, c_dim), dtype="float32")
            diff_batch = np.zeros((batch_size, image_size[0], image_size[1],
                                   K-1, 1), dtype="float32")
            t0 = time.time()
            Ts = np.repeat(np.array([T]),batch_size,axis=0)
            Ks = np.repeat(np.array([K]),batch_size,axis=0)
            paths  = np.repeat(data_path, batch_size,axis=0)
            tfiles = np.array(trainfiles)[batchidx]
            output = parallel(delayed(load_s1m_data)(f, p, trainfiles, k, t)
                                                 for f, p, k, t in zip(tfiles,
                                                                       paths,
                                                                       Ks, Ts))
            for i in xrange(batch_size):
              seq_batch[i]  = output[i][0]
              diff_batch[i] = output[i][1]

            if updateD:
              _, summary_str = sess.run([d_optim, d_sum],
                                  feed_dict={model.diff_in: diff_batch,
                                             model.xt: seq_batch[:,:,:,K-1],
                                             model.target: seq_batch})
              writer.add_summary(summary_str, counter)
  
            if updateG:
              _, summary_str = sess.run([g_optim, g_sum],
                                  feed_dict={model.diff_in: diff_batch,
                                             model.xt: seq_batch[:,:,:,K-1],
                                             model.target: seq_batch})
              writer.add_summary(summary_str, counter)

            errD_fake = model.d_loss_fake.eval({model.diff_in: diff_batch,
                                                model.xt: seq_batch[:,:,:,K-1],
                                                model.target: seq_batch})
            errD_real = model.d_loss_real.eval({model.diff_in: diff_batch,
                                                model.xt: seq_batch[:,:,:,K-1],
                                                model.target: seq_batch})
            errG = model.L_GAN.eval({model.diff_in: diff_batch,
                                     model.xt: seq_batch[:,:,:,K-1],
                                     model.target: seq_batch})
 
            if errD_fake < margin or errD_real < margin:
              updateD = False
            if errD_fake > (1.-margin) or errD_real > (1.-margin):
              updateG = False
            if not updateD and not updateG:
              updateD = True
              updateG = True
  
            counter += 1
  
            print(
                "Iters: [%2d] time: %4.4f, d_loss: %.8f, L_GAN: %.8f"
                % (iters, time.time() - start_time, errD_fake+errD_real,errG)
            )
 
            if np.mod(counter, 100) == 1:
              samples = sess.run([model.G],
                                  feed_dict={model.diff_in: diff_batch,
                                             model.xt: seq_batch[:,:,:,K-1],
                                             model.target: seq_batch})[0]
              samples = np.concatenate((samples[:,:,:,0,:],
                                        seq_batch[:,:,:,K,:]), axis=0)
              print("Saving sample ...")
              save_images(samples[:,:,:,::-1], [batch_size, batch_size], 
                          samples_dir+"train_%s.png" % (iters))
            if np.mod(counter, 500) == 2:
              model.save(sess, checkpoint_dir, counter)
  
            iters += 1

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--lr", type=float, dest="lr",
                      default=0.0001, help="Base Learning Rate")
  parser.add_argument("--batch_size", type=int, dest="batch_size",
                      default=8, help="Mini-batch size")
  parser.add_argument("--alpha", type=float, dest="alpha",
                      default=1.0, help="Image loss weight")
  parser.add_argument("--beta", type=float, dest="beta",
                      default=0.001, help="GAN loss weight")
  parser.add_argument("--K", type=int, dest="K",
                      default=4, help="Number of steps to observe from the past")
  parser.add_argument("--T", type=int, dest="T",
                      default=1, help="Number of steps into the future")
  parser.add_argument("--num_iter", type=int, dest="num_iter",
                      default=150000, help="Number of iterations")
  parser.add_argument("--gpu", type=int, nargs="+", dest="gpu", required=True,
                      help="GPU device id")

  args = parser.parse_args()
  main(**vars(args))
