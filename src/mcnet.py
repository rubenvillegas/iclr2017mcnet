import os
import tensorflow as tf

from BasicConvLSTMCell import BasicConvLSTMCell
from ops import *
from utils import *

class MCNET(object):
  def __init__(self, image_size, batch_size=32, c_dim=3,
               K=10, T=10, checkpoint_dir=None, is_train=True):

    self.batch_size = batch_size
    self.image_size = image_size
    self.is_train = is_train

    self.gf_dim = 64
    self.df_dim = 64

    self.c_dim = c_dim
    self.K = K
    self.T = T
    self.diff_shape = [batch_size, self.image_size[0],
                       self.image_size[1], K-1, 1]
    self.xt_shape = [batch_size, self.image_size[0], self.image_size[1], c_dim]
    self.target_shape = [batch_size, self.image_size[0], self.image_size[1],
                         K+T, c_dim]

    self.build_model()

  def build_model(self):
    self.diff_in = tf.placeholder(tf.float32, self.diff_shape, name='diff_in')
    self.xt = tf.placeholder(tf.float32, self.xt_shape, name='xt')
    self.target = tf.placeholder(tf.float32, self.target_shape, name='target')
    
    cell = BasicConvLSTMCell([self.image_size[0]/8, self.image_size[1]/8],
                             [3, 3], 256)
    pred = self.forward(self.diff_in, self.xt, cell)

    self.G = tf.concat(axis=3,values=pred)
    if self.is_train:
      true_sim = inverse_transform(self.target[:,:,:,self.K:,:])
      if self.c_dim == 1: true_sim = tf.tile(true_sim,[1,1,1,1,3])
      true_sim = tf.reshape(tf.transpose(true_sim,[0,3,1,2,4]),
                                         [-1, self.image_size[0],
                                          self.image_size[1], 3])
      gen_sim = inverse_transform(self.G)
      if self.c_dim == 1: gen_sim = tf.tile(gen_sim,[1,1,1,1,3])
      gen_sim = tf.reshape(tf.transpose(gen_sim,[0,3,1,2,4]),
                                        [-1, self.image_size[0],
                                        self.image_size[1], 3])
      binput = tf.reshape(self.target[:,:,:,:self.K,:],
                          [self.batch_size, self.image_size[0],
                           self.image_size[1], -1])
      btarget = tf.reshape(self.target[:,:,:,self.K:,:],
                           [self.batch_size, self.image_size[0],
                            self.image_size[1], -1])
      bgen = tf.reshape(self.G,[self.batch_size,
                                self.image_size[0],
                                self.image_size[1], -1])

      good_data = tf.concat(axis=3,values=[binput,btarget])
      gen_data  = tf.concat(axis=3,values=[binput,bgen])

      with tf.variable_scope("DIS", reuse=False):
        self.D, self.D_logits = self.discriminator(good_data)

      with tf.variable_scope("DIS", reuse=True):
        self.D_, self.D_logits_ = self.discriminator(gen_data)

      self.L_p = tf.reduce_mean(
          tf.square(self.G-self.target[:,:,:,self.K:,:])
      )
      self.L_gdl = gdl(gen_sim, true_sim, 1.)
      self.L_img = self.L_p + self.L_gdl

      self.d_loss_real = tf.reduce_mean(
          tf.nn.sigmoid_cross_entropy_with_logits(
              logits=self.D_logits, labels=tf.ones_like(self.D)
          )
      )
      self.d_loss_fake = tf.reduce_mean(
          tf.nn.sigmoid_cross_entropy_with_logits(
              logits=self.D_logits_, labels=tf.zeros_like(self.D_)
          )
      )
      self.d_loss = self.d_loss_real + self.d_loss_fake
      self.L_GAN = tf.reduce_mean(
          tf.nn.sigmoid_cross_entropy_with_logits(
              logits=self.D_logits_, labels=tf.ones_like(self.D_)
          )
      )

      self.loss_sum = tf.summary.scalar("L_img", self.L_img)
      self.L_p_sum = tf.summary.scalar("L_p", self.L_p)
      self.L_gdl_sum = tf.summary.scalar("L_gdl", self.L_gdl)
      self.L_GAN_sum = tf.summary.scalar("L_GAN", self.L_GAN)
      self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
      self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
      self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

      self.t_vars = tf.trainable_variables()
      self.g_vars = [var for var in self.t_vars if 'DIS' not in var.name]
      self.d_vars = [var for var in self.t_vars if 'DIS' in var.name]
      num_param = 0.0
      for var in self.g_vars:
        num_param += int(np.prod(var.get_shape()));
      print("Number of parameters: %d"%num_param)
    self.saver = tf.train.Saver(max_to_keep=10)

  def forward(self, diff_in, xt, cell):
    # Initial state
    state = tf.zeros([self.batch_size, self.image_size[0]/8,
                      self.image_size[1]/8, 512])
    reuse = False
    # Encoder
    for t in xrange(self.K-1):
      enc_h, res_m = self.motion_enc(diff_in[:,:,:,t,:], reuse=reuse)
      h_dyn, state = cell(enc_h, state, scope='lstm', reuse=reuse)
      reuse = True

    pred = []
    # Decoder
    for t in xrange(self.T):
      if t == 0:
        h_cont, res_c = self.content_enc(xt, reuse=False)
        h_tp1 = self.comb_layers(h_dyn, h_cont, reuse=False)
        res_connect = self.residual(res_m, res_c, reuse=False)
        x_hat = self.dec_cnn(h_tp1, res_connect, reuse=False)
      else:
        enc_h, res_m = self.motion_enc(diff_in, reuse=True)
        h_dyn, state = cell(enc_h, state, scope='lstm', reuse=True)
        h_cont, res_c = self.content_enc(xt, reuse=reuse)
        h_tp1 = self.comb_layers(h_dyn, h_cont, reuse=True)
        res_connect = self.residual(res_m,res_c, reuse=True)
        x_hat = self.dec_cnn(h_tp1, res_connect, reuse=True)

      if self.c_dim == 3:
        # Network outputs are BGR so they need to be reversed to use
        # rgb_to_grayscale
        x_hat_rgb = tf.concat(axis=3,
                              values=[x_hat[:,:,:,2:3], x_hat[:,:,:,1:2],
                                      x_hat[:,:,:,0:1]])
        xt_rgb = tf.concat(axis=3,
                           values=[xt[:,:,:,2:3], xt[:,:,:,1:2],
                                   xt[:,:,:,0:1]])

        x_hat_gray = 1./255.*tf.image.rgb_to_grayscale(
            inverse_transform(x_hat_rgb)*255.
        )
        xt_gray = 1./255.*tf.image.rgb_to_grayscale(
            inverse_transform(xt_rgb)*255.
        )
      else:
        x_hat_gray = inverse_transform(x_hat)
        xt_gray = inverse_transform(xt)

      diff_in = x_hat_gray - xt_gray
      xt = x_hat
      pred.append(tf.reshape(x_hat,[self.batch_size, self.image_size[0],
                                    self.image_size[1], 1, self.c_dim]))

    return pred

  def motion_enc(self, diff_in, reuse):
    res_in = []
    conv1 = relu(conv2d(diff_in, output_dim=self.gf_dim, k_h=5, k_w=5,
                        d_h=1, d_w=1, name='dyn_conv1', reuse=reuse))
    res_in.append(conv1)
    pool1 = MaxPooling(conv1, [2,2])

    conv2 = relu(conv2d(pool1, output_dim=self.gf_dim*2, k_h=5, k_w=5,
                        d_h=1, d_w=1, name='dyn_conv2',reuse=reuse))
    res_in.append(conv2)
    pool2 = MaxPooling(conv2, [2,2])

    conv3 = relu(conv2d(pool2, output_dim=self.gf_dim*4, k_h=7, k_w=7,
                        d_h=1, d_w=1, name='dyn_conv3',reuse=reuse))
    res_in.append(conv3)
    pool3 = MaxPooling(conv3, [2,2])
    return pool3, res_in

  def content_enc(self, xt, reuse):
    res_in  = []
    conv1_1 = relu(conv2d(xt, output_dim=self.gf_dim, k_h=3, k_w=3,
                          d_h=1, d_w=1, name='cont_conv1_1',reuse=reuse))
    conv1_2 = relu(conv2d(conv1_1, output_dim=self.gf_dim, k_h=3, k_w=3,
                          d_h=1, d_w=1, name='cont_conv1_2',reuse=reuse))
    res_in.append(conv1_2)
    pool1 = MaxPooling(conv1_2, [2,2])

    conv2_1 = relu(conv2d(pool1, output_dim=self.gf_dim*2, k_h=3, k_w=3,
                          d_h=1, d_w=1, name='cont_conv2_1',reuse=reuse))
    conv2_2 = relu(conv2d(conv2_1, output_dim=self.gf_dim*2, k_h=3, k_w=3,
                          d_h=1, d_w=1, name='cont_conv2_2',reuse=reuse))
    res_in.append(conv2_2)
    pool2 = MaxPooling(conv2_2, [2,2])

    conv3_1 = relu(conv2d(pool2, output_dim=self.gf_dim*4, k_h=3, k_w=3,
                          d_h=1, d_w=1, name='cont_conv3_1', reuse=reuse))
    conv3_2 = relu(conv2d(conv3_1, output_dim=self.gf_dim*4, k_h=3, k_w=3,
                          d_h=1, d_w=1, name='cont_conv3_2', reuse=reuse))
    conv3_3 = relu(conv2d(conv3_2, output_dim=self.gf_dim*4, k_h=3, k_w=3,
                          d_h=1, d_w=1, name='cont_conv3_3',reuse=reuse))
    res_in.append(conv3_3)
    pool3   = MaxPooling(conv3_3, [2,2])
    return pool3, res_in

  def comb_layers(self, h_dyn, h_cont, reuse=False):
    comb1 = relu(conv2d(tf.concat(axis=3,values=[h_dyn, h_cont]),
                        output_dim=self.gf_dim*4, k_h=3, k_w=3,
                        d_h=1, d_w=1, name='comb1',reuse=reuse))
    comb2 = relu(conv2d(comb1, output_dim=self.gf_dim*2, k_h=3, k_w=3,
                        d_h=1, d_w=1, name='comb2', reuse=reuse))
    h_comb = relu(conv2d(comb2, output_dim=self.gf_dim*4, k_h=3, k_w=3,
                         d_h=1, d_w=1, name='h_comb', reuse=reuse))
    return h_comb

  def residual(self, input_dyn, input_cont, reuse=False):
    n_layers = len(input_dyn)
    res_out = []
    for l in xrange(n_layers):
      input_ = tf.concat(axis=3,values=[input_dyn[l],input_cont[l]])
      out_dim = input_cont[l].get_shape()[3]
      res1 = relu(conv2d(input_, output_dim=out_dim,
                         k_h=3, k_w=3, d_h=1, d_w=1,
                         name='res'+str(l)+'_1', reuse=reuse))
      res2 = conv2d(res1, output_dim=out_dim, k_h=3, k_w=3,
                    d_h=1, d_w=1, name='res'+str(l)+'_2', reuse=reuse)
      res_out.append(res2)
    return res_out

  def dec_cnn(self, h_comb, res_connect, reuse=False):
    shapel3 = [self.batch_size, self.image_size[0]/4,
               self.image_size[1]/4, self.gf_dim*4]
    shapeout3 = [self.batch_size, self.image_size[0]/4,
                 self.image_size[1]/4, self.gf_dim*2]
    depool3 = FixedUnPooling(h_comb, [2,2])
    deconv3_3 = relu(deconv2d(relu(tf.add(depool3, res_connect[2])),
                              output_shape=shapel3, k_h=3, k_w=3,
                              d_h=1, d_w=1, name='dec_deconv3_3', reuse=reuse))
    deconv3_2 = relu(deconv2d(deconv3_3, output_shape=shapel3, k_h=3, k_w=3,
                              d_h=1, d_w=1, name='dec_deconv3_2', reuse=reuse))
    deconv3_1 = relu(deconv2d(deconv3_2, output_shape=shapeout3, k_h=3, k_w=3,
                              d_h=1, d_w=1, name='dec_deconv3_1', reuse=reuse))

    shapel2 = [self.batch_size, self.image_size[0]/2,
               self.image_size[1]/2, self.gf_dim*2]
    shapeout3 = [self.batch_size, self.image_size[0]/2,
                 self.image_size[1]/2, self.gf_dim]
    depool2 = FixedUnPooling(deconv3_1, [2,2])
    deconv2_2 = relu(deconv2d(relu(tf.add(depool2, res_connect[1])),
                              output_shape=shapel2, k_h=3, k_w=3,
                              d_h=1, d_w=1, name='dec_deconv2_2', reuse=reuse))
    deconv2_1 = relu(deconv2d(deconv2_2, output_shape=shapeout3, k_h=3, k_w=3,
                              d_h=1, d_w=1, name='dec_deconv2_1', reuse=reuse))

    shapel1 = [self.batch_size, self.image_size[0],
               self.image_size[1], self.gf_dim]
    shapeout1 = [self.batch_size, self.image_size[0],
                 self.image_size[1], self.c_dim]
    depool1 = FixedUnPooling(deconv2_1, [2,2])
    deconv1_2 = relu(deconv2d(relu(tf.add(depool1, res_connect[0])),
                     output_shape=shapel1, k_h=3, k_w=3, d_h=1, d_w=1,
                     name='dec_deconv1_2', reuse=reuse))
    xtp1 = tanh(deconv2d(deconv1_2, output_shape=shapeout1, k_h=3, k_w=3,
                         d_h=1, d_w=1, name='dec_deconv1_1', reuse=reuse))
    return xtp1

  def discriminator(self, image):
    h0 = lrelu(conv2d(image, self.df_dim, name='dis_h0_conv'))
    h1 = lrelu(batch_norm(conv2d(h0, self.df_dim*2, name='dis_h1_conv'),
                          "bn1"))
    h2 = lrelu(batch_norm(conv2d(h1, self.df_dim*4, name='dis_h2_conv'),
                          "bn2"))
    h3 = lrelu(batch_norm(conv2d(h2, self.df_dim*8, name='dis_h3_conv'),
                          "bn3"))
    h = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'dis_h3_lin')

    return tf.nn.sigmoid(h), h

  def save(self, sess, checkpoint_dir, step):
    model_name = "MCNET.model"

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

  def load(self, sess, checkpoint_dir, model_name=None):
    print(" [*] Reading checkpoints...")

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      if model_name is None: model_name = ckpt_name
      self.saver.restore(sess, os.path.join(checkpoint_dir, model_name))
      print("     Loaded model: "+str(model_name))
      return True, model_name
    else:
      return False, None
