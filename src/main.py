#!/home/chang/anaconda3/envs/project/bin python
from __future__ import print_function

import sys
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import struct
import numpy as np
from helper_tool import DataProcessing as DP
from sklearn.neighbors import KDTree
import tensorflow as tf
import os
from helper_tool import ConfigSST as cfg
from RandLANet import Network
import pickle

class SST :
  def __init__(self):
    self.name = 'SST'
    self.label_to_names = {0: 'unlabeled',
                           1: 'kickboard'}
    self.num_classes = len(self.label_to_names)
    self.label_values = np.sort([k for k, v in self.label_to_names.items()])
    self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
    self.ignored_labels = np.array([])
    self.points = np.load("/home/chang/data/11/velodyne/0_00000.npy")
    self.search_tree = KDTree(self.points)
    with open("/home/chang/data/11/proj/0_00000_proj.pkl", 'rb') as f:
      self.proj_inds = pickle.load(f)
    self.possibility = []
    self.min_possibility = []
    self.init_input_pipeline()
    self.proj_inds = np.squeeze(self.search_tree.query(self.points, return_distance=False))
    self.proj_inds = self.proj_inds.astype(np.int32)
    self.proj_inds = [self.proj_inds]
    c_proto = tf.ConfigProto()
    c_proto.gpu_options.allow_growth = True
    self.sess = tf.Session(config=c_proto)
    self.sess.run(tf.global_variables_initializer())
    self.remap_lut = np.zeros(101, dtype=np.int32)
    self.remap_lut[[0,1]] = [0,1]
    self.fields = [PointField('x', 0, PointField.FLOAT32, 1),
          PointField('y', 8, PointField.FLOAT32, 1),
          PointField('z', 16, PointField.FLOAT32, 1),
          # PointField('rgb', 12, PointField.UINT32, 1),
          PointField('rgba', 24, PointField.UINT32, 1),
          ]
    self.header = Header()
    self.header.frame_id = "map"
    self.pub = rospy.Publisher('/kickboard', PointCloud2,queue_size=100)

  def get_batch_gen(self):
    num_per_epoch = 4

    print("get_batch")
    def spatially_regular_gen():
      # Generator loop
      print("spatially")
      self.possibility = [np.random.rand(self.points.shape[0]) * 1e-3]
      self.min_possibility = [float(np.min(self.possibility[-1]))]
      for i in range(num_per_epoch):
        print("?")
        cloud_ind = int(np.argmin(self.min_possibility))
        pick_idx = np.argmin(self.possibility[cloud_ind])
        pc = np.array(self.search_tree.data, copy=False)
        
        tree = self.search_tree
        labels = np.zeros(np.shape(self.points)[0], dtype=np.uint8)
        selected_pc, selected_labels, selected_idx = self.crop_pc(pc, labels, tree, pick_idx)

        # update the possibility of the selected pc
        dists = np.sum(np.square((selected_pc - pc[pick_idx]).astype(np.float32)), axis=1)
        delta = np.square(1 - dists / np.max(dists))
        print(pc.shape)
        print(pick_idx)
        self.possibility[cloud_ind][selected_idx] += delta
        self.min_possibility[cloud_ind] = np.min(self.possibility[cloud_ind])

        if True:
          yield (selected_pc.astype(np.float32),
                 selected_labels.astype(np.int32),
                 selected_idx.astype(np.int32),
                 np.array([cloud_ind], dtype=np.int32))

    gen_func = spatially_regular_gen
    gen_types = (tf.float32, tf.int32, tf.int32, tf.int32)
    gen_shapes = ([None, 3], [None], [None], [None])

    return gen_func, gen_types, gen_shapes

  @staticmethod
  def crop_pc(points, labels, search_tree, pick_idx):
    # crop a fixed size point cloud for training
    center_point = points[pick_idx, :].reshape(1, -1)
    print(center_point)
    select_idx = search_tree.query(center_point, k=cfg.num_points)[1][0]
    print(select_idx)
    print(max(select_idx))
    select_idx = DP.shuffle_idx(select_idx)
    select_points = points[select_idx]
    select_labels = labels[select_idx]
    return select_points, select_labels, select_idx

  @staticmethod
  def get_tf_mapping2():

    def tf_map(batch_pc, batch_label, batch_pc_idx, batch_cloud_idx):
      features = batch_pc
      input_points = []
      input_neighbors = []
      input_pools = []
      input_up_samples = []

      for i in range(cfg.num_layers):
        neighbour_idx = tf.py_func(DP.knn_search, [batch_pc, batch_pc, cfg.k_n], tf.int32)
        sub_points = batch_pc[:, :tf.shape(batch_pc)[1] // cfg.sub_sampling_ratio[i], :]
        pool_i = neighbour_idx[:, :tf.shape(batch_pc)[1] // cfg.sub_sampling_ratio[i], :]
        up_i = tf.py_func(DP.knn_search, [sub_points, batch_pc, 1], tf.int32)
        input_points.append(batch_pc)
        input_neighbors.append(neighbour_idx)
        input_pools.append(pool_i)
        input_up_samples.append(up_i)
        batch_pc = sub_points

      input_list = input_points + input_neighbors + input_pools + input_up_samples
      input_list += [features, batch_label, batch_pc_idx, batch_cloud_idx]

      return input_list

    return tf_map

  def init_input_pipeline(self):
    gen_function_test, gen_types, gen_shapes = self.get_batch_gen()
    self.test_data = tf.data.Dataset.from_generator(gen_function_test, gen_types, gen_shapes)
    self.batch_test_data = self.test_data.batch(cfg.val_batch_size)
    map_func = self.get_tf_mapping2()
    self.batch_test_data = self.batch_test_data.map(map_func=map_func)
    self.batch_test_data = self.batch_test_data.prefetch(cfg.val_batch_size)
    iter = tf.data.Iterator.from_structure(self.batch_test_data.output_types, self.batch_test_data.output_shapes)
    self.flat_inputs = iter.get_next()
    self.test_init_op = iter.make_initializer(self.batch_test_data)

  def pointcloud_callback(self, msg):
      print('point cloud in!')
      points = []
      for data in pc2.read_points(msg, field_names=("x","y","z",), skip_nans=True):
        points = np.append(points, np.array([data[0],data[1],data[2]]))
      points = points.reshape((-1, 3))
      points = points.astype(np.float32)
      sub_points = DP.grid_sub_sampling(points, grid_size=0.06)
      search_tree = KDTree(sub_points)
      proj_inds = np.squeeze(search_tree.query(points, return_distance=False))
      proj_inds = proj_inds.astype(np.int32)
      self.points = sub_points
      self.search_tree = search_tree
      self.proj_inds = [proj_inds]
      self.sess.run(self.test_init_op)
      
      test_smooth = 0.98
      epoch_ind = 0
      while True:
        try:
          ops = (self.prob_logits,
                self.model.labels,
                self.model.inputs['input_inds'],
                self.model.inputs['cloud_inds'])
          print("ops")
          stacked_probs, labels, point_inds, cloud_inds = self.sess.run(ops, {self.model.is_training: False})
          self.test_probs = [np.zeros(shape=[len(l), self.model.config.num_classes], dtype=np.float16)
                                    for l in self.possibility]
          print('step ' + str(self.idx))
          self.idx += 1
          stacked_probs = np.reshape(stacked_probs, [self.model.config.val_batch_size,
                                                    self.model.config.num_points,
                                                    self.model.config.num_classes])
          for j in range(np.shape(stacked_probs)[0]):
            probs = stacked_probs[j, :, :]
            inds = point_inds[j, :]
            c_i = cloud_inds[j][0]
            self.test_probs[c_i][inds] = test_smooth * self.test_probs[c_i][inds] + (1 - test_smooth) * probs


        except tf.errors.OutOfRangeError:
          new_min = np.min(self.min_possibility)
          print('\nReproject Vote #{:d}'.format(int(np.floor(new_min))))

          for j in range(len(self.test_probs)):
            proj_inds = self.proj_inds
            probs = self.test_probs[j][proj_inds[0], :]
            pred = np.argmax(probs, 1)
            pred = pred + 1
            pred = pred.astype(np.uint32)
            upper_half = pred >> 16  # get upper half for instances
            lower_half = pred & 0xFFFF  # get lower half for semantics
            lower_half = self.remap_lut[lower_half]  # do the remapping of semantics
            pred = (upper_half << 16) + lower_half  # reconstruct full label
            pred = pred.astype(np.uint32)
          print("something happen")
          kickboards = []
          for i, point in enumerate(points):
            x = float(point[0])
            y = float(point[1])
            z = float(point[2])
            if int(pred[i]) == 1:
              r = 255
              g = 0
              b = 0
            if int(pred[i]) == 0:
              r = 0
              g = 255
              b = 0
            a = 255
            rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
            pt = [x, y, z, rgb]
            kickboards.append(pt)
          pc_kickboards = pc2.create_cloud(self.header, self.fields, kickboards)
          self.pub.publish(pc_kickboards)
          print("pub")
          break

  def start_tester(self, model):
    my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    self.saver = tf.train.Saver(my_vars, max_to_keep=100)
    self.restore_snap = "/home/chang/catkin_ws/src/randlanode/model/snap-39817"
    self.saver.restore(self.sess, self.restore_snap)
    print("Model restored from " + self.restore_snap)
    self.prob_logits = tf.nn.softmax(model.logits)
    self.test_probs = 0
    self.idx = 0
    self.model = model


def main(args=None):
  dataset = SST()
  model = Network(dataset, cfg)
  dataset.start_tester(model)
  cfg.saving = False
  print("ready")
  rospy.init_node('pointcloud_subscriber')
  sub = rospy.Subscriber('/ouster/points', PointCloud2, dataset.pointcloud_callback)
  rospy.spin()

if __name__ == '__main__':
  main()