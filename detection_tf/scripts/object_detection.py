#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 12:01:40 2017

@author: GustavZ
"""
import numpy as np
import os
import tensorflow as tf
import copy
import yaml
import cv2
import tarfile
import six.moves.urllib as urllib
from tensorflow.core.framework import graph_pb2

# Protobuf Compilation (once necessary)
#os.system('protoc object_detection/protos/*.proto --python_out=.')

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from stuff.helper import FPS2, WebcamVideoStream


## LOAD CONFIG PARAMS ##
with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)
video_input         = cfg['video_input']
visualize           = cfg['visualize']
vis_text            = cfg['vis_text']
max_frames          = cfg['max_frames']
width               = cfg['width']
height              = cfg['height']
fps_interval        = cfg['fps_interval']
allow_memory_growth = cfg['allow_memory_growth']
det_interval        = cfg['det_interval']
det_th              = cfg['det_th']
model_name          = cfg['model_name']
model_path          = cfg['model_path']
label_path          = cfg['label_path']
num_classes         = cfg['num_classes']
split_model         = cfg['split_model']
log_device          = cfg['log_device']


def _node_name(n):
  if n.startswith("^"):
    return n[1:]
  else:
    return n.split(":")[0]


# Download Model form TF's Model Zoo
def download_model():
    model_file = model_name + '.tar.gz'
    download_base = 'http://download.tensorflow.org/models/object_detection/'   
    if not os.path.isfile(model_path):
        print('Model not found. Downloading it now.')
        opener = urllib.request.URLopener()
        opener.retrieve(download_base + model_file, model_file)
        tar_file = tarfile.open(model_file)
        for file in tar_file.getmembers():
          file_name = os.path.basename(file.name)
          if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd() + '/models/')
        os.remove(os.getcwd() + '/' + model_file)
    else:
        print('Model found. Proceed.')
     
        
# Load a (frozen) Tensorflow model into memory.
def load_frozenmodel():
    print('Loading frozen model into memory')
    if not split_model:
        detection_graph = tf.Graph()
        with detection_graph.as_default():
          od_graph_def = tf.GraphDef()
          with tf.gfile.GFile(model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        return detection_graph, None, None
    
    else:
        # load a frozen Model and split it into GPU and CPU graphs
        input_graph = tf.Graph()
        with tf.Session(graph=input_graph):
            score = tf.placeholder(tf.float32, shape=(None, 1917, 90), name="Postprocessor/convert_scores")
            expand = tf.placeholder(tf.float32, shape=(None, 1917, 1, 4), name="Postprocessor/ExpandDims_1")
            for node in input_graph.as_graph_def().node:
                if node.name == "Postprocessor/convert_scores":
                    score_def = node
                if node.name == "Postprocessor/ExpandDims_1":
                    expand_def = node
                    
        detection_graph = tf.Graph()
        with detection_graph.as_default():
          od_graph_def = tf.GraphDef()
          with tf.gfile.GFile(model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            dest_nodes = ['Postprocessor/convert_scores','Postprocessor/ExpandDims_1']
        
            edges = {}
            name_to_node_map = {}
            node_seq = {}
            seq = 0
            for node in od_graph_def.node:
              n = _node_name(node.name)
              name_to_node_map[n] = node
              edges[n] = [_node_name(x) for x in node.input]
              node_seq[n] = seq
              seq += 1
        
            for d in dest_nodes:
              assert d in name_to_node_map, "%s is not in graph" % d
        
            nodes_to_keep = set()
            next_to_visit = dest_nodes[:]
            while next_to_visit:
              n = next_to_visit[0]
              del next_to_visit[0]
              if n in nodes_to_keep:
                continue
              nodes_to_keep.add(n)
              next_to_visit += edges[n]
        
            nodes_to_keep_list = sorted(list(nodes_to_keep), key=lambda n: node_seq[n])
        
            nodes_to_remove = set()
            for n in node_seq:
              if n in nodes_to_keep_list: continue
              nodes_to_remove.add(n)
            nodes_to_remove_list = sorted(list(nodes_to_remove), key=lambda n: node_seq[n])
        
            keep = graph_pb2.GraphDef()
            for n in nodes_to_keep_list:
              keep.node.extend([copy.deepcopy(name_to_node_map[n])])
        
            remove = graph_pb2.GraphDef()
            remove.node.extend([score_def])
            remove.node.extend([expand_def])
            for n in nodes_to_remove_list:
              remove.node.extend([copy.deepcopy(name_to_node_map[n])])
        
            with tf.device('/gpu:0'):
              tf.import_graph_def(keep, name='')
            with tf.device('/cpu:0'):
              tf.import_graph_def(remove, name='')
              
        return detection_graph, score, expand


def load_labelmap():
    print('Loading label map')
    label_map = label_map_util.load_labelmap(label_path)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return category_index


def detection(detection_graph, category_index, score, expand):
    print('Starting detection')
    # Session Config: allow seperate GPU/CPU adressing and limit memory allocation
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=log_device)
    config.gpu_options.allow_growth=allow_memory_growth
    cur_frames = 0
    with detection_graph.as_default():
      with tf.Session(graph=detection_graph,config=config) as sess:
        # Define Input and Ouput tensors
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        if split_model:
            score_out = detection_graph.get_tensor_by_name('Postprocessor/convert_scores:0')
            expand_out = detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1:0')
            score_in = detection_graph.get_tensor_by_name('Postprocessor/convert_scores_1:0')
            expand_in = detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1_1:0')
        # Start Video Stream
        video_stream = WebcamVideoStream(video_input,width,height).start()
        print ("Press 'q' to Exit")
        # fps calculation
        fps = FPS2(fps_interval).start()
        cur_frames = 0
        while video_stream.isActive():
          # read video frame and expand dimensions
          image_np = video_stream.read()
          image_np_expanded = np.expand_dims(image_np, axis=0)
          # actual Detection
          if not split_model:
              (boxes, scores, classes, num) = sess.run(
                      [detection_boxes, detection_scores, detection_classes, num_detections],
                      feed_dict={image_tensor: image_np_expanded})
          else:
              # Split Detection in two sessions.
              (score, expand) = sess.run([score_out, expand_out], feed_dict={image_tensor: image_np_expanded})
              (boxes, scores, classes, num) = sess.run(
                      [detection_boxes, detection_scores, detection_classes, num_detections],
                      feed_dict={score_in:score, expand_in: expand}) 
          # Visualization of the results of a detection.
          if visualize:
              vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=8)
              if vis_text:
                  cv2.putText(image_np,"fps: {}".format(fps.fps_local()), (10,30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
              cv2.imshow('object_detection', image_np)
              # Exit Option
              if cv2.waitKey(1) & 0xFF == ord('q'):
                  break
          else:
              # Exit after max frames if no visualization
              cur_frames += 1
              for box, score, _class in zip(np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)):
                    if cur_frames%det_interval==0 and score > det_th:
                        label = category_index[_class]['name']
                        print(label, score, box)
              if cur_frames >= max_frames:
                  break
          fps.update()
    # End everything
    fps.stop()
    video_stream.stop()     
    cv2.destroyAllWindows()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))


def main():  
    download_model()                 
    graph, score, expand = load_frozenmodel()
    category = load_labelmap()
    detection(graph, category, score, expand)
    
    
if __name__ == '__main__':
    main()
