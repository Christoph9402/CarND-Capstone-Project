from styx_msgs.msg import TrafficLight
import rospy
import numpy as np
import cv2
import yaml
import tensorflow as tf
from object_detection.utils import label_map_util
import visualization_utils as vis_util
import os
class TLClassifier(object):
    def __init__(self):
        """
        self.config=yaml.load(rospy.get_param("/traffic_light_config"))
        self.model_graph = tf.Graph()

        self.state=TrafficLight.UNKNOWN

        with tf.Session(graph=self.model_graph,config=self.config) as sess:
            self.session=sess
        image_tensor = self.model_graph.get_tensor_by_name('image_tensor:0')
        boxes_tensor = self.model_graph.get_tensor_by_name('boxes_tensor:0')
        score_tensor = self.model_graph.get_tensor_by_name('score_tensor:0')
        classes_tensor = self.model_graph.get_tensor_by_name('classes_tensor:0')
        """
        path = os.path.dirname(os.path.abspath(__file__))
        #define path for labelmap and frozen inference model
        label_pth = os.path.join(path, 'Model', 'labelmap.pbtxt')
        graph_pth = os.path.join(path, 'Model', 'frozen_inference_graph.pb')
        #load the frozen inference model
        self.graph_pth = graph_pth
        self.detection_graph = self.load_graph(self.graph_pth)
        #load labelmap
        self.label_pth = label_pth
        self.label_map = label_map_util.load_labelmap(label_pth)
        
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=3, use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)
        #pass

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        #TODO implement light color prediction
        with self.detection_graph.as_default():
            #start session compatible for tensorflow ersion 1.x
            with tf.compat.v1.Session(graph=self.detection_graph) as sess:
                
                np_image = np.expand_dims(image, axis=0)
                #use get tensor_by_name to extract the tensors, just as in the objec detection lab
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                detect_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                detect_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                detect_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
                #start detection
                (boxes, scores, classes, num) = sess.run([detect_boxes, detect_scores, detect_classes, num_detections],
                    feed_dict={image_tensor: np_image})
                #remove unnecessary dimensions
                boxes = np.squeeze(boxes)
                scores = np.squeeze(scores)
                classes = np.squeeze(classes).astype(np.int32)
                #define a threshold, which represents lower limit for detection 
                threshold = scores > 0.8
                #just look at boxes, scores and classes, if the detection is more than 80percent sure
                if np.any(threshold):
                    boxes = boxes[threshold]
                    scores = scores[threshold]
                    classes = classes[threshold]
                    #states represents all detected classes in the image
                    states, index, counts = np.unique(classes, return_inverse=True, return_counts=True)
                    state_scores = np.zeros((len(states),))

                    for i in range(len(states)):
                       #final state is the one, which has the highest aggregated score
                        state_scores[i] = np.sum(scores[index == i])
                    
                    final = states[np.argmax(state_scores)]
                    #return the final state
                    state = final
                    return state
                else:
                    
                    state =0
                    return state
    # use the load_graph function that was provided in the object detection lab (adjusted so it iscompatible with tensorflow 1.x)
    def load_graph(self, graph_file):
        """Loads a frozen inference graph"""
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return graph
