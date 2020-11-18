from styx_msgs.msg import TrafficLight
import rospy
import numpy as np
import cv2
import yaml
import tensorflow as tf
from object_detection.utils import label_map_util
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
        base_path = os.path.dirname(os.path.abspath(__file__))
        graph_pth = os.path.join(base_path, 'Model', 'frozen_inference_graph.pb')

        label_pth = os.path.join(base_path, 'Model', 'labelmap.pbtxt')

        self.graph_pth = graph_pth

        self.detection_graph = self.load_tf_graph(self.graph_pth)

        self.label_pth = label_pth

        self.label_map = label_map_util.load_labelmap(label_pth)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=3, use_display_name=True)

        self.category_index = label_map_util.create_category_index(self.categories)

        self.class_map = {
            1: TrafficLight.RED,
            2: TrafficLight.YELLOW,
            3: TrafficLight.GREEN
        }
        pass

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        #TODO implement light color prediction

        """
        img=cv2.resize(image,(512,512))
        #img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        with self.model_graph.as_default():
            (boxes,scores,classes) = self.session.run([boxes_tensor,score_tensor,classes_tensor],feed_dict={image_tensor:np.expand_dims(img,axis=0)})

        classes=np.squeeze(classes)
        scores=np.squeeze(scores)
        boxes=np.squeeze(boxes)

        for i in enumerate(boxes):
            if scores[i] > 0.5:
                light_state=self.classes[classes[i]]
                if light_state==0:
                    self.state=TrafficLight.RED
                elif light_state==1:
                    self.state=TrafficLight.YELLOW
                elif light_state==2:
                    self.state=TrafficLight.GREEN
                else:
                    self.state=TrafficLight.UNKNOWN
        return self.state
        """
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:

                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                detect_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                detect_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                detect_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

                image_expanded = np.expand_dims(image, axis=0)

                (boxes, scores, classes, num) = sess.run(
                    [detect_boxes, detect_scores, detect_classes, num_detections],
                    feed_dict={image_tensor: image_expanded})

                vis_util.visualize_boxes_and_labels_on_image_array(
                    image,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    self.category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=5,
                    line_thickness=5)

                boxes = np.squeeze(boxes)
                scores = np.squeeze(scores)
                classes = np.squeeze(classes).astype(np.int32)

                keep = scores > 0.5

                # if scores[0] > 0.5:

                #     state = self.class_map[classes[0]]

                # else:
                #     state = TrafficLight.UNKNOWN

                #### select the dominant traffic light class
                if np.any(keep):

                    boxes = boxes[keep]
                    scores = scores[keep]
                    classes = classes[keep]

                    members, index, counts = np.unique(classes, return_inverse=True, return_counts=True)
                    member_scores = np.zeros((len(members),))

                    for i in range(len(members)):
                        member_scores[i] = np.sum(scores[index == i])

                    select = np.argmax(member_scores)
                    winner = members[select]

                    state = self.class_map[winner]

                else:
                    state = TrafficLight.UNKNOWN
        #return TrafficLight.UNKNOWN

    def load_graph(graph_file):
        """Loads a frozen inference graph"""
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return graph
