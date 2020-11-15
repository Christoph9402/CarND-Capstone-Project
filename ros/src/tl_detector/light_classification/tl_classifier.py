from styx_msgs.msg import TrafficLight
import rospy
import numpy as np
import cv2
import yaml
import tensorflow as tf

class TLClassifier(object):
    def __init__(self):

        self.config=yaml.load(rospy.get_param("/traffic_light_config"))
        self.model_graph = tf.Graph()

        self.state=TrafficLight.UNKNOWN

        with tf.Session(graph=self.model_graph,config=self.config) as sess:
            self.session=sess
        image_tensor = self.model_graph.get_tensor_by_name('image_tensor:0')
        boxes_tensor = self.model_graph.get_tensor_by_name('boxes_tensor:0')
        score_tensor = self.model_graph.get_tensor_by_name('score_tensor:0')
        classes_tensor = self.model_graph.get_tensor_by_name('classes_tensor:0')
        #pass

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        #TODO implement light color prediction


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