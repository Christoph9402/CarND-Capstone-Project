This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car. For more information about the project, see the project introduction [here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/5ab4b122-83e6-436d-850f-9f4d26627fd9).

Please use **one** of the two installation options, either native **or** docker installation.

### Native Installation

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases).

### Docker Installation
[Install Docker](https://docs.docker.com/engine/installation/)

Build the docker container
```bash
docker build . -t capstone
```

Run the docker file
```bash
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone
```

### Port Forwarding
To set up port forwarding, please refer to the "uWebSocketIO Starter Guide" found in the classroom (see Extended Kalman Filter Project lesson).

### Usage
To run the program in the Udacity workspace, navigate to the "workspace" dictionary and then run the following inside the terminal:

```
git clone https://github.com/Christoph9402/CarND-Capstone-Project.git && cd CarND-Capstone-Project && pip3 install -r requirements.txt && pip install tensorflow==1.15.0 && pip install matplotlib && cd ros && chmod -R +x src && cd src && cd tl_detector && cd light_classification && PYTHONPATH=$PYTHONPATH:/home/workspace/CarND-Capstone-Project/ros/src/tl_detector/light_classification && export PYTHONPATH && PYTHONPATH=$PYTHONPATH:/home/workspace/CarND-Capstone-Project/ros/src/tl_detector/light_classification/object_detection && export PYTHONPATH && PYTHONPATH=$PYTHONPATH:/home/workspace/CarND-Capstone-Project/ros/src/tl_detector/light_classification/slim && export PYTHONPATH && cd ../ && cd ../ && cd ../  && catkin_make clean && catkin_make && source devel/setup.sh && roslaunch launch/styx.launch --screen
```

Then run the simulator.
