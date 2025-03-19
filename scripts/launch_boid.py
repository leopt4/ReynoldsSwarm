#!/usr/bin/env python

import roslaunch
import rospy

def launch_boids(num_boids):
    # Initialize the ROS node
    rospy.init_node('boid_launcher', anonymous=True)

    # Create a ROSLaunch parent that will manage all the nodes
    launch = roslaunch.scriptapi.ROSLaunch()
    launch.start()

    # Loop to create and launch each boid node
    for i in range(num_boids):
        # Define the node for each boid
        node = roslaunch.core.Node(
            package='mrs_project',   # Replace with your actual package name
            node_type='boid_node.py',      # Node type (Python script)
            name=f'robot_{i}', 
            output='screen',
            args='{}'.format(i)
        )

        # Set a private parameter (id) for the node
        # rospy.set_param('id', i)

        # Launch the node
        process = launch.launch(node)

    # Keep the script alive (like roslaunch would)
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        launch.shutdown()  # Shutdown all nodes if ROS is interrupted

if __name__ == '__main__':
    # Fetch number of boids from ROS parameter server or use a default value
    num_boids = rospy.get_param('boid_count', 5)  # Default is 5 if not specified

    
    # Launch the boid nodes
    launch_boids(num_boids)
