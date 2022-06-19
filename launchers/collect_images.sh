#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# YOUR CODE BELOW THIS LINE
# ----------------------------------------------------------------------------


# NOTE: Use the variable DT_REPO_PATH to know the absolute path to your code
# NOTE: Use `dt-exec COMMAND` to run the main process (blocking process)

# launching app
export ROS_MASTER_URI=http://yoneduckie.local:11311  # TEMP
dt-exec rosrun pedestrian_detection collect_img.py
# dt-exec echo "This is an empty launch script. Update it to launch your application."


# ----------------------------------------------------------------------------
# YOUR CODE ABOVE THIS LINE

# wait for app to end
dt-launchfile-join
