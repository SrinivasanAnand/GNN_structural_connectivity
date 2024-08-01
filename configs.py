# Configs and path variables
# Author: Anand Srinivasan
# Reddick Lab

import os
# Root directory containing all DHCP and Bright raw data
root_dir = "/research_jude/rgs01_jude/dept/DI/DIcode/Anand/Data/"

bright_parent_dir = os.path.join(root_dir, "Bright")

dhcp_parent_dir = os.path.join(root_dir, "DHCP")
dhcp_raw_dir = os.path.join(dhcp_parent_dir, "raw")

dhcp_labels_filename = "cognitivescores_135subjects.csv"
bright_labels_filename = "bright_cognitive_scores.csv"

dhcp_labels_filepath = os.path.join(dhcp_raw_dir, dhcp_labels_filename)

