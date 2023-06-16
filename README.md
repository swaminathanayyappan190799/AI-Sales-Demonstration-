# AI-Sales-Demonstration-

Grape Detection and tracking
	This project has many library dependenices especially cython_bbox and byte_track
		1)For cython download the .tar file from it's official pypi page and extract and modify the line 31 in the setup.py : https://stackoverflow.com/questions/60349980/is-there-a-way-to-install-cython-bbox-for-windows
			After doing the change move to the cython_bbox path and type this command "pip install -e ."
		2)For Bytetrack installation run the below commands on the notebook file (also run the requirements.txt for byte_track)
			import os
			import sys
			!git clone https://github.com/ifzhang/ByteTrack.git
			%cd {os.getcwd()}{os.sep}ByteTrack
			!pip install -q -r requirements.txt
			!python3 setup.py -q develop
			!pip install -q loguru lap
			sys.path.append(f"{os.getcwd()}{os.sep}ByteTrack")
			%cd .