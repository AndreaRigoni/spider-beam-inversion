include $(top_srcdir)/Common.mk


bin_PROGRAMS = main
main_SOURCES = main.cpp



DOWNLOADS = yolo3

yolo3: #@ downloaf yolo (mystic123)
yolo3_URL    = https://github.com/mystic123/tensorflow-yolo-v3.git
yolo3_BRANCH = master


yolo3_init: | yolo3
	@ cd yolo3; \
	  wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names; \
	  wget https://pjreddie.com/media/files/yolov3.weights; \
		python ./convert_weights.py

