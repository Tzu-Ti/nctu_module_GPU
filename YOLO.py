from yoloPydarknet import pydarknetYOLO
import cv2
import imutils
import time

yolo = pydarknetYOLO(obdata="/home/titi/darknet/cfg/coco.data", weights="/home/titi/darknet/yolov3-tiny.weights", cfg="/home/titi/darknet/cfg/yolov3-tiny.cfg")

start_time = time.time()

if __name__ == "__main__":

	results = {
		'labels': [],
		'scores': [],
		'middleX': [],
		'middleY': []
	}

	img = cv2.imread("socketImage.jpg")
	
	yolo.getObject(img, labelWant="", drawBox=True, bold=1, textsize=0.6, bcolor=(0,0,255), tcolor=(255,255,255))
	print("Object counts:", yolo.objCounts)

	for i in range(yolo.objCounts):
		left, top, width, height, label, score = yolo.list_Label(i)
		middle_x = left + width/2
		middle_y = top + height/2
		results['labels'].append(label)
		results['scores'].append(score)
		results['middleX'].append(middle_x)
		results['middleY'].append(middle_y)	
	
	print("results:", results)
	cv2.imwrite("results.jpg", img)
	cv2.imshow("image", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
