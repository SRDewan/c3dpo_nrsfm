import cv2
import os

resFolder = 'data/exps/c3dpo/pretrained_cars'

for file in os.listdir(resFolder):
    image_folder = os.path.join(resFolder, file)

    if os.path.isdir(image_folder):
       video_name = image_folder + '/canonical3D.avi'

       images = [img for img in os.listdir(image_folder) if img.endswith(".png") and img.find("canonical") != -1 and img.find("3D") != -1]
       frame = cv2.imread(os.path.join(image_folder, images[0]))
       height, width, layers = frame.shape

       video = cv2.VideoWriter(video_name, 0, 1, (width,height))

       for image in images:
           video.write(cv2.imread(os.path.join(image_folder, image)))

       cv2.destroyAllWindows()
       video.release()
