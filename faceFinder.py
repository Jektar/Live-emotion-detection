import cv2

classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def findFaces(image, debug=False):
    pixels = cv2.imread(image)
    fboxes = classifier.detectMultiScale(pixels, 1.05, 3)
    if debug:
        for box in fboxes:
            x, y, w, h = box
            x2, y2 = x+w, y+h
            cv2.rectangle(pixels, (x, y), (x2, y2), (0, 255, 0), 2)

        cv2.imshow('face', pixels)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return fboxes

if __name__ == '__main__':
    findFaces('test.jpg', debug=True)