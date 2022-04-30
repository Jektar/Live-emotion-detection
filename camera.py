import cv2, threading, PIL, detectEmotions

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

emotions = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Sad', 'Surprised', 'Neutral']

color = (255, 0, 0)
font = cv2.FONT_HERSHEY_DUPLEX
font_size = 1
font_color = color
font_thickness = 2

def getPrediction(frame, faces):
    pilImg = PIL.Image.fromarray(frame)
    preds = []
    for (x, y, w, h) in faces:
        face = pilImg.crop((x, y, x+w, y+h))
        prediction = detectEmotions.predict(face)[0]
        prediction = [p for p in prediction]  # convert to list from ndarray
        best = max(prediction)
        preds.append([emotions[prediction.index(best)], best])

    return preds

def getBoxes():
    global frame, fboxes, predictions, done
    while not done:
        if not frame is None:
            fboxes = classifier.detectMultiScale(frame, 1.3, 5)
            predictions = getPrediction(frame, fboxes)

fboxes = []
frame = None
predictions = []
done = False

def gui():
    global frame, fboxes, predictions, done
    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        cv2.imshow("preview", frame)
        rval, frame = vc.read()

        for i, box in enumerate(fboxes):
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            fSize = font_size * (w / 250)
            if i < len(predictions):

                text = '{} {}%'.format(predictions[i][0], round(predictions[i][1] * 100, 2))
                if fSize > 1:
                    fontT = 2
                else:
                    fontT = 1
                cv2.putText(frame, text, (x, int(y - 9 * fSize)), font, fSize, font_color, fontT)

        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            done = True
            break

    cv2.destroyWindow("preview")
    vc.release()
    #import sys
    #sys.exit()

if __name__ == '__main__':
    t = threading.Thread(target=getBoxes)
    t.start()
    gui()
    import sys
    sys.exit()