import os
import cv2

for idx in [0, 1, 2, 4]:
    for fname in os.listdir("./{}".format(idx)):
        if fname == 'bkp':
            continue

        img = cv2.imread("./{}/{}".format(idx,fname))

        img = img[280:750, :, :]
        cv2.imwrite("./{}/{}".format(idx,fname), img)
        new_name = '{}-{}'.format(idx, fname[2:])
        os.rename("./{}/{}".format(idx,fname), "./{}/{}".format(idx,new_name))
        print("{}".format(new_name))


