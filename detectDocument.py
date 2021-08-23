import tensorflow as tf
import cv2
import numpy as np
from dummy_perspective import perspective_transform
import matplotlib.pyplot as plt
from Crop_kimlik import crop_kimlik
from blob_detection import blob_detector


def argsProcessor():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--imagePath", help="Path to the document image")
    parser.add_argument("-o", "--outputPath", help="Path to store the result")
    parser.add_argument("-o1", "--outputPath1", help="Path to store the result1")
    parser.add_argument("-rf", "--retainFactor", help="Floating point in range (0,1) specifying retain factor",
                        default="0.85")
    parser.add_argument("-cm", "--cornerModel", help="Model for corner point refinement",
                        default="./TrainedModel/cornerRefiner.pb")
    parser.add_argument("-dm", "--documentModel", help="Model for document corners detection",
                        default="./TrainedModel/getCorners.pb")
    return parser.parse_args()


def load_graph(frozen_graph_filename, inputName, outputName):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.io.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    # Then, we can use again a convenient built-in function to import a graph_def into the
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    x = graph.get_tensor_by_name('prefix/'+inputName+':0')
    y = graph.get_tensor_by_name('prefix/'+outputName+':0')
    return graph, x, y


def refineCorner(img, sess, x,y_eval, retainFactor):
    ans_x = 0.0
    ans_y = 0.0
    o_img = np.copy(img)
    y = None
    x_start = 0
    y_start = 0
    up_scale_factor = (img.shape[1], img.shape[0])
    myImage = np.copy(o_img)
    CROP_FRAC = retainFactor
    while(myImage.shape[0]>10 and myImage.shape[1]>10):
        img_temp = cv2.resize(myImage, (32, 32))
        img_temp = np.expand_dims(img_temp, axis=0)
        response = y_eval.eval(feed_dict={
            x: img_temp}, session=sess)
        response_up = response[0]
        response_up = response_up * up_scale_factor
        y = response_up + (x_start, y_start)
        x_loc = int(y[0])
        y_loc = int(y[1])

        if x_loc > myImage.shape[1] / 2:
            start_x = min(x_loc + int(round(myImage.shape[1] * CROP_FRAC / 2)), myImage.shape[1]) - int(round(
                myImage.shape[1] * CROP_FRAC))
        else:
            start_x = max(x_loc - int(myImage.shape[1] * CROP_FRAC / 2), 0)
        if y_loc > myImage.shape[0] / 2:
            start_y = min(y_loc + int(myImage.shape[0] * CROP_FRAC / 2), myImage.shape[0]) - int(
                myImage.shape[0] * CROP_FRAC)
        else:
            start_y = max(y_loc - int(myImage.shape[0] * CROP_FRAC / 2), 0)
        ans_x += start_x
        ans_y += start_y
        myImage = myImage[start_y:start_y + int(myImage.shape[0] * CROP_FRAC),
                  start_x:start_x + int(myImage.shape[1] * CROP_FRAC)]
        img = img[start_y:start_y + int(img.shape[0] * CROP_FRAC), start_x:start_x + int(img.shape[1] * CROP_FRAC)]
        up_scale_factor = (img.shape[1], img.shape[0])

    ans_x += y[0]
    ans_y += y[1]
    return (int(round(ans_x)), int(round(ans_y)))

def getCorners(img, sess, x, output):
        o_img = np.copy(img)
        myImage = np.copy(o_img)
        img_temp = cv2.resize(myImage, (32, 32))
        img_temp = np.expand_dims(img_temp, axis=0)
        response = output.eval(feed_dict={
            x: img_temp}, session=sess)
        response = response[0]
        x = response[[0, 2, 4, 6]]
        y = response[[1, 3, 5, 7]]
        x = x*myImage.shape[1]
        y = y*myImage.shape[0]

        tl = myImage[max(0, int(2*y[0] - (y[3]+y[0])/2)):int((y[3]+y[0])/2),
            max(0, int(2*x[0] - (x[1]+x[0])/2)):int((x[1]+x[0])/2)]

        tr = myImage[max(0, int(2*y[1] - (y[1]+y[2])/2)):int((y[1]+y[2])/2),
            int((x[1]+x[0])/2):min(myImage.shape[1]-1, int(x[1]+(x[1]-x[0])/2))]

        br = myImage[int((y[1]+y[2])/2):min(myImage.shape[0]-1, int(y[2]+(y[2]-y[1])/2)),
              int((x[2]+x[3])/2):min(myImage.shape[1]-1, int(x[2]+(x[2]-x[3])/2))]

        bl = myImage[int((y[0]+y[3])/2):min(myImage.shape[0]-1, int(y[3]+(y[3]-y[0])/2)),
             max(0, int(2*x[3] - (x[2]+x[3])/2)):int((x[3]+x[2])/2)]

        tl = (tl, max(0, int(2*x[0] -(x[1]+x[0])/2)), max(0, int(2*y[0] - (y[3]+y[0])/2)))
        tr = (tr, int((x[1]+x[0])/2), max(0, int(2*y[1] - (y[1]+y[2])/2)))
        br = (br, int((x[2]+x[3])/2), int((y[1]+y[2])/2))
        bl = (bl, max(0, int(2*x[3] - (x[2]+x[3])/2)), int((y[0]+y[3])/2))
        return tl, tr, br, bl


if __name__ == "__main__":
    args = argsProcessor()
    graph, x, y = load_graph(args.cornerModel, "Corner/inputTensor", "Corner/outputTensor")
    graphCorners, xCorners, yCorners = load_graph(args.documentModel, "Input/inputTensor", "FCLayers/outputTensor")
    img_before_crop = cv2.imread(args.imagePath)
    sess = tf.compat.v1.Session(graph=graph)
    sessCorners = tf.compat.v1.Session(graph=graphCorners)
    result = np.copy(img_before_crop)
    cols = img_before_crop.shape[0]
    rows = img_before_crop.shape[1]

    img, cropped_front, crop_point = crop_kimlik(img_before_crop)

    paste_rows = rows-int(crop_point)
    paste_cols = cols

    data = getCorners(img, sessCorners, xCorners, yCorners)
    corner_address = []
    counter = 0
    for b in data:
        a = b[0]
        temp = np.array(refineCorner(a, sess, x, y, float(args.retainFactor)))
        temp[0] += b[1]
        temp[1] += b[2]
        corner_address.append(temp)
        print(temp)
        counter += 1
    for a in range(0,len(data)):
        cv2.line(img, tuple(corner_address[a % 4]), tuple(corner_address[(a+1) % 4]), (255, 0, 0), 2)
    # @Handenur Caliskan added----------------------------------------------------------------------
    new_corner_address = [corner_address[0], corner_address[1], corner_address[3], corner_address[2]]
    transformed_image = perspective_transform(img, paste_rows, paste_cols, new_corner_address)
    # ----------------------------------------------------------------------------------------------
    
    paste_back = 255* np.ones((cols,rows-int(crop_point),3),np.uint8)
    concat_ids = cv2.hconcat([cropped_front,paste_back])

    if (cropped_front.shape[1]+int(3*paste_back.shape[1]/4))-(cropped_front.shape[1]+int(paste_back.shape[1]/4))!= transformed_image.shape[1]:
        x = transformed_image.shape[1]+(cropped_front.shape[1]+int(paste_back.shape[1]/4))
        concat_ids[int(paste_back.shape[0]/6):int(5*paste_back.shape[0]/6), cropped_front.shape[1]+int(paste_back.shape[1]/4):x] = transformed_image
    else:
        concat_ids[int(paste_back.shape[0]/6):int(5*paste_back.shape[0]/6), cropped_front.shape[1]+int(paste_back.shape[1]/4):cropped_front.shape[1]+int(3*paste_back.shape[1]/4)] = transformed_image
    
    # fig1 = plt.figure('img')
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.show()
    
    fig3 = plt.figure('Final')
    plt.imshow(cv2.cvtColor(concat_ids, cv2.COLOR_BGR2RGB))
    
    fig2 = plt.figure('Covered back')
    plt.imshow(cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB))

    plt.show()
    cv2.imwrite(args.outputPath, transformed_image)
    cv2.imwrite(args.outputPath1, concat_ids)
    