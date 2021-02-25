import streamlit as st
import cv2
from PIL import Image
import numpy as np
import tempfile

def colorize_image(our_image):
    net = cv2.dnn.readNetFromCaffe('model/colorization_deploy_v2.prototxt', 'model/colorization_release_v2.caffemodel')
    pts = np.load('model/pts_in_hull.npy')
    # add the cluster centers as 1x1 convolutions to the model
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    # load the input image from disk, scale the pixel intensities to the
    # range [0, 1], and then convert the image from the BGR to Lab color
    # space
    image = np.array(our_image.convert('RGB'))
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)

    # resize the Lab image to 224x224 (the dimensions the colorization
    # network accepts), split channels, extract the 'L' channel, and then
    # perform mean centering
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    # pass the L channel through the network which will *predict* the 'a'
    # and 'b' channel values
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    # resize the predicted 'ab' volume to the same dimensions as our
    # input image
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

    # grab the 'L' channel from the *original* input image (not the
    # resized one) and concatenate the original 'L' channel with the
    # predicted 'ab' channels
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    # convert the output image from the Lab color space to RGB, then
    # clip any values that fall outside the range [0, 1]
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
    colorized = np.clip(colorized, 0, 1)

    # the current colorized image is represented as a floating point
    # data type in the range [0, 1] -- let's convert to an unsigned
    # 8-bit integer representation in the range [0, 255]
    colorized = (255 * colorized).astype("uint8")

    return colorized

def colorize_video(our_video) :
    net = cv2.dnn.readNetFromCaffe('model/colorization_deploy_v2.prototxt', 'model/colorization_release_v2.caffemodel')
    pts = np.load('model/pts_in_hull.npy')

    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    tf = tempfile.NamedTemporaryFile(delete=False)
    tf.write(our_video.read())
    vs = cv2.VideoCapture(tfile.name)   
    # vs = np.array(our_video.convert('RGB'))
    # loop over frames from the video stream
    while True:
        # grab the next frame and handle if we are reading from either
	    # VideoCapture or VideoStream
	    frame = vs.read()
	    frame = frame[1]

	    # if we are viewing a video and we did not grab a frame then we
	    # have reached the end of the video
	    if frame is None:
		    break

	    # resize the input frame, scale the pixel intensities to the
	    # range [0, 1], and then convert the frame from the BGR to Lab
	    # color space
	    frame = imutils.resize(frame, width=args["width"])
	    scaled = frame.astype("float32") / 255.0
	    lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)

	    # resize the Lab frame to 224x224 (the dimensions the colorization
	    # network accepts), split channels, extract the 'L' channel, and
	    # then perform mean centering
	    resized = cv2.resize(lab, (224, 224))
	    L = cv2.split(resized)[0]
	    L -= 50

	    # pass the L channel through the network which will *predict* the
	    # 'a' and 'b' channel values
	    net.setInput(cv2.dnn.blobFromImage(L))
	    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

	    # resize the predicted 'ab' volume to the same dimensions as our
	    # input frame, then grab the 'L' channel from the *original* input
	    # frame (not the resized one) and concatenate the original 'L'
	    # channel with the predicted 'ab' channels
	    ab = cv2.resize(ab, (frame.shape[1], frame.shape[0]))
	    L = cv2.split(lab)[0]
	    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

	    # convert the output frame from the Lab color space to RGB, clip
	    # any values that fall outside the range [0, 1], and then convert
	    # to an 8-bit unsigned integer ([0, 255] range)
	    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
	    colorized = np.clip(colorized, 0, 1)
        # colorized = (255 * colorized).astype("uint8")
    return colorized

def main():
    """BW2COLOR"""

    st.title("COLORIZE!!!")

    activities = ["Image","Video","About"]
    choice = st.sidebar.selectbox("Choose what you want to Colorize.",activities)

    if choice == "Image":
        html_temp = """
        <body style="background-color:red">
        <div style="background-color:teal ;padding:10px">
        <h2 style="color:white;text-align:center;">Colorization of B/W images</h2>
        <h3 style="color:white;text-align:center;"><div>BY </div> 
        <span>Anirudh Lodh, </span>
        <span> Utkarsh Saxena,</span>
        <span> ,Ajmal Khan</span>
        <span> ,Himey Patel</span>
        </div>
        </body>
        """
        st.markdown(html_temp, unsafe_allow_html=True)

        image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
        if image_file is not None:
            our_image = Image.open(image_file)
            st.text("Original Image")
            st.image(our_image)
            if st.button("COLORIZE!!"):
                result_img= colorize_image(our_image)
                st.image(result_img)
    
        else :
            st.text('Please select an image first!!')

    elif choice == "Video" :
        html_temp = """
        <body style="background-color:red;">
        <div style="background-color:teal ;padding:10px">
        <h2 style="color:white;text-align:center;">Colorization of B/W videos</h2>
        <h3 style="color:white;text-align:center;"><div>BY </div> 
        <span>Anirudh Lodh, </span>
        <span> Utkarsh Saxena,</span>
        <span> ,Ajmal Khan</span>
        <span> ,Himey Patel</span>
        </div>
        </body>
        """
        st.markdown(html_temp, unsafe_allow_html=True)

        video_file = st.file_uploader("Upload Video", type=['mp4','mkv'])
        if video_file is not None:
            # tfile = tempfile.NamedTemporaryFile(delete=False)
            # tfile.write(video_file.read())
            # our_video = cv2.VideoCapture(tfile.name)
            our_video = open("video/test.mp4","rb").read()
            st.text("Original Video")
            st.video(our_video)
            if st.button("COLORIZE!!"):
                result_video =  colorize_video(our_video)
                st.video(result_video)
    
        else :
            st.text('Please select a Video first!!')
if __name__ == '__main__':
    main()