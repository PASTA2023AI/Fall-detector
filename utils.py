import cv2
import mediapipe as mp
import pandas as pd

#Returns a dictionary of all the keypoint names
def pose_landmark_to_dict(results, mp_pose, pose_data_dict):
    '''
    Helper function to extract all 33 landmarks and return as a dictionary for easy conversion to pandas dataframe 
    See - https://google.github.io/mediapipe/solutions/pose.html for more information on the landmarks
    '''

    # Names of the pose landmarks - total 33 of them
    pose_landmark_names = {
        "NOSE": mp_pose.PoseLandmark.NOSE,
        "LEFT_EYE_INNER": mp_pose.PoseLandmark.LEFT_EYE_INNER,
        "LEFT_EYE": mp_pose.PoseLandmark.LEFT_EYE,
        "LEFT_EYE_OUTER": mp_pose.PoseLandmark.LEFT_EYE_OUTER,
        "RIGHT_EYE_INNER": mp_pose.PoseLandmark.RIGHT_EYE_INNER,
        "RIGHT_EYE": mp_pose.PoseLandmark.RIGHT_EYE,
        "RIGHT_EYE_OUTER": mp_pose.PoseLandmark.RIGHT_EYE_OUTER,
        "LEFT_EAR": mp_pose.PoseLandmark.LEFT_EAR,
        "RIGHT_EAR": mp_pose.PoseLandmark.RIGHT_EAR,
        "MOUTH_LEFT": mp_pose.PoseLandmark.MOUTH_LEFT,
        "MOUTH_RIGHT": mp_pose.PoseLandmark.MOUTH_RIGHT,
        "LEFT_SHOULDER": mp_pose.PoseLandmark.LEFT_SHOULDER,
        "RIGHT_SHOULDER": mp_pose.PoseLandmark.RIGHT_SHOULDER,
        "LEFT_ELBOW": mp_pose.PoseLandmark.LEFT_ELBOW,
        "RIGHT_ELBOW": mp_pose.PoseLandmark.RIGHT_ELBOW,
        "LEFT_WRIST": mp_pose.PoseLandmark.LEFT_WRIST,
        "RIGHT_WRIST": mp_pose.PoseLandmark.RIGHT_WRIST,
        "LEFT_PINKY": mp_pose.PoseLandmark.LEFT_PINKY,
        "RIGHT_PINKY": mp_pose.PoseLandmark.RIGHT_PINKY,
        "LEFT_INDEX": mp_pose.PoseLandmark.LEFT_INDEX,
        "RIGHT_INDEX": mp_pose.PoseLandmark.RIGHT_INDEX,
        "LEFT_THUMB": mp_pose.PoseLandmark.LEFT_THUMB,
        "RIGHT_THUMB": mp_pose.PoseLandmark.RIGHT_THUMB,
        "LEFT_HIP": mp_pose.PoseLandmark.LEFT_HIP,
        "RIGHT_HIP": mp_pose.PoseLandmark.RIGHT_HIP,
        "LEFT_KNEE": mp_pose.PoseLandmark.LEFT_KNEE,
        "RIGHT_KNEE": mp_pose.PoseLandmark.RIGHT_KNEE,
        "LEFT_ANKLE": mp_pose.PoseLandmark.LEFT_ANKLE,
        "RIGHT_ANKLE": mp_pose.PoseLandmark.RIGHT_ANKLE,
        "LEFT_HEEL": mp_pose.PoseLandmark.LEFT_HEEL,
        "RIGHT_HEEL": mp_pose.PoseLandmark.RIGHT_HEEL,
        "LEFT_FOOT_INDEX": mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
        "RIGHT_FOOT_INDEX": mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
    }
    #Creates dictionary keys for each kaypoint's x,y,z,v values
    for key in pose_landmark_names:
        # Get the landmark data (x , y , z , visibility)
        landmark_data = results.pose_landmarks.landmark[pose_landmark_names[key]]

        try:
            # If key exists, just appends to previous list
            pose_data_dict[key + '_X'].append(landmark_data.x)
            pose_data_dict[key + '_Y'].append(landmark_data.y)
            pose_data_dict[key + '_Z'].append(landmark_data.z)
            pose_data_dict[key + '_V'].append(landmark_data.visibility)
        except:
            # If key does not exist, create it
            pose_data_dict[key + '_X'] = [landmark_data.x]
            pose_data_dict[key + '_Y'] = [landmark_data.y]
            pose_data_dict[key + '_Z'] = [landmark_data.z]
            pose_data_dict[key + '_V'] = [landmark_data.visibility]

    return pose_data_dict

#Get all the keypoints of a video frame by frame
def get_pose_data(fps, duration, FILE_PATH, annotate_image=True, print_landmarks=True):
    '''
    Extract the pose landmarks using mediapipe and cv2 from a .avi file. 
    See - https://google.github.io/mediapipe/solutions/pose.html

    fps : int
        - Number of Frames Per Second the video should be processed

    duration : int
        - Time (in seconds) the video should be processed

    FILE_PATH : str
        - Path to .avi video file

    annotate_image : bool
        - If true, the image with annoted landmark will be shown on screen

    print_landmarks : bool
        - If true, all landmarks will be printed out 
    '''

    sample_cycle = 1000 // fps
    frames = duration * fps

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    frame_count = 0

    pose_data_dict = {}

    # Open .avi file at filepath
    cap = cv2.VideoCapture(FILE_PATH)

    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:

        # print(pose)
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty file frame.")
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            if results.pose_landmarks is not None:
                if print_landmarks:
                    print(
                        f'Frame {frame_count}: {results.pose_landmarks}')

                if frame_count > 0:
                    # store all landmarks as a dict
                    pose_data_dict = pose_landmark_to_dict(
                        results, mp_pose, pose_data_dict)

                if annotate_image:
                    # Draw the pose annotation on the image.
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            else:
                print("results.pose_landmarks is None for", FILE_PATH)
     
            cv2.waitKey(sample_cycle)
            frame_count += 1
            if frame_count == frames:
                # Exit loop after reaching frame count
                break
            
            # Display the annotated image - Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))

    cap.release()
    print(pd.DataFrame(data=pose_data_dict))
    print(cv2.__version__)
    return pd.DataFrame(data=pose_data_dict)
    #return pose_data_dict
