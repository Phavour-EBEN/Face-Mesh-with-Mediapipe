import cv2 as cv
import mediapipe as mp

drawing_utils = mp.solutions.drawing_utils
face_mesh_utils = mp.solutions.face_mesh

drawing_spec = drawing_utils.DrawingSpec(thickness=1,circle_radius=1)

cap = cv.VideoCapture(0)
with face_mesh_utils.FaceMesh(min_detection_confidence=0.5,
               min_tracking_confidence=0.5) as face_mesh:
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # rgb_frame.flags.writeable = False
       
        
        rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        results = face_mesh.process(rgb_frame)
        rgb_frame.flags.writeable = True
        if results.multi_face_landmarks:
            for face_landmark in results.multi_face_landmarks:
                drawing_utils.draw_landmarks(image=rgb_frame,
                                             landmark_list=face_landmark,
                                             connections=face_mesh_utils.FACEMESH_TESSELATION,
                                             landmark_drawing_spec=drawing_spec,
                                            #  connections_drawing_spec=drawing_spec
                                             )
        cv.imshow('window', rgb_frame)
        key = cv.waitKey(1)

        if key== ord('q'):
            break

cap.release()
cv.destroyAllWindows()