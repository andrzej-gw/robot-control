import cv2
import mujoco

def generate_marker():
    
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    marker = cv2.aruco.generateImageMarker(dictionary, 23, 200);
    MARGIN = 10
    image = cv2.copyMakeBorder(
        marker,
        MARGIN, MARGIN,
        MARGIN, MARGIN,
        cv2.BORDER_CONSTANT,
        value=255
    )
    cv2.imwrite("marker23.png", image);


def main():
    model = mujoco.MjModel.from_xml_path("world1.xml")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=480, width=640)
    renderer.update_scene(data, camera="main_camera")
    image = renderer.render()

    cv2.imwrite("main_camera.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    corners, ids, _ = detector.detectMarkers(image)
    
    print(corners)
    print(ids)
    
    cv2.aruco.drawDetectedMarkers(image, corners, ids)

    cv2.imwrite("main_camera-decetions.png", image);

    
    generate_marker()
    

if __name__ == "__main__":
    main()
