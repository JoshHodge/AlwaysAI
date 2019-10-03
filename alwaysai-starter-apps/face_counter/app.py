"""
Sample application using object detection and centroid tracking to count
human faces.
"""

import time
import edgeiq


def main():
    obj_detect = edgeiq.ObjectDetection("alwaysai/mobilenet_ssd_face")
    obj_detect.load(engine=edgeiq.Engine.DNN)

    print("Engine: {}".format(obj_detect.engine))
    print("Accelerator: {}\n".format(obj_detect.accelerator))
    print("Model:\n{}\n".format(obj_detect.model_id))

    centroid_tracker = edgeiq.CentroidTracker(
            deregister_frames=20, max_distance=50)

    fps = edgeiq.FPS()

    try:
        with edgeiq.WebcamVideoStream(cam=0) as video_stream, \
                edgeiq.Streamer() as streamer:
            # Allow Webcam to warm up
            time.sleep(2.0)
            fps.start()

            # loop detection and centroid tracker
            while True:
                frame = video_stream.read()
                frame = edgeiq.resize(frame, width=400)
                results = obj_detect.detect_objects(frame, confidence_level=.5)

                text = ["Model: {}".format(obj_detect.model_id)]
                text.append(
                        "Inference time: {:1.3f} s".format(results.duration))
                text.append("Objects:")

                objects = centroid_tracker.update(results.predictions)

                # Update the label to reflect the object ID
                predictions = []
                for (object_id, prediction) in objects.items():
                    new_label = 'face {}'.format(object_id)
                    prediction.label = new_label
                    text.append(new_label)
                    predictions.append(prediction)

                frame = edgeiq.markup_image(frame, predictions)
                streamer.send_data(frame, text)
                fps.update()

                if streamer.check_exit():
                    break

    finally:
        fps.stop()
        print("elapsed time: {:.2f}".format(fps.get_elapsed_seconds()))
        print("approx. FPS: {:.2f}".format(fps.compute_fps()))

        print("Program Ending")


if __name__ == "__main__":
    main()
