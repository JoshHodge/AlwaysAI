import time
import edgeiq


def main():
    facial_detector = edgeiq.ObjectDetection("alwaysai/mobilenet_ssd_face")
    facial_detector.load(engine=edgeiq.Engine.DNN)

    print("Engine: {}".format(facial_detector.engine))
    print("Accelerator: {}\n".format(facial_detector.accelerator))
    print("Model:\n{}\n".format(facial_detector.model_id))

    fps = edgeiq.FPS()

    try:
        with edgeiq.WebcamVideoStream(cam=0) as webcam, \
                edgeiq.Streamer() as streamer:
            # Allow webcam to warm up
            time.sleep(2.0)
            fps.start()

            # loop detection
            while True:
                frame = webcam.read()
                # detect human faces
                results = facial_detector.detect_objects(
                        frame, confidence_level=.5)
                frame = edgeiq.markup_image(
                        frame, results.predictions, show_labels=False)

                # Generate text to display on streamer
                text = ["Model: {}".format(facial_detector.model_id)]
                text.append(
                        "Inference time: {:1.3f} s".format(results.duration))
                text.append("Faces:")

                for prediction in results.predictions:
                    text.append("{:2.2f}%".format(prediction.confidence * 100))

                streamer.send_data(frame, text)

                fps.update()

                if streamer.check_exit():
                    break

    finally:
        # stop fps counter and display information
        fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(fps.get_elapsed_seconds()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.compute_fps()))

        print("Program Ending")


if __name__ == "__main__":
    main()
