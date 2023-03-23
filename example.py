from src.byte_tracker import BYTETracker
import numpy as np

class MockDetector(object):
    def __init__(self):
        pass

    def inference(self, image):
        """
        outputs shall be your final clean detetcor outputs, usually after nms
        this is the outputs of the demo: https://github.com/ifzhang/ByteTrack/blob/c8554543e1ba183058642f99fadd5680875344ee/tools/demo_track.py#L261
        and you can find how a good reference of it : https://github.com/ifzhang/ByteTrack/blob/57d86bd465a09494ca1e8de7d013857e7deaddf4/yolox/utils/boxes.py#L33
        below is just a mock up
        outputs has the length of detector outputs bboxes number, which is usually a fixed number
          - we use 3 just as a mock up
          - each element in outputs is a vector: (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        however that's from the official code demo with yolox
        if you take a look at src/byte_tracker.py update function, you will see it can accept both in shape of (N, 5) or more
        whereas N represents the number of outputs.
        Here we just use 5 with first 4 as boxes and the last as the socres
        """
        outputs = np.random.rand(1, 10, 5) * 416 # 1 just for batch size as 1
        # ideally, you don't want to contain this info in each inference but just as a meta in your prod env
        img_info = {
            'height': 416,
            'width': 416
        }
        return [
            outputs, img_info
        ]


def run():
    # Hyper Params to tune:
    # Best practice, you shall move them to a config
    aspect_ratio_thresh = 0.6
    min_box_area = 10
    track_thresh = 0.5
    track_buffer = 30
    match_thresh = 0.8
    fuse_score = False
    frame_rate = 30
    test_size = [416, 416]

    detector = MockDetector()
    tracker = BYTETracker(
        track_thresh=track_thresh,
        track_buffer=track_buffer,
        match_thresh=match_thresh,
        fuse_score=fuse_score,
        frame_rate=frame_rate)
    results = []
    mock_video = [
        np.random.rand(416, 416, 3),
        np.random.rand(416, 416, 3),
        np.random.rand(416, 416, 3),
        np.random.rand(416, 416, 3)
    ]

    for frame_id, image in enumerate(mock_video, 1):
        outputs, img_info = detector.inference(image)
        if outputs[0] is not None:
            online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], test_size)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    # save results
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )
    print("Mock up test results are:")
    print(results)
    return results


if __name__ == "__main__":
    run()
