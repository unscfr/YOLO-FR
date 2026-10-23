from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'/data/FR/FRYOLO/train/runs/AI-TOD/AI-TOD-yolo11n-500-0.292-0.127/weights/best.pt')  # select your model.pt path
    model.predict(source=r'/data/FR/datasets/AI-TOD/val/images/1e3b41061.png',
                  imgsz=800,
                  project='featuremap',
                  name='aitod-yolo11sum',
                  save=True,
                  # conf=0.2,
                  # iou=0.7,
                  # agnostic_nms=True,
                  visualize=True,  # visualize model features maps
                  # line_width=2, # line width of the bounding boxes
                  # show_conf=False, # do not show prediction confidence
                  # show_labels=False, # do not show prediction labels
                  # save_txt=True, # save results as .txt file
                  # save_crop=True, # save cropped images with results
                  )