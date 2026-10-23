import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"/data/FR/FRYOLO/train/runs/AI-TOD/AI-TOD-yolo11n-500-0.292-0.127/weights/best.pt")
    model.val(data=r'/data/FR/FRYOLO/ultralytics/cfg/datasets/AI-TOD.yaml',
              split='val',
              imgsz=640,
              batch=1,
              project='VISFPS',
              name='YOLO11n',
              )