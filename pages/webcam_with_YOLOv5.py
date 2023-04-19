import streamlit as st
import cv2
import numpy as np
from yolov5_video_transformer import YOLOv5VideoTransformer

# YOLOv5VideoTransformer 클래스의 인스턴스 생성
yolo = YOLOv5VideoTransformer()

# 모델 로드
yolo.load_model("../yolov5s.pt")

# 웹캠에서 영상을 캡쳐하는 함수
def capture_video():
    # 웹캠 캡쳐 객체 생성
    cap = cv2.VideoCapture(0)

    # 캡쳐 객체가 정상적으로 열렸는지 확인
    if not cap.isOpened():
        st.error("Unable to open camera.")
        return

    # 캡쳐한 영상 처리를 위한 루프
    while True:
        # 영상 프레임 읽기
        ret, frame = cap.read()

        # 프레임 읽기에 실패한 경우 루프 종료
        if not ret:
            break

        # 프레임 전처리
        image = yolo.preprocess(frame)

        # 객체 감지 수행
        output = yolo.predict(image)

        # 결과 시각화
        result = yolo.draw_bboxes(frame, output)

        # 결과 이미지 출력
        cv2.imshow("Object Detection", result)

        # 키 입력 대기
        key = cv2.waitKey(1)

        # 'q' 키를 입력하면 루프 종료
        if key == ord('q'):
            break

    # 사용한 자원 해제
    cap.release()
    cv2.destroyAllWindows()

# 스트림릿 앱 실행
def main():
    st.title("YOLOv5 Object Detection with Webcam")

    # 웹캠 캡쳐 시작 버튼
    if st.button("Start Webcam"):
        # 영상 캡쳐 함수 실행
        capture_video()

if __name__ == "__main__":
    main()
