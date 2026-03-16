import numpy as np
import rclpy
from rclpy.node import Node
from typing import Any, Callable, Optional, Tuple
import time

from ament_index_python.packages import get_package_share_directory
from od_msg.srv import SrvDepthPosition
from pick_test.realsense import ImgNode
from pick_test.yolo_obb import YoloModel
import cv2


PACKAGE_NAME = 'pick_test'
PACKAGE_PATH = get_package_share_directory(PACKAGE_NAME)


class ObjectDetectionNode(Node):
    def __init__(self, model_name = 'yolo'):
        super().__init__('object_detection_node')
        self.img_node = ImgNode()
        self.model = self._load_model(model_name)
        self.intrinsics = self._wait_for_valid_data(
            self.img_node.get_camera_intrinsic, "camera intrinsics"
        )
        
        self.latest_box = None  # [cx, cy, w, h, r] 저장용
        self.latest_name = ""   # 클래스 이름 저장용
        
        # 실시간 화면 갱신을 위한 타이머 추가 (30 FPS 목표)
        self.display_timer = self.create_timer(0.03, self._display_monitor)
        
        self.create_service(
            SrvDepthPosition,
            'get_3d_position',
            self.handle_get_depth
        )
        self.get_logger().info("ObjectDetectionNode initialized.")


    def _display_monitor(self):
        rclpy.spin_once(self.img_node, timeout_sec=0)
        frame = self.img_node.get_color_frame()
        if frame is not None:
            display_frame = frame.copy()

            # --------------------- 수정 -------------------------------------------------------
            # [수정] 현재 탐지 대상이 box가 아닐 때만 ROI 가이드라인 표시
            if self.latest_name != "box":
                search_roi = [580, 120, 400, 380]
                rx, ry, rw, rh = search_roi
                cv2.rectangle(display_frame, (rx, ry), (rx + rw, ry + rh), (255, 0, 0), 2)
                cv2.putText(display_frame, "ROI Mode", (rx, ry-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            else:
                # 박스 탐지 중일 때는 전체 화면임을 알리는 텍스트 추가
                cv2.putText(display_frame, "Full Frame Mode (Box)", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            # ----------------------------------------------------------------------------------

            # 2. 최신 탐지 결과가 있다면 OBB 박스 그리기 (초록색)
            # (OpenCV로 찾은 박스도 여기서 그려집니다)
            if self.latest_box is not None:
                cx, cy, w, h, r = self.latest_box
                
                # 회전된 4개 꼭짓점 계산
                cos_r = np.cos(r)
                sin_r = np.sin(r)
                dx, dy = w / 2, h / 2
                
                # 로컬 좌표 pts -> 회전 및 중심 이동
                pts = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]])
                rotated_pts = []
                for px, py in pts:
                    nx = px * cos_r - py * sin_r + cx
                    ny = px * sin_r + py * cos_r + cy
                    rotated_pts.append([int(nx), int(ny)])
                
                pts_array = np.array(rotated_pts, np.int32)
                cv2.polylines(display_frame, [pts_array], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.putText(display_frame, self.latest_name, (int(cx), int(cy)-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Detection Monitor", display_frame)
            cv2.waitKey(1)


    def _load_model(self, name):
        """모델 이름에 따라 인스턴스를 반환합니다."""
        if name.lower() == 'yolo':
            return YoloModel()
        raise ValueError(f"Unsupported model: {name}")

    def handle_get_depth(self, request, response):
        target_name = request.target 
        
        # [수정] 박스 내부 전체 스캔 모드 추가
        if target_name == "box_contents":
            result = self._compute_all_objects_in_box("auto")
        # ==========================================================
        # 🌟 [핵심 변경] 박스 찾기 요청이 오면 YOLO 대신 OpenCV로 바로 점프!
        # ==========================================================
        elif target_name == "box":
            result = self._find_box_with_opencv()
        else:
            # 기존 컨베이어 단일 물체 탐지 (상품)
            result = self._compute_position_with_verification(target_name)
        
        response.depth_position = [float(x) for x in result]
        return response

    # ==========================================================
    # 🌟 [새로 추가된 함수] OpenCV 윤곽선을 이용한 초고속/초정밀 박스 탐지!
    # ==========================================================
    def _find_box_with_opencv(self):
        """OpenCV의 윤곽선과 minAreaRect를 이용해 흔들림 없는 박스 좌표를 찾습니다."""
        self.get_logger().info("📦 OpenCV 알고리즘으로 박스 좌표를 정밀 탐색합니다...")
        
        while rclpy.ok():
            rclpy.spin_once(self.img_node, timeout_sec=0.01)
            color_image = self.img_node.get_color_frame()
            
            if color_image is None:
                continue
            
            # ==========================================================
            # 🌟 [수정된 1~3단계] Canny 대신 이진화(Thresholding) 적용!
            # ==========================================================
            # 1단계: 흑백 변환 및 블러(노이즈 제거)
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)

            # 2단계: 이진화 (Threshold) - 박스만 하얗게 도려내기
            # 130이라는 숫자는 밝기 기준입니다. (0: 검은색 ~ 255: 완전 흰색)
            # 130보다 밝은 박스는 흰색(255)으로, 어두운 매트나 파란선은 검은색(0)으로 밀어버립니다.
            _, thresh = cv2.threshold(blurred, 130, 255, cv2.THRESH_BINARY)

            # 박스 안쪽의 희미한 그림자나 끊어진 부분을 메꿔서 딴딴한 사각형으로 만듭니다.
            kernel = np.ones((11, 11), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            # 🌟 [치트키] 로봇 팔 강제 삭제 (마스킹)
            # 사진 하단에 자꾸 로봇 팔이 찍혀서 방해하므로, 화면 아래쪽 100픽셀을 까맣게 칠해버립니다.
            img_h, img_w = thresh.shape
            thresh[img_h - 100 : img_h, :] = 0  # 숫자를 키우면 더 위까지 까맣게 지워집니다.

            # 화면에 어떻게 분리됐는지 실시간으로 확인하고 싶으시다면 아래 주석을 풀어보세요!
            cv2.imshow("Threshold Box", thresh)

            # 3단계: 윤곽선(Contours) 찾기
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # ==========================================================
            # (이 아래 4단계부터는 기존 코드와 완전히 동일하게 유지하시면 됩니다!)
            if not contours:
                self.get_logger().warn("화면에서 테두리를 찾을 수 없습니다.")
                time.sleep(0.5)
                continue

            if not contours:
                self.get_logger().warn("화면에서 테두리를 찾을 수 없습니다.")
                time.sleep(0.5)
                continue

            # 4단계: 가장 면적이 큰 테두리가 박스!
            largest_contour = max(contours, key=cv2.contourArea)

            # 너무 작은 노이즈(먼지 등)는 박스로 취급하지 않음 (10000 픽셀 이상)
            if cv2.contourArea(largest_contour) < 10000:
                self.get_logger().warn("가장 큰 물체가 박스 면적보다 너무 작습니다. 재탐색합니다.")
                time.sleep(0.5)
                continue

            # 5단계: 삐딱한 박스를 감싸는 최소 면적의 직사각형(minAreaRect) 씌우기!
            rect = cv2.minAreaRect(largest_contour)
            box_center, box_size, box_angle = rect
            
            cx, cy = int(box_center[0]), int(box_center[1])
            w_pixel, h_pixel = box_size[0], box_size[1]
            
            # OpenCV minAreaRect의 각도 기준을 수학적 라디안(radian)으로 통일
            # OpenCV는 가로(w)보다 세로(h)가 길면 각도를 -90도 틀어버리는 특징이 있습니다.
            angle_deg = box_angle
            if w_pixel < h_pixel:
                angle_deg += 90.0
            angle_rad = np.radians(angle_deg)

            # 모니터링 창(디스플레이) 업데이트용 변수 저장
            self.latest_name = "box"
            self.latest_box = [cx, cy, max(w_pixel, h_pixel), min(w_pixel, h_pixel), angle_rad]

            # 6단계: 정중앙의 깊이(Depth) 값 읽기
            cz = self._get_depth(cx, cy, w_pixel, h_pixel)
            
            if cz is None or cz == 0:
                self.get_logger().error("🛑 박스 중앙의 깊이를 읽을 수 없습니다! 반사/가림 발생.")
                time.sleep(0.5)
                continue
                
            # 7단계: 픽셀을 3D 절대 좌표로 변환
            x, y, z = self._pixel_to_camera_coords(cx, cy, cz)

            self.get_logger().info(f"✅ OpenCV 박스 찾기 성공! (Ang: {angle_deg:.1f}deg)")
            
            # ==========================================================
            # 🌟 [초핵심] 카메라는 가로/세로 길이를 재지 않습니다!
            # 사용자가 정해준 고정 규격(266.0, 185.0)을 무조건 박아넣습니다!
            # [X, Y, Z, 각도(디그리), 가로 고정, 세로 고정, ID(없음)]
            # ==========================================================
            fixed_w = 266.0
            fixed_h = 185.0
            
            # 로봇이 7개의 값을 기대하므로 끝에 -1.0 (클래스 ID 없음)을 붙여줍니다.
            return [x, y, z, angle_deg, fixed_w, fixed_h, -1.0]

    
    def _compute_all_objects_in_box(self, target):
        """박스 내부의 모든 물체를 탐지하여 리스트로 반환"""
        self.get_logger().info("🔍 박스 내부 모든 물체 탐지 중...")
        
        detections = self.model.get_all_detections(self.img_node, target)
        
        if not detections:
            return [0.0] # 감지된 물체 0개
        
        class_names = {v: k for k, v in self.model.reversed_class_dict.items()}
        valid_results = []
        detected_item_names = []  
        fx, fy = self.intrinsics['fx'], self.intrinsics['fy']
        
        for det in detections:
            label_name = class_names.get(det['class'], "unknown")
            
            if label_name == "box":
                continue 
            
            cx, cy, w_pixel, h_pixel, angle_rad = det['box']
            cz = self._get_depth(int(cx), int(cy), int(w_pixel), int(h_pixel))
            
            if cz is None or cz == 0:
                self.get_logger().warn(f"⚠️ {label_name}의 깊이를 읽지 못했습니다! 임시값으로 장애물 처리합니다.")
                cz = 500.0  
            
            obj_w_mm = (w_pixel * cz) / fx
            obj_h_mm = (h_pixel * cz) / fy
            
            x, y, z = self._pixel_to_camera_coords(cx, cy, cz)
            
            class_id = self.model.reversed_class_dict.get(label_name, -1)
            valid_results.extend([x, y, z, np.degrees(angle_rad), obj_w_mm, obj_h_mm, float(class_id)])
            
            detected_item_names.append(label_name)

        final_count = float(len(valid_results) // 7)
        
        if final_count > 0:
            self.get_logger().info(f"✅ 탐지 완료 ({int(final_count)}개): {', '.join(detected_item_names)}")
        else:
            self.get_logger().warn("⚠️ 탐지된 물체는 있으나 유효한 깊이 정보를 찾을 수 없습니다.")

        return [final_count] + valid_results
    
    def _compute_position_with_verification(self, target):
        """1차 탐지 후 2초 대기, 2차 확정 탐지 후 좌표 반환 (성공할 때까지 무한 반복)"""
        
        if target == "box":
            current_roi = None  # 전체 화면
        else:
            current_roi = [580, 120, 400, 380]
            
        while rclpy.ok():
            box1, score1, name1 = self.model.get_best_detection(self.img_node, target if target else "auto", roi=current_roi)
            
            if box1 is None:
                self.latest_box = None  
                self.latest_name = ""
                time.sleep(0.5) 
                continue 
            
            self.latest_box = box1
            self.latest_name = name1
            self.get_logger().warn(f"1차 탐지 성공: {name1}. 2초간 대기하며 검증합니다...")
            
            start_wait = time.time()
            while time.time() - start_wait < 2.0:
                rclpy.spin_once(self.img_node, timeout_sec=0.01)
                time.sleep(0.01)

            self.get_logger().info("2차 확정 탐지를 시작합니다.")
            box2, score2, name2 = self.model.get_best_detection(self.img_node, target if target else "auto", roi=current_roi)

            if box2 is not None:
                self.latest_box = box2
                self.latest_name = name2
                
                cx, cy, w_pixel, h_pixel, angle_rad = box2
                cz = self._get_depth(int(cx), int(cy), int(w_pixel), int(h_pixel))

                if cz is None or cz == 0:
                    self.get_logger().error("🛑 깊이 정보를 읽을 수 없습니다! 다시 처음부터 탐지를 재시도합니다.")
                    time.sleep(0.5) 
                    continue        
                
                # ==========================================================
                # 🌟 [해결!] 타겟(Target)에 따라 그림자 커트라인을 2개로 분리!
                # ==========================================================
                if target == "box":
                    # 1. 박스 찾을 때 (로봇 자세 A): 바닥이 466이므로 450으로 방어
                    FLOOR_DEPTH_LIMIT = 445.0  
                
                
                    if cz > FLOOR_DEPTH_LIMIT:
                        self.get_logger().warn(f"👻 그림자 감지됨! ({target} 모드, 측정 깊이: {cz:.1f}mm) -> 무시합니다.")
                        self.latest_box = None 
                        self.latest_name = ""
                        time.sleep(0.5)
                        continue
                # ==========================================================

                fx = self.intrinsics['fx']
                fy = self.intrinsics['fy']

                obj_w_mm = ((w_pixel * cz) / fx)
                obj_h_mm = ((h_pixel * cz) / fy)

                x, y, z = self._pixel_to_camera_coords(cx, cy, cz)
                angle_deg = np.degrees(angle_rad)

                self.get_logger().info(f"✅ 최종 확정: {name2} / Ang: {angle_deg:.1f}deg")
                self.get_logger().info(f"✅ 측정 완료: {name2} -> {obj_w_mm:.1f}x{obj_h_mm:.1f}mm")

                class_id = self.model.reversed_class_dict.get(name2, -1)
                
                return x, y, z, angle_deg, obj_w_mm, obj_h_mm, class_id
            
            else:
                self.get_logger().error("❌ 검증 실패: 2초 후 물체가 사라졌습니다. 다시 시도합니다.")
                self.latest_box = None 
                time.sleep(0.5)
                continue
        
    def _get_depth(self, cx, cy, w_pixel=10, h_pixel=10):
        """바운딩 박스 크기에 비례하는 거대한 안전 영역(중심 30%)의 depth 값을 추출합니다."""
        frame = self._wait_for_valid_data(self.img_node.get_depth_frame, "depth frame")
        
        try:
            img_h, img_w = frame.shape
            
            roi_w = max(5, int(w_pixel * 0.3))
            roi_h = max(5, int(h_pixel * 0.3))
            
            y_min = max(0, cy - roi_h // 2)
            y_max = min(img_h, cy + roi_h // 2 + 1)
            x_min = max(0, cx - roi_w // 2)
            x_max = min(img_w, cx + roi_w // 2 + 1)
            
            roi = frame[y_min:y_max, x_min:x_max]
            valid_depths = roi[roi > 0]
            
            if len(valid_depths) > 0:
                return float(np.median(valid_depths))
            else:
                self.get_logger().warn(f"⚠️ 중앙 {roi_w}x{roi_h} 영역 전체가 0입니다! (극심한 빛 반사)")
                return 0.0
                
        except Exception as e:
            self.get_logger().error(f"❌ Depth 추출 중 에러 발생: {e}")
            return None
            
    def _wait_for_valid_data(self, getter, description):
        """getter 함수가 유효한 데이터를 반환할 때까지 spin 하며 재시도합니다."""
        data = getter()
        while data is None or (isinstance(data, np.ndarray) and not data.any()):
            rclpy.spin_once(self.img_node)
            self.get_logger().info(f"Retry getting {description}.")
            data = getter()
        return data

    def _pixel_to_camera_coords(self, x, y, z):
        """픽셀 좌표와 intrinsics를 이용해 카메라 좌표계로 변환합니다."""
        fx = self.intrinsics['fx']
        fy = self.intrinsics['fy']
        ppx = self.intrinsics['ppx']
        ppy = self.intrinsics['ppy']
        return (
            (x - ppx) * z / fx,
            (y - ppy) * z / fy,
            z
        )


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()